package io.github.c0rwin.coreset.node;

import static io.github.c0rwin.coreset.node.IngestResult.Status.ACCEPTED;
import static io.github.c0rwin.coreset.node.IngestResult.Status.BUSY;
import static io.github.c0rwin.coreset.node.IngestResult.Status.FAILED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.node.testing.TestData;
import io.github.c0rwin.coreset.node.testing.TestResolver;
import java.time.Duration;
import java.util.AbstractList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** Interleavings between ingest, close, failure, snapshots and the dedupe window (review findings F5-F9). */
class SessionRacesTest {

    private static final Duration WAIT = Duration.ofSeconds(30);
    private static final int DIM = 4;

    private CoresetNode node;
    private ExecutorService pool;

    @AfterEach
    void tearDown() {
        if (pool != null) {
            pool.shutdownNow();
        }
        if (node != null) {
            node.close();
        }
    }

    private CoresetNode node(Map<String, CoresetAlgorithm> extra) {
        node = new CoresetNode(new TestResolver(extra), MetricsSink.NOOP, Duration.ofMillis(200));
        return node;
    }

    /**
     * A batch whose first element read (the payload checks and copy into the batch block, which run after the state
     * checks and before the enqueue) runs {@code during} once.
     */
    private static List<WeightedPoint> copyRuns(List<WeightedPoint> batch, Runnable during) {
        return new AbstractList<>() {
            private boolean ran;

            @Override
            public WeightedPoint get(int i) {
                if (!ran) {
                    ran = true;
                    during.run();
                }
                return batch.get(i);
            }

            @Override
            public int size() {
                return batch.size();
            }
        };
    }

    @Test
    void ingestThatOverlapsCloseIsNotAcknowledgedAccepted() {
        CoresetNode n = node(Map.of());
        n.createSession("closing", new SessionSpec("identity", Map.of(), 1L, 1, 4));
        List<WeightedPoint> b = TestData.batch(new SplittableRandom(1), 2, DIM);
        IngestResult r = n.ingest("closing", 0, copyRuns(b, () -> n.closeSession("closing")));
        assertEquals(FAILED, r.status(), r.toString());
    }

    @Test
    void ingestThatOverlapsSessionFailureIsNotAcknowledgedAccepted() {
        CoresetNode n = node(Map.of());
        n.createSession("failing", new SessionSpec("failing", Map.of("poison", "13"), 1L, 1, 4));
        SparseVector v = SparseVector.dense(new double[] {1, 2, 3, 4});
        List<WeightedPoint> b = List.of(new WeightedPoint(v, 1));
        IngestResult r = n.ingest("failing", 1, copyRuns(b, () -> {
            n.ingest("failing", 0, List.of(new WeightedPoint(v, 13)));
            long deadline = System.nanoTime() + WAIT.toNanos();
            while (n.snapshot("failing").state() != SessionState.FAILED && System.nanoTime() < deadline) {
                LockSupport.parkNanos(100_000);
            }
        }));
        assertEquals(SessionState.FAILED, n.snapshot("failing").state());
        assertEquals(FAILED, r.status(), r.toString());
    }

    @Test
    void createSessionThatOverlapsNodeCloseFails() throws Exception {
        pool = Executors.newSingleThreadExecutor();
        // Starting 1024 workers usually takes far longer than noticing the registration and closing the node; a
        // descheduled poller can still miss that window, so try a few times. A miss returns true, which is legal.
        int overlapped = 0;
        for (int attempt = 0; attempt < 20 && overlapped == 0; attempt++) {
            CoresetNode n = new CoresetNode(new TestResolver(Map.of()), MetricsSink.NOOP, Duration.ofMillis(200));
            try {
                Future<Boolean> created = pool.submit(() -> n.createSession("late",
                        new SessionSpec("identity", Map.of(), 1L, SessionSpec.MAX_WORKERS, 2)));
                while (!n.sessionIds().contains("late") && !created.isDone()) {
                    Thread.onSpinWait();
                }
                n.close();
                try {
                    assertTrue(created.get(30, TimeUnit.SECONDS));
                } catch (ExecutionException e) {
                    assertTrue(e.getCause() instanceof IllegalStateException, e.getCause().toString());
                    assertEquals("node closed", e.getCause().getMessage());
                    overlapped++;
                }
            } finally {
                n.close();
            }
        }
        assertEquals(1, overlapped, "never observed createSession overlapping close in 20 attempts");
    }

    @Test
    void snapshotNeverReportsMoreAppliedThanAcceptedWeight() throws Exception {
        CoresetNode n = node(Map.of());
        n.createSession("order", new SessionSpec("identity", Map.of(), 1L, 1, 1024));
        pool = Executors.newSingleThreadExecutor();
        AtomicBoolean stop = new AtomicBoolean();
        WeightedPoint p = new WeightedPoint(SparseVector.dense(new double[] {1, 0, 0, 0}), 1);
        // Bounded so a fast host cannot outgrow the identity session's snapshot limit within the 3 s window.
        long cap = NodeLimits.defaults().maxSnapshotPoints() / 2;
        Future<?> producer = pool.submit(() -> {
            for (long seq = 0; !stop.get() && seq < cap; ) {
                IngestResult r = n.ingest("order", seq, List.of(p));
                if (r.status() == ACCEPTED) {
                    seq++;
                } else if (r.status() != BUSY) {
                    throw new AssertionError(r.toString());
                }
            }
        });
        try {
            long deadline = System.nanoTime() + Duration.ofSeconds(3).toNanos();
            while (System.nanoTime() < deadline) {
                NodeSnapshot s = n.snapshot("order");
                assertTrue(s.appliedWeight() <= s.acceptedWeight(),
                        "applied " + s.appliedWeight() + " > accepted " + s.acceptedWeight());
            }
        } finally {
            stop.set(true);
            producer.get(30, TimeUnit.SECONDS);
        }
    }

    @Test
    void seqsFarAheadOfTheCommittedFloorAreRejected() {
        CoresetNode n = node(Map.of());
        n.createSession("window", new SessionSpec("identity", Map.of(), 1L, 1, 4));
        SplittableRandom rng = new SplittableRandom(4);
        int window = NodeLimits.defaults().maxSeqWindow();
        IngestResult far = n.ingest("window", window, TestData.batch(rng, 1, DIM));
        assertEquals(FAILED, far.status());
        assertTrue(far.reason().contains("dedupe window"), far.reason());
        assertEquals(ACCEPTED, n.ingest("window", window - 1, TestData.batch(rng, 1, DIM)).status());
        // Committing seq 0 moves the floor, and with it the window.
        assertEquals(ACCEPTED, n.ingest("window", 0, TestData.batch(rng, 1, DIM)).status());
        assertEquals(ACCEPTED, n.ingest("window", window, TestData.batch(rng, 1, DIM)).status());
    }

    /** Identity whose merges (inputs of 2+ points) take a while, so CAS carries are almost always in flight. */
    private static final CoresetAlgorithm SLOW_MERGE = (input, rng, metrics) -> {
        if (input.size() > 1) {
            LockSupport.parkNanos(2_000_000);
        }
        return input;
    };

    @Test
    void casAwaitAndSnapshotSucceedUnderSustainedIngest() throws Exception {
        // Generous read wait: the test is about liveness, not about how fast a loaded host drains the carries.
        node = new CoresetNode(new TestResolver(Map.of("slow-merge", SLOW_MERGE)), MetricsSink.NOOP,
                Duration.ofMillis(200), NodeLimits.builder().casSnapshotWait(Duration.ofSeconds(30)).build());
        CoresetNode n = node;
        n.createSession("busy", new SessionSpec("slow-merge", Map.of(), 1L, 4, 8, TreeMode.CAS));
        pool = Executors.newSingleThreadExecutor();
        AtomicBoolean stop = new AtomicBoolean();
        WeightedPoint p = new WeightedPoint(SparseVector.dense(new double[] {1, 0, 0, 0}), 1);
        Future<?> producer = pool.submit(() -> {
            for (long seq = 0; !stop.get(); ) {
                if (n.ingest("busy", seq, List.of(p)).status() == ACCEPTED) {
                    seq++;
                } else {
                    Thread.onSpinWait();
                }
            }
        });
        try {
            LockSupport.parkNanos(Duration.ofMillis(200).toNanos()); // let the carries pile up
            assertTrue(n.awaitApplied("busy", Duration.ofSeconds(30)), "awaitApplied starved by other carries");
            NodeSnapshot s = n.snapshot("busy");
            assertTrue(s.consistent(), "CAS snapshot starved by other carries");
            assertEquals(s.appliedBatches(), s.coreset().totalWeight(), 1e-9);
        } finally {
            stop.set(true);
            producer.get(30, TimeUnit.SECONDS);
        }
    }
}
