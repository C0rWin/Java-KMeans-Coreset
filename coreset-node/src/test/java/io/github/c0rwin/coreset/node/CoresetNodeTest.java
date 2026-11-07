package io.github.c0rwin.coreset.node;

import static io.github.c0rwin.coreset.node.IngestResult.Status.ACCEPTED;
import static io.github.c0rwin.coreset.node.IngestResult.Status.BUSY;
import static io.github.c0rwin.coreset.node.IngestResult.Status.DUPLICATE;
import static io.github.c0rwin.coreset.node.IngestResult.Status.FAILED;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSnapshot;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.node.testing.TestData;
import io.github.c0rwin.coreset.node.testing.TestResolver;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class CoresetNodeTest {

    private static final Duration WAIT = Duration.ofSeconds(30);
    private static final int DIM = 16;

    private final RecordingMetrics metrics = new RecordingMetrics();
    private CoresetNode node;

    @AfterEach
    void closeNode() {
        if (node != null) {
            node.close();
        }
    }

    private CoresetNode node(Map<String, io.github.c0rwin.coreset.core.CoresetAlgorithm> extra) {
        node = new CoresetNode(new TestResolver(extra), metrics, Duration.ofMillis(200));
        return node;
    }

    private static SessionSpec spec(String algorithm, int workers, int capacity, TreeMode mode) {
        return new SessionSpec(algorithm, Map.of(), 42L, workers, capacity, mode);
    }

    @Test
    void retriedSeqIsAcknowledgedWithoutDoubleCounting() throws Exception {
        CoresetNode n = node(Map.of());
        n.createSession("retry", spec("identity", 2, 16, TreeMode.SINGLE_WRITER));
        SplittableRandom rng = new SplittableRandom(1);
        List<List<WeightedPoint>> batches = new ArrayList<>();
        double total = 0;
        for (int seq = 0; seq < 10; seq++) {
            List<WeightedPoint> b = TestData.batch(rng, 5, DIM);
            batches.add(b);
            total += TestData.weight(b);
        }
        // Out of order, every seq sent twice, plus late retries after the dedupe floor has compacted them.
        int[] order = {3, 0, 1, 2, 9, 4, 5, 6, 7, 8};
        for (int seq : order) {
            assertEquals(IngestResult.accepted(5), n.ingest("retry", seq, batches.get(seq)));
            assertEquals(DUPLICATE, n.ingest("retry", seq, batches.get(seq)).status());
        }
        assertTrue(n.awaitApplied("retry", WAIT));
        for (int seq : order) {
            assertEquals(DUPLICATE, n.ingest("retry", seq, batches.get(seq)).status());
        }
        NodeSnapshot s = n.snapshot("retry");
        assertEquals(10, s.acceptedBatches());
        assertEquals(10, s.appliedBatches());
        assertEquals(total, s.acceptedWeight());
        assertEquals(total, s.coreset().totalWeight());
        assertEquals(50, s.coreset().size());
        assertTrue(s.complete());
        assertEquals(20, metrics.snapshot().counters().get(NodeMetricNames.INGEST_DUPLICATES));
    }

    @Test
    void fullQueueReturnsBusyThenAcceptsOnceDrained() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(TestAlgorithms.IDENTITY);
        CoresetNode n = node(Map.of("gated", gated));
        n.createSession("busy", spec("gated", 1, 2, TreeMode.SINGLE_WRITER));
        SplittableRandom rng = new SplittableRandom(2);
        List<WeightedPoint> b0 = TestData.batch(rng, 3, DIM);
        assertEquals(ACCEPTED, n.ingest("busy", 0, b0).status());
        gated.awaitEntered(); // the worker holds seq 0, so the ring is empty
        assertEquals(ACCEPTED, n.ingest("busy", 1, TestData.batch(rng, 3, DIM)).status());
        assertEquals(ACCEPTED, n.ingest("busy", 2, TestData.batch(rng, 3, DIM)).status());
        List<WeightedPoint> b3 = TestData.batch(rng, 3, DIM);
        for (int attempt = 0; attempt < 100; attempt++) {
            IngestResult r = n.ingest("busy", 3, b3);
            assertEquals(BUSY, r.status());
            assertEquals("worker queue full", r.reason());
        }
        NodeSnapshot blocked = n.snapshot("busy");
        // Bounded memory: one batch in the worker plus queueCapacity in the ring, nothing more.
        assertEquals(3, blocked.acceptedBatches());
        assertEquals(0, blocked.appliedBatches());
        assertFalse(blocked.complete());
        assertTrue(blocked.pendingWeight() > 0);
        assertFalse(n.awaitApplied("busy", Duration.ofMillis(20)));

        gated.release();
        IngestResult r = assertTimeoutPreemptively(WAIT, () -> {
            IngestResult last;
            while ((last = n.ingest("busy", 3, b3)).status() == BUSY) {
                Thread.onSpinWait();
            }
            return last;
        });
        assertEquals(ACCEPTED, r.status());
        assertTrue(n.awaitApplied("busy", WAIT));
        NodeSnapshot s = n.snapshot("busy");
        assertEquals(4, s.appliedBatches());
        assertEquals(0.0, s.pendingWeight());
        assertTrue(s.complete());
        assertTrue(metrics.snapshot().counters().get(NodeMetricNames.INGEST_BUSY) >= 100);
    }

    @ParameterizedTest
    @EnumSource(TreeMode.class)
    void snapshotWeightEqualsIngestedWeight(TreeMode mode) throws Exception {
        CoresetNode n = node(Map.of());
        String id = "weight-" + mode.id();
        n.createSession(id, spec("merge-equal", 4, 64, mode));
        SplittableRandom rng = new SplittableRandom(3);
        double total = 0;
        int seq = 0;
        while (seq < 1000) {
            List<WeightedPoint> b = TestData.batch(rng, 1 + rng.nextInt(8), DIM);
            IngestResult r = n.ingest(id, seq, b);
            if (r.status() == ACCEPTED) {
                total += TestData.weight(b);
                seq++;
            } else {
                assertEquals(BUSY, r.status());
                Thread.onSpinWait();
            }
        }
        assertTrue(n.awaitApplied(id, WAIT));
        NodeSnapshot s = n.snapshot(id);
        assertEquals(1000, s.acceptedBatches());
        assertEquals(1000, s.appliedBatches());
        assertEquals(total, s.acceptedWeight());
        assertEquals(total, s.appliedWeight());
        assertEquals(total, s.coreset().totalWeight());
        assertTrue(s.consistent());
        assertTrue(s.complete());
    }

    @Test
    void batchCoresetIsReproducedWhenItsSeqIsRetried() throws Exception {
        TestAlgorithms.Recording direct = new TestAlgorithms.Recording(TestAlgorithms.sample(3));
        TestAlgorithms.Recording shifted = new TestAlgorithms.Recording(TestAlgorithms.sample(3));
        TestAlgorithms.Recording retried = new TestAlgorithms.Recording(TestAlgorithms.sample(3));
        TestAlgorithms.Gated gate = new TestAlgorithms.Gated(retried);
        CoresetNode n = node(Map.of("direct", direct, "shifted", shifted, "retried", gate));
        SplittableRandom rng = new SplittableRandom(4);
        List<WeightedPoint> x = TestData.batch(rng, 20, DIM);

        n.createSession("direct", spec("direct", 1, 2, TreeMode.SINGLE_WRITER));
        assertEquals(ACCEPTED, n.ingest("direct", 5, x).status());
        assertTrue(n.awaitApplied("direct", WAIT));

        n.createSession("shifted", spec("shifted", 1, 2, TreeMode.SINGLE_WRITER));
        assertEquals(ACCEPTED, n.ingest("shifted", 6, x).status());
        assertTrue(n.awaitApplied("shifted", WAIT));

        // Different worker count, other batches first, and seq 5 rejected BUSY before its retry is accepted.
        n.createSession("retried", spec("retried", 3, 2, TreeMode.SINGLE_WRITER));
        assertEquals(ACCEPTED, n.ingest("retried", 2, TestData.batch(rng, 20, DIM)).status());
        gate.awaitEntered();
        assertEquals(ACCEPTED, n.ingest("retried", 8, TestData.batch(rng, 20, DIM)).status());
        assertEquals(ACCEPTED, n.ingest("retried", 11, TestData.batch(rng, 20, DIM)).status());
        assertEquals(BUSY, n.ingest("retried", 5, x).status());
        gate.release();
        assertTimeoutPreemptively(WAIT, () -> {
            while (n.ingest("retried", 5, x).status() == BUSY) {
                Thread.onSpinWait();
            }
        });
        assertTrue(n.awaitApplied("retried", WAIT));

        Coreset expected = direct.outputFor(x);
        assertEquals(3, expected.size());
        assertEquals(expected, retried.outputFor(x));
        assertNotEquals(expected, shifted.outputFor(x)); // the seq, not just the seed, drives the draw
    }

    @ParameterizedTest
    @EnumSource(TreeMode.class)
    void concurrentProducersGetExactTotals(TreeMode mode) throws Exception {
        CoresetNode n = node(Map.of());
        String id = "concurrent-" + mode.id();
        n.createSession(id, spec("identity", 4, 32, mode));
        int producers = 8;
        int perProducer = 500;
        CyclicBarrier start = new CyclicBarrier(producers);
        ExecutorService pool = Executors.newFixedThreadPool(producers);
        try {
            List<Future<double[]>> results = new ArrayList<>();
            for (int t = 0; t < producers; t++) {
                final int producer = t;
                results.add(pool.submit(() -> {
                    SplittableRandom rng = new SplittableRandom(100 + producer);
                    long accepted = 0;
                    long duplicates = 0;
                    double weight = 0;
                    start.await();
                    for (int i = 0; i < perProducer; i++) {
                        long seq = (long) i * producers + producer; // producers partition the seq space
                        List<WeightedPoint> b = TestData.batch(rng, 3, DIM);
                        for (int send = 0; send < 2; send++) { // every batch is retried once
                            IngestResult r;
                            while ((r = n.ingest(id, seq, b)).status() == BUSY) {
                                Thread.onSpinWait();
                            }
                            if (r.status() == ACCEPTED) {
                                accepted++;
                                weight += TestData.weight(b);
                            } else {
                                assertEquals(DUPLICATE, r.status());
                                duplicates++;
                            }
                        }
                    }
                    return new double[] {accepted, duplicates, weight};
                }));
            }
            double accepted = 0;
            double duplicates = 0;
            double weight = 0;
            for (Future<double[]> f : results) {
                double[] r = f.get(60, TimeUnit.SECONDS);
                accepted += r[0];
                duplicates += r[1];
                weight += r[2];
            }
            assertEquals(producers * perProducer, accepted);
            assertEquals(producers * perProducer, duplicates);
            assertTrue(n.awaitApplied(id, WAIT));
            NodeSnapshot s = n.snapshot(id);
            assertEquals(producers * perProducer, s.acceptedBatches());
            assertEquals(producers * perProducer, s.appliedBatches());
            assertEquals(producers * perProducer * 3, s.coreset().size());
            assertEquals(weight, s.coreset().totalWeight());
            assertEquals(weight, s.acceptedWeight());
            assertTrue(s.complete());
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    void algorithmFailureSurfacesAsFailedSession() throws Exception {
        CoresetNode n = node(Map.of());
        n.createSession("fail", new SessionSpec("failing", Map.of("poison", "13"), 1L, 2, 4));
        SparseVector v = SparseVector.dense(new double[] {1, 2});
        assertEquals(ACCEPTED, n.ingest("fail", 0, List.of(new WeightedPoint(v, 1))).status());
        assertEquals(ACCEPTED, n.ingest("fail", 1, List.of(new WeightedPoint(v, 13))).status());

        SessionFailedException e = assertThrows(SessionFailedException.class, () -> n.awaitApplied("fail", WAIT));
        assertTrue(e.getCause().getMessage().contains("poisoned batch"), e.getMessage());
        NodeSnapshot s = n.snapshot("fail");
        assertEquals(SessionState.FAILED, s.state());
        assertTrue(s.failure().contains("poisoned batch"), s.failure());
        assertFalse(s.complete());
        IngestResult r = n.ingest("fail", 2, List.of(new WeightedPoint(v, 1)));
        assertEquals(FAILED, r.status());
        assertTrue(r.reason().contains("poisoned batch"), r.reason());
        assertEquals(1, metrics.snapshot().counters().get(NodeMetricNames.SESSION_FAILURES));
        for (Thread t : TestData.liveWorkers("fail")) {
            t.join(WAIT.toMillis()); // failed workers exit on their own, before close
        }
        assertEquals(List.of(), TestData.liveWorkers("fail"));
    }

    @Test
    void closeStopsEveryWorkerThread() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(TestAlgorithms.IDENTITY);
        CoresetNode n = node(Map.of("gated", gated));
        SplittableRandom rng = new SplittableRandom(5);
        for (TreeMode mode : TreeMode.values()) {
            String id = "shutdown-" + mode.id();
            n.createSession(id, spec("identity", 4, 8, mode));
            for (int seq = 0; seq < 20; seq++) {
                n.ingest(id, seq, TestData.batch(rng, 2, DIM));
            }
        }
        n.createSession("shutdown-idle", spec("identity", 3, 2, TreeMode.SINGLE_WRITER));
        // A worker stuck inside the algorithm is interrupted after the grace period at the latest.
        n.createSession("shutdown-stuck", spec("gated", 1, 2, TreeMode.SINGLE_WRITER));
        n.ingest("shutdown-stuck", 0, TestData.batch(rng, 2, DIM));
        gated.awaitEntered();
        assertEquals(4 + 4 + 3 + 1, TestData.liveWorkers("shutdown-").size());

        assertTimeoutPreemptively(Duration.ofSeconds(60), n::close);
        assertEquals(List.of(), TestData.liveWorkers("shutdown-"));
        assertEquals(java.util.Set.of(), n.sessionIds());
        assertThrows(IllegalStateException.class, () -> n.createSession("late", spec("identity", 1, 2,
                TreeMode.SINGLE_WRITER)));
        assertEquals(FAILED, n.ingest("shutdown-idle", 0, TestData.batch(rng, 1, DIM)).status());
    }

    @Test
    void casModeMatchesSingleWriterTotals() throws Exception {
        CoresetNode n = node(Map.of());
        double[] totals = new double[2];
        for (TreeMode mode : TreeMode.values()) {
            String id = "parity-" + mode.id();
            n.createSession(id, spec("merge-equal", 4, 16, mode));
            SplittableRandom rng = new SplittableRandom(6);
            for (int seq = 0; seq < 300; seq++) {
                List<WeightedPoint> b = TestData.batch(rng, 4, DIM);
                while (n.ingest(id, seq, b).status() == BUSY) {
                    Thread.onSpinWait();
                }
            }
            assertTrue(n.awaitApplied(id, WAIT));
            NodeSnapshot s = n.snapshot(id);
            assertTrue(s.complete());
            totals[mode.ordinal()] = s.coreset().totalWeight();
        }
        assertEquals(totals[0], totals[1]);
    }

    @Test
    void sessionLifecycleAndValidation() throws Exception {
        CoresetNode n = node(Map.of());
        SessionSpec spec = spec("identity", 1, 2, TreeMode.SINGLE_WRITER);
        assertTrue(n.createSession("life", spec));
        assertFalse(n.createSession("life", spec)); // idempotent create
        assertThrows(IllegalStateException.class,
                () -> n.createSession("life", spec("identity", 2, 2, TreeMode.SINGLE_WRITER)));
        assertThrows(UnknownAlgorithmException.class,
                () -> n.createSession("other", spec("nope", 1, 2, TreeMode.SINGLE_WRITER)));
        assertThrows(IllegalArgumentException.class, () -> n.createSession(" ", spec));
        assertThrows(IllegalArgumentException.class, () -> new SessionSpec("identity", Map.of(), 1, 1, 3));
        assertThrows(IllegalArgumentException.class, () -> new SessionSpec("identity", Map.of(), 1, 0, 2));
        assertEquals(spec, n.spec("life"));

        WeightedPoint p2 = new WeightedPoint(SparseVector.dense(new double[] {1, 2}), 1);
        WeightedPoint p3 = new WeightedPoint(SparseVector.dense(new double[] {1, 2, 3}), 1);
        assertEquals(FAILED, n.ingest("missing", 0, List.of(p2)).status());
        assertEquals(FAILED, n.ingest("life", -1, List.of(p2)).status());
        assertEquals(FAILED, n.ingest("life", 0, List.of(p2, p3)).status());
        assertEquals(ACCEPTED, n.ingest("life", 0, List.of(p2)).status());
        IngestResult wrongDim = n.ingest("life", 1, List.of(p3));
        assertEquals(FAILED, wrongDim.status());
        assertTrue(wrongDim.reason().contains("session dimension 2"), wrongDim.reason());
        assertEquals(ACCEPTED, n.ingest("life", 1, List.of()).status()); // empty batches count as applied

        assertTrue(n.awaitApplied("life", WAIT));
        assertEquals(2, n.snapshot("life").appliedBatches());
        assertTrue(n.closeSession("life"));
        assertFalse(n.closeSession("life"));
        assertThrows(UnknownSessionException.class, () -> n.snapshot("life"));
        assertThrows(UnknownSessionException.class, () -> n.awaitApplied("life", WAIT));
    }

    @Test
    void partialCostMatchesCoreCost() throws Exception {
        CoresetNode n = node(Map.of());
        SparseVector a = SparseVector.dense(new double[] {0, 0});
        SparseVector b = SparseVector.dense(new double[] {3, 4});
        List<WeightedPoint> pts = List.of(new WeightedPoint(a, 2), new WeightedPoint(b, 3));
        List<SparseVector> center = List.of(SparseVector.dense(new double[] {0, 0}));
        assertEquals(3 * 25.0, n.partialCost(center, pts));

        n.createSession("cost", spec("identity", 2, 2, TreeMode.SINGLE_WRITER));
        n.ingest("cost", 0, pts);
        assertTrue(n.awaitApplied("cost", WAIT));
        assertEquals(75.0, n.partialCost("cost", center));
    }

    @Test
    void emitsNodeMetrics() throws Exception {
        CoresetNode n = node(Map.of());
        n.createSession("metrics", spec("merge-equal", 2, 4, TreeMode.SINGLE_WRITER));
        SplittableRandom rng = new SplittableRandom(7);
        for (int seq = 0; seq < 8; seq++) {
            List<WeightedPoint> b = TestData.batch(rng, 3, DIM);
            while (n.ingest("metrics", seq, b).status() == BUSY) {
                Thread.onSpinWait();
            }
        }
        n.ingest("metrics", 0, TestData.batch(rng, 3, DIM));
        assertTrue(n.awaitApplied("metrics", WAIT));
        n.snapshot("metrics");
        MetricsSnapshot m = metrics.snapshot();
        assertEquals(8, m.counters().get(NodeMetricNames.INGEST_ACCEPTED));
        assertEquals(24, m.counters().get(NodeMetricNames.INGEST_ACCEPTED_POINTS));
        assertEquals(1, m.counters().get(NodeMetricNames.INGEST_DUPLICATES));
        assertEquals(1, m.counters().get(NodeMetricNames.SESSIONS_CREATED));
        assertTrue(m.distributions().get(NodeMetricNames.INGEST_LATENCY).count() >= 9);
        assertEquals(8, m.distributions().get(NodeMetricNames.QUEUE_DEPTH).count());
        assertEquals(8, m.distributions().get(NodeMetricNames.BATCH_COMPRESS + ".wall_ns").count());
        // Seqs alternate between the two workers: 4 insertions each, tree height 3 after the 4th.
        assertEquals(4, m.distributions().get(NodeMetricNames.workerTreeDepth(0)).count());
        assertEquals(3.0, m.distributions().get(NodeMetricNames.workerTreeDepth(1)).max());
        assertEquals(1, m.distributions().get(NodeMetricNames.SNAPSHOT_POINTS).count());
        assertTrue(m.distributions().get(NodeMetricNames.SNAPSHOT_BYTES).max() > 0);
    }
}
