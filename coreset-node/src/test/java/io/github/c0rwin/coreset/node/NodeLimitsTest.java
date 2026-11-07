package io.github.c0rwin.coreset.node;

import static io.github.c0rwin.coreset.node.IngestResult.Status.ACCEPTED;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** Configurable {@link NodeLimits} enforced by {@link CoresetNode} itself, before any allocation. */
class NodeLimitsTest {

    private static final Duration WAIT = Duration.ofSeconds(30);
    private CoresetNode node;

    @AfterEach
    void close() {
        if (node != null) {
            node.close();
        }
    }

    private CoresetNode node(NodeLimits limits, Map<String, CoresetAlgorithm> extra) {
        node = new CoresetNode(new TestResolver(extra), MetricsSink.NOOP, Duration.ofMillis(200), limits);
        return node;
    }

    private static SessionSpec spec(int workers) {
        return new SessionSpec("identity", Map.of(), 1L, workers, 2);
    }

    @Test
    void sessionsAndWorkersAreReservedAndReleased() {
        CoresetNode n = node(NodeLimits.builder().maxSessions(2).maxTotalWorkers(4).build(), Map.of());
        assertTrue(n.createSession("a", spec(3)));
        NodeLimitException workers = assertThrows(NodeLimitException.class, () -> n.createSession("b", spec(2)));
        assertTrue(workers.getMessage().contains("node limit of 4"), workers.getMessage());
        assertTrue(n.createSession("b", spec(1)));
        assertThrows(NodeLimitException.class, () -> n.createSession("c", spec(1)));
        assertTrue(n.closeSession("a"));
        assertTrue(n.createSession("c", spec(3)));
        // An idempotent re-create reserves nothing.
        assertEquals(false, n.createSession("c", spec(3)));
    }

    @Test
    void shapeLimitsAreCheckedBeforeAnythingIsResolved() {
        AtomicInteger resolved = new AtomicInteger();
        CoresetAlgorithm counting = (input, rng, metrics) -> input;
        CoresetNode n = new CoresetNode((id, params) -> {
            resolved.incrementAndGet();
            return counting;
        }, MetricsSink.NOOP, Duration.ofMillis(200), NodeLimits.builder().maxQueueCapacity(8).maxParams(1).build());
        node = n;
        assertThrows(IllegalArgumentException.class,
                () -> n.createSession("q", new SessionSpec("x", Map.of(), 1L, 1, 16)));
        assertThrows(IllegalArgumentException.class,
                () -> n.createSession("p", new SessionSpec("x", Map.of("a", "1", "b", "2"), 1L, 1, 2)));
        assertThrows(IllegalArgumentException.class, () -> n.createSession("bad id", spec(1)));
        assertEquals(0, resolved.get());
    }

    @Test
    void ingestRefusesHugeDimensionsAndBatches() {
        CoresetNode n = node(NodeLimits.builder().maxDim(8).maxPointsPerBatch(2).build(), Map.of());
        n.createSession("s", spec(1));
        SplittableRandom rng = new SplittableRandom(1);
        IngestResult dim = n.ingest("s", 0, TestData.batch(rng, 1, 9));
        assertEquals(FAILED, dim.status());
        assertTrue(dim.reason().contains("dim 9 exceeds"), dim.reason());
        IngestResult size = n.ingest("s", 0, TestData.batch(rng, 3, 8));
        assertEquals(FAILED, size.status());
        assertTrue(size.reason().contains("3 points exceed"), size.reason());
        assertEquals(ACCEPTED, n.ingest("s", 0, TestData.batch(rng, 2, 8)).status());
    }

    @Test
    void costIsBudgetedAndCancellable() {
        CoresetNode n = node(NodeLimits.builder().maxCenters(2).maxCostWork(9_999).build(), Map.of());
        SparseVector origin = SparseVector.dense(new double[] {0});
        List<WeightedPoint> points = new ArrayList<>();
        for (int i = 0; i < 5000; i++) {
            points.add(new WeightedPoint(SparseVector.dense(new double[] {1}), 1));
        }
        assertThrows(IllegalArgumentException.class, () -> n.partialCost(List.of(origin, origin, origin), points));
        assertThrows(NodeLimitException.class, () -> n.partialCost(List.of(origin, origin), points));
        assertEquals(new PartialCost(5000, 5000, 5000), n.partialCost(List.of(origin), points));
        AtomicInteger polls = new AtomicInteger();
        CancellationException e = assertThrows(CancellationException.class,
                () -> n.partialCost(List.of(origin), points, () -> polls.incrementAndGet() > 1));
        assertTrue(e.getMessage().contains("after " + PartialCost.CANCEL_CHECK_EVERY + " of 5000"), e.getMessage());
    }

    @Test
    void oversizedSnapshotIsRefusedBeforeTheUnionIsBuilt() throws Exception {
        CoresetNode n = node(NodeLimits.builder().maxSnapshotPoints(3).build(), Map.of());
        n.createSession("big", spec(1));
        assertEquals(ACCEPTED, n.ingest("big", 0, TestData.batch(new SplittableRandom(2), 4, 4)).status());
        assertTrue(n.awaitApplied("big", WAIT));
        NodeLimitException e = assertThrows(NodeLimitException.class, () -> n.snapshot("big"));
        assertTrue(e.getMessage().contains("4 points, above the limit of 3"), e.getMessage());
    }

    @Test
    void awaitIsCappedByMaxAwait() throws Exception {
        CoresetNode n = node(NodeLimits.builder().maxAwait(Duration.ofMillis(50)).build(),
                Map.of("stuck", (input, rng, metrics) -> {
                    try {
                        Thread.sleep(WAIT.toMillis());
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    return input;
                }));
        n.createSession("slow", new SessionSpec("stuck", Map.of(), 1L, 1, 2));
        n.ingest("slow", 0, TestData.batch(new SplittableRandom(3), 1, 4));
        long start = System.nanoTime();
        assertEquals(false, n.awaitApplied("slow", Duration.ofDays(365)));
        assertTrue(System.nanoTime() - start < Duration.ofSeconds(10).toNanos());
    }

    @Test
    void idleSessionsExpireWhileActiveOnesStay() throws Exception {
        CoresetNode n = node(NodeLimits.builder().sessionIdleTimeout(Duration.ofMillis(200)).build(), Map.of());
        n.createSession("idle", spec(1));
        n.createSession("active", spec(1));
        SplittableRandom rng = new SplittableRandom(5);
        long deadline = System.nanoTime() + Duration.ofSeconds(20).toNanos();
        for (long seq = 0; n.sessionIds().contains("idle"); seq++) {
            assertTrue(System.nanoTime() < deadline, "idle session never expired");
            assertEquals(ACCEPTED, n.ingest("active", seq, TestData.batch(rng, 1, 4)).status());
            Thread.sleep(20);
        }
        assertEquals(java.util.Set.of("active"), n.sessionIds());
        assertTrue(n.createSession("idle", spec(1)), "the expired session's reservation was released");
    }

    @Test
    void anErrorInOneSessionLeavesTheOthersServing() throws Exception {
        CoresetNode n = node(NodeLimits.defaults(), Map.of("oom", (input, rng, metrics) -> {
            throw new OutOfMemoryError("simulated");
        }));
        n.createSession("victim", new SessionSpec("oom", Map.of(), 1L, 1, 2));
        n.createSession("healthy", spec(2));
        SplittableRandom rng = new SplittableRandom(4);
        n.ingest("victim", 0, TestData.batch(rng, 1, 4));
        assertThrows(SessionFailedException.class, () -> n.awaitApplied("victim", WAIT));
        List<WeightedPoint> b = TestData.batch(rng, 3, 4);
        assertEquals(ACCEPTED, n.ingest("healthy", 0, b).status());
        assertTrue(n.awaitApplied("healthy", WAIT));
        assertEquals(TestData.weight(b), n.snapshot("healthy").coreset().totalWeight());
        assertEquals(SessionState.FAILED, n.snapshot("victim").state());
    }
}
