package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.node.testing.TestData;
import io.grpc.ForwardingServerCall;
import io.grpc.ForwardingServerCallListener;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class CoresetCoordinatorTest {

    private static final int DIM = 8;
    private static final Duration WAIT = Duration.ofSeconds(30);

    private static CoordinatorOptions options(int inFlight) {
        return new CoordinatorOptions(inFlight, WAIT, Duration.ofMinutes(5), Duration.ofMillis(1),
                Duration.ofMillis(20), 5, CoordinatorOptions.DEFAULT_MAX_MESSAGE_BYTES, WAIT, WAIT, 1 << 24, 2);
    }

    private static double ingestAll(IngestStream stream, SplittableRandom rng, int batches, List<WeightedPoint> all)
            throws Exception {
        double weight = 0;
        for (int i = 0; i < batches; i++) {
            List<WeightedPoint> b = TestData.batch(rng, 1 + rng.nextInt(6), DIM);
            weight += TestData.weight(b);
            all.addAll(b);
            stream.submit(b);
        }
        return weight;
    }

    @Test
    void distributesGathersClustersAndEvaluatesCost() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(2, Map.of(), 16);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(8))) {
            coordinator.createSession("s", new SessionSpec("merge-equal", Map.of(), 9L, 2, 8, TreeMode.CAS));
            coordinator.createSession("s", new SessionSpec("merge-equal", Map.of(), 9L, 2, 8, TreeMode.CAS));
            List<WeightedPoint> all = new ArrayList<>();
            double weight;
            try (IngestStream stream = coordinator.openIngest("s")) {
                weight = ingestAll(stream, new SplittableRandom(1), 200, all);
                stream.flush(WAIT);
                assertEquals(200, stream.acceptedBatches());
            }
            GatheredSnapshot snap = coordinator.gather("s", WAIT);
            assertTrue(snap.complete());
            assertEquals(200, snap.acceptedBatches());
            for (NodeSnapshot n : snap.nodes()) {
                assertEquals(100, n.appliedBatches()); // round-robin
            }
            assertEquals(weight, snap.coreset().totalWeight());

            KMeansResult km = CoresetCoordinator.cluster(snap.coreset(), 3, 20, 5L, null);
            assertEquals(3, km.centers().size());
            KMeansResult again = CoresetCoordinator.cluster(snap.coreset(), 3, 20, 5L, TestAlgorithms.IDENTITY);
            assertEquals(km.centers(), again.centers()); // seeded and deterministic

            double expected = Cost.cost(snap.coreset(), km.centers());
            assertEquals(expected, coordinator.cost("s", km.centers()), 1e-9 * expected);
            // merge-equal only sums weights of equal points, so the coreset cost is the exact cost.
            double raw = Cost.cost(all, km.centers());
            assertEquals(raw, coordinator.cost(km.centers(), all, 64), 1e-9 * raw);
            assertEquals(raw, expected, 1e-9 * raw);

            assertEquals(2, coordinator.stats().size());
            coordinator.closeSession("s");
            assertEquals(java.util.Set.of(), cluster.nodes.get(0).sessionIds());
        }
    }

    @Test
    void consecutiveIngestRunsIntoOneKeptSessionBothLand() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(2, Map.of(), 16);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(8))) {
            coordinator.createSession("kept", new SessionSpec("merge-equal", Map.of(), 3L, 2, 8));
            double weight = 0;
            for (int run = 0; run < 2; run++) {
                try (IngestStream stream = coordinator.openIngest("kept")) {
                    weight += ingestAll(stream, new SplittableRandom(10 + run), 20, new ArrayList<>());
                    stream.flush(WAIT);
                    assertEquals(20, stream.acceptedBatches(), "run " + run);
                    assertEquals(0, stream.duplicateAcks(), "run " + run);
                }
            }
            GatheredSnapshot snap = coordinator.gather("kept", WAIT);
            assertTrue(snap.complete());
            assertEquals(40, snap.appliedBatches());
            assertEquals(weight, snap.coreset().totalWeight(), 1e-9 * weight);
        }
    }

    @Test
    void duplicateAckForABatchSentOnceFailsTheStream() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(4))) {
            coordinator.createSession("s", new SessionSpec("identity", Map.of(), 1L, 1, 4));
            try (IngestStream stream = coordinator.openIngest("s")) {
                List<WeightedPoint> b = TestData.batch(new SplittableRandom(7), 2, DIM);
                // Some other sender already used this producer id and seq 0 with different content.
                assertEquals(io.github.c0rwin.coreset.node.IngestResult.Status.ACCEPTED, cluster.nodes.get(0)
                        .ingest("s", stream.producerId(), 0, TestData.batch(new SplittableRandom(8), 3, DIM)).status());
                stream.submit(b);
                IngestFailedException e = assertThrows(IngestFailedException.class, () -> stream.flush(WAIT));
                assertTrue(e.getMessage().contains("as DUPLICATE although it was sent once"), e.getMessage());
                assertEquals(0, stream.duplicateAcks());
            }
        }
    }

    @Test
    void busyNodesAreRetriedWithTheSameSeqUntilAccepted() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(TestAlgorithms.IDENTITY);
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of("gated", gated), 16);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(8))) {
            coordinator.createSession("s", new SessionSpec("gated", Map.of(), 1L, 1, 2));
            List<WeightedPoint> all = new ArrayList<>();
            try (IngestStream stream = coordinator.openIngest("s")) {
                // The node can hold 1 batch in the worker + 2 in the ring; the other 5 must bounce as BUSY.
                double weight = ingestAll(stream, new SplittableRandom(2), 8, all);
                assertTimeoutPreemptively(WAIT, () -> {
                    while (stream.busyRetries() < 5) {
                        Thread.onSpinWait();
                    }
                });
                gated.release();
                stream.flush(WAIT);
                assertEquals(8, stream.acceptedBatches());
                assertTrue(cluster.nodes.get(0).awaitApplied("s", WAIT));
                NodeSnapshot s = cluster.nodes.get(0).snapshot("s");
                assertEquals(8, s.appliedBatches());
                assertEquals(weight, s.coreset().totalWeight());
            }
        }
    }

    @Test
    void persistentBusyFailsTheIngestInsteadOfRetryingForever() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(TestAlgorithms.IDENTITY);
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of("gated", gated), 16);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels,
                     options(8).withBusyAndSubmitLimits(Duration.ofMillis(300), WAIT))) {
            coordinator.createSession("s", new SessionSpec("gated", Map.of(), 1L, 1, 2));
            try (IngestStream stream = coordinator.openIngest("s")) {
                ingestAll(stream, new SplittableRandom(2), 8, new ArrayList<>());
                IngestFailedException e = assertThrows(IngestFailedException.class, () -> stream.flush(WAIT));
                assertTrue(e.getMessage().contains("answered BUSY"), e.getMessage());
            } finally {
                gated.release();
            }
        }
    }

    @Test
    void submitGivesUpWhenANodeStopsAcknowledging() throws Exception {
        ServerInterceptor silent = new ServerInterceptor() {
            @Override
            public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
                                                               ServerCallHandler<Q, R> next) {
                if (!call.getMethodDescriptor().getBareMethodName().equals("Ingest")) {
                    return next.startCall(call, headers);
                }
                call.request(Integer.MAX_VALUE);
                return new ServerCall.Listener<>() { // reads everything, answers nothing
                };
            }
        };
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16, silent);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels,
                     options(1).withBusyAndSubmitLimits(WAIT, Duration.ofMillis(300)));
             IngestStream stream = coordinator.openIngest("s")) {
            stream.submit(TestData.batch(new SplittableRandom(9), 1, DIM));
            IngestFailedException e = assertThrows(IngestFailedException.class,
                    () -> stream.submit(TestData.batch(new SplittableRandom(10), 1, DIM)));
            assertTrue(e.getMessage().contains("acknowledged nothing"), e.getMessage());
        }
    }

    /** Drops acks after the 3rd message of the first Ingest stream and kills it after the 5th. */
    private static final class FlakyIngest implements ServerInterceptor {
        final AtomicInteger streams = new AtomicInteger();

        @Override
        public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
                                                           ServerCallHandler<Q, R> next) {
            if (!call.getMethodDescriptor().getBareMethodName().equals("Ingest") || streams.getAndIncrement() > 0) {
                return next.startCall(call, headers);
            }
            AtomicInteger seen = new AtomicInteger();
            ServerCall<Q, R> lossy = new ForwardingServerCall.SimpleForwardingServerCall<>(call) {
                @Override
                public void sendMessage(R message) {
                    if (seen.get() <= 3) {
                        super.sendMessage(message);
                    }
                }
            };
            return new ForwardingServerCallListener.SimpleForwardingServerCallListener<>(
                    next.startCall(lossy, headers)) {
                @Override
                public void onMessage(Q message) {
                    int k = seen.incrementAndGet();
                    if (k > 5) {
                        return; // already closed: anything still queued is lost with the connection
                    }
                    super.onMessage(message);
                    if (k == 5) {
                        call.close(Status.UNAVAILABLE.withDescription("injected fault"), new Metadata());
                    }
                }
            };
        }
    }

    @Test
    void brokenStreamIsReopenedAndUnackedBatchesAreResentIdempotently() throws Exception {
        FlakyIngest flaky = new FlakyIngest();
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16, flaky);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(16))) {
            coordinator.createSession("s", new SessionSpec("identity", Map.of(), 1L, 2, 16));
            try (IngestStream stream = coordinator.openIngest("s")) {
                double weight = ingestAll(stream, new SplittableRandom(3), 12, new ArrayList<>());
                stream.flush(WAIT);
                assertTrue(stream.reconnects() >= 1);
                assertTrue(flaky.streams.get() >= 2);
                // Batches 4 and 5 were applied but their acks were dropped: the resend is acknowledged DUPLICATE.
                assertEquals(2, stream.duplicateAcks());
                assertEquals(10, stream.acceptedBatches());
                assertTrue(cluster.nodes.get(0).awaitApplied("s", WAIT));
                NodeSnapshot s = cluster.nodes.get(0).snapshot("s");
                assertEquals(12, s.appliedBatches());
                assertEquals(weight, s.coreset().totalWeight());
            }
        }
    }

    @Test
    void snapshotsCanBeGatheredWhileIngesting() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(3, Map.of(), 16);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(4))) {
            coordinator.createSession("live", new SessionSpec("merge-equal", Map.of(), 1L, 2, 4));
            ExecutorService producer = Executors.newSingleThreadExecutor();
            try (IngestStream stream = coordinator.openIngest("live")) {
                Future<Double> sent = producer.submit(() -> {
                    double w = ingestAll(stream, new SplittableRandom(4), 600, new ArrayList<>());
                    stream.flush(WAIT);
                    return w;
                });
                long lastApplied = 0;
                int snapshots = 0;
                while (!sent.isDone()) {
                    GatheredSnapshot g = coordinator.gather("live", Duration.ZERO);
                    assertTrue(g.appliedBatches() >= lastApplied);
                    assertTrue(g.appliedBatches() <= g.acceptedBatches());
                    lastApplied = g.appliedBatches();
                    snapshots++;
                }
                double weight = sent.get(60, TimeUnit.SECONDS);
                GatheredSnapshot done = coordinator.gather("live", WAIT);
                assertTrue(done.complete());
                assertEquals(600, done.appliedBatches());
                assertEquals(weight, done.coreset().totalWeight());
                assertTrue(snapshots >= 1);
            } finally {
                producer.shutdownNow();
            }
        }
    }

    @Test
    void permanentRejectionFailsTheStream() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(4))) {
            coordinator.createSession("s", new SessionSpec("identity", Map.of(), 1L, 1, 4));
            try (IngestStream stream = coordinator.openIngest("s")) {
                stream.submit(List.of(new WeightedPoint(SparseVector.dense(new double[] {1, 2}), 1)));
                stream.flush(WAIT);
                stream.submit(List.of(new WeightedPoint(SparseVector.dense(new double[] {1, 2, 3}), 1)));
                IngestFailedException e = assertThrows(IngestFailedException.class, () -> stream.flush(WAIT));
                assertTrue(e.getMessage().contains("session dimension 2"), e.getMessage());
                assertThrows(IngestFailedException.class, () -> stream.submit(List.of()));
            }
        }
    }

    @Test
    void invalidArgumentFailsTheStreamWithoutReconnecting() throws Exception {
        AtomicInteger ingestCalls = new AtomicInteger();
        ServerInterceptor rejectIngest = new ServerInterceptor() {
            @Override
            public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
                                                               ServerCallHandler<Q, R> next) {
                if (!call.getMethodDescriptor().getBareMethodName().equals("Ingest")) {
                    return next.startCall(call, headers);
                }
                ingestCalls.incrementAndGet();
                call.close(Status.INVALID_ARGUMENT.withDescription("bad seq"), new Metadata());
                return new ServerCall.Listener<>() {
                };
            }
        };
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16, rejectIngest);
             CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(4));
             IngestStream stream = coordinator.openIngest("s")) {
            stream.submit(TestData.batch(new SplittableRandom(6), 2, DIM));
            IngestFailedException e = assertThrows(IngestFailedException.class, () -> stream.flush(WAIT));
            assertTrue(e.getMessage().contains("rejected the stream: bad seq"), e.getMessage());
            assertEquals(0, stream.reconnects());
            assertEquals(1, ingestCalls.get());
        }
    }

    @Test
    void unreachableNodeFailsAfterBoundedReconnects() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16)) {
            cluster.channels.get(0).shutdownNow();
            try (CoresetCoordinator coordinator = new CoresetCoordinator(cluster.channels, options(4));
                 IngestStream stream = coordinator.openIngest("s")) {
                stream.submit(TestData.batch(new SplittableRandom(5), 2, DIM));
                IngestFailedException e = assertThrows(IngestFailedException.class, () -> stream.flush(WAIT));
                assertTrue(e.getMessage().contains("unreachable"), e.getMessage());
            }
        }
    }

    @Test
    void unionOfNodeSnapshotsAddsOffsets() {
        WeightedPoint p = new WeightedPoint(SparseVector.dense(new double[] {1}), 2);
        NodeSnapshot a = new NodeSnapshot(new Coreset(List.of(p), 1.5), io.github.c0rwin.coreset.node.SessionState.ACTIVE,
                "", 3, 3, 4.0, 4.0, true);
        NodeSnapshot b = new NodeSnapshot(new Coreset(List.of(p), 0.5), io.github.c0rwin.coreset.node.SessionState.ACTIVE,
                "", 2, 1, 4.0, 2.0, true);
        GatheredSnapshot g = new GatheredSnapshot(Coreset.union(a.coreset(), b.coreset()), List.of(a, b));
        assertEquals(2.0, g.coreset().offset());
        assertEquals(5, g.acceptedBatches());
        assertEquals(4, g.appliedBatches());
        assertEquals(2.0, g.pendingWeight());
        assertTrue(!g.complete());
    }
}
