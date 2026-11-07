package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.AckStatus;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import io.github.c0rwin.coreset.grpc.proto.CloseSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.PartialCostRequest;
import io.github.c0rwin.coreset.grpc.proto.PartialCostResponse;
import io.github.c0rwin.coreset.grpc.proto.PointList;
import io.github.c0rwin.coreset.grpc.proto.SessionState;
import io.github.c0rwin.coreset.grpc.proto.SnapshotRequest;
import io.github.c0rwin.coreset.grpc.proto.SnapshotResponse;
import io.github.c0rwin.coreset.grpc.proto.StatsRequest;
import io.github.c0rwin.coreset.grpc.proto.StatsResponse;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.node.testing.TestData;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.health.v1.HealthCheckRequest;
import io.grpc.health.v1.HealthCheckResponse;
import io.grpc.health.v1.HealthGrpc;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

class CoresetNodeGrpcServiceTest {

    private static final int DIM = 8;
    private InProcessCluster cluster;
    private CoresetNodeServiceGrpc.CoresetNodeServiceBlockingStub stub;

    private void start(Map<String, io.github.c0rwin.coreset.core.CoresetAlgorithm> extra, int window)
            throws Exception {
        cluster = new InProcessCluster(1, extra, window);
        stub = CoresetNodeServiceGrpc.newBlockingStub(cluster.channels.get(0));
    }

    @AfterEach
    void stop() throws Exception {
        if (cluster != null) {
            cluster.close();
        }
    }

    private void create(String id, SessionSpec spec) {
        stub.createSession(CreateSessionRequest.newBuilder().setSessionId(id).setSpec(ProtoCodec.toProto(spec)).build());
    }

    /** Raw bidi stream whose acks land in a queue. */
    private static final class AckCollector implements StreamObserver<Ack> {
        final BlockingQueue<Ack> acks = new LinkedBlockingQueue<>();
        final CountDownLatch done = new CountDownLatch(1);
        volatile Throwable error;

        @Override
        public void onNext(Ack value) {
            acks.add(value);
        }

        @Override
        public void onError(Throwable t) {
            error = t;
            done.countDown();
        }

        @Override
        public void onCompleted() {
            done.countDown();
        }

        Ack next() throws InterruptedException {
            Ack a = acks.poll(30, TimeUnit.SECONDS);
            assertNotNull(a, "no ack within 30s");
            return a;
        }
    }

    private static Batch batch(String session, long seq, List<WeightedPoint> points) {
        return Batch.newBuilder().setSessionId(session).setSeq(seq)
                .addAllPoints(ProtoCodec.pointsToProto(points)).build();
    }

    @Test
    void createAndCloseSession() throws Exception {
        start(Map.of(), 4);
        SessionSpec spec = new SessionSpec("identity", Map.of("a", "1"), 3L, 2, 4, TreeMode.CAS);
        CreateSessionRequest req = CreateSessionRequest.newBuilder().setSessionId("s")
                .setSpec(ProtoCodec.toProto(spec)).build();
        assertTrue(stub.createSession(req).getCreated());
        assertFalse(stub.createSession(req).getCreated());
        assertEquals(spec, cluster.nodes.get(0).spec("s"));

        StatusRuntimeException mismatch = assertThrows(StatusRuntimeException.class, () -> create("s",
                new SessionSpec("identity", Map.of(), 3L, 2, 4)));
        assertEquals(Status.Code.FAILED_PRECONDITION, mismatch.getStatus().getCode());
        StatusRuntimeException unknown = assertThrows(StatusRuntimeException.class, () -> create("t",
                new SessionSpec("nope", Map.of(), 3L, 2, 4)));
        assertEquals(Status.Code.INVALID_ARGUMENT, unknown.getStatus().getCode());
        StatusRuntimeException badCapacity = assertThrows(StatusRuntimeException.class,
                () -> stub.createSession(req.toBuilder().setSessionId("u")
                        .setSpec(req.getSpec().toBuilder().setQueueCapacity(3)).build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, badCapacity.getStatus().getCode());

        CloseSessionRequest close = CloseSessionRequest.newBuilder().setSessionId("s").build();
        assertTrue(stub.closeSession(close).getClosed());
        assertFalse(stub.closeSession(close).getClosed());
    }

    @Test
    void ingestAcksEveryBatchInOrderAndDeduplicatesRetries() throws Exception {
        start(Map.of(), 4);
        create("s", new SessionSpec("identity", Map.of(), 1L, 2, 8));
        SplittableRandom rng = new SplittableRandom(1);
        List<WeightedPoint> b0 = TestData.batch(rng, 4, DIM);
        List<WeightedPoint> b1 = TestData.batch(rng, 2, DIM);
        AckCollector acks = new AckCollector();
        StreamObserver<Batch> requests = CoresetNodeServiceGrpc.newStub(cluster.channels.get(0)).ingest(acks);
        requests.onNext(batch("s", 0, b0));
        requests.onNext(batch("s", 1, b1));
        requests.onNext(batch("s", 1, b1));
        requests.onNext(batch("missing", 0, b1));
        requests.onNext(Batch.newBuilder().setSessionId("s").setSeq(2).addPoints(
                io.github.c0rwin.coreset.grpc.proto.WeightedPoint.newBuilder().setWeight(1).setVector(
                        io.github.c0rwin.coreset.grpc.proto.SparseVector.newBuilder().setDim(4).addIndices(1)))
                .build());
        requests.onCompleted();

        assertEquals(Ack.newBuilder().setSeq(0).setStatus(AckStatus.ACK_STATUS_ACCEPTED).setAcceptedPoints(4).build(),
                acks.next());
        assertEquals(AckStatus.ACK_STATUS_ACCEPTED, acks.next().getStatus());
        assertEquals(AckStatus.ACK_STATUS_DUPLICATE, acks.next().getStatus());
        Ack missing = acks.next();
        assertEquals(AckStatus.ACK_STATUS_FAILED, missing.getStatus());
        assertTrue(missing.getMessage().contains("unknown session"), missing.getMessage());
        Ack invalid = acks.next();
        assertEquals(2, invalid.getSeq());
        assertEquals(AckStatus.ACK_STATUS_FAILED, invalid.getStatus());
        assertTrue(invalid.getMessage().startsWith("invalid batch"), invalid.getMessage());

        SnapshotResponse snap = stub.snapshot(SnapshotRequest.newBuilder().setSessionId("s")
                .setAwaitAppliedMs(30_000).build());
        assertEquals(2, snap.getAcceptedBatches());
        assertEquals(2, snap.getAppliedBatches());
        assertTrue(snap.getComplete());
        assertEquals(TestData.weight(b0) + TestData.weight(b1), ProtoCodec.fromProto(snap.getCoreset(), DIM).totalWeight());
    }

    @Test
    void seqAtOrAbove2To63IsRejectedBeforeReachingTheNode() throws Exception {
        start(Map.of(), 4);
        create("s", new SessionSpec("identity", Map.of(), 1L, 1, 2));
        List<WeightedPoint> b = TestData.batch(new SplittableRandom(9), 2, DIM);
        AckCollector acks = new AckCollector();
        StreamObserver<Batch> requests = CoresetNodeServiceGrpc.newStub(cluster.channels.get(0)).ingest(acks);
        requests.onNext(batch("s", 0, b));
        assertEquals(AckStatus.ACK_STATUS_ACCEPTED, acks.next().getStatus());
        requests.onNext(batch("s", Long.MAX_VALUE, b)); // 2^63 - 1 is valid on the wire; the node's window refuses it
        Ack far = acks.next();
        assertEquals(AckStatus.ACK_STATUS_FAILED, far.getStatus());
        assertTrue(far.getMessage().contains("dedupe window"), far.getMessage());
        requests.onNext(batch("s", Long.MIN_VALUE, b)); // uint64 2^63 on the wire
        requests.onNext(batch("s", 1, b)); // never read: the stream is already failed
        assertTrue(acks.done.await(30, TimeUnit.SECONDS));
        Status status = Status.fromThrowable(acks.error);
        assertEquals(Status.Code.INVALID_ARGUMENT, status.getCode());
        assertEquals("seq 9223372036854775808 exceeds 2^63-1", status.getDescription());
        assertTrue(acks.acks.isEmpty());

        var counters = cluster.metrics.get(0).snapshot().counters();
        assertEquals(1, counters.get(io.github.c0rwin.coreset.node.NodeMetricNames.INGEST_ACCEPTED));
        assertEquals(1, counters.get(io.github.c0rwin.coreset.node.NodeMetricNames.INGEST_FAILED)); // 2^63 never got there
        assertTrue(cluster.nodes.get(0).awaitApplied("s", Duration.ofSeconds(30)));
        assertEquals(1, cluster.nodes.get(0).snapshot("s").appliedBatches());
    }

    @Test
    void busyAckDelaysReadsAndTheSameSeqIsAcceptedLater() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(TestAlgorithms.IDENTITY);
        // Window 1: after a BUSY ack, only the delayed re-request lets the server read the retry.
        start(Map.of("gated", gated), 1);
        create("s", new SessionSpec("gated", Map.of(), 1L, 1, 2));
        SplittableRandom rng = new SplittableRandom(2);
        AckCollector acks = new AckCollector();
        StreamObserver<Batch> requests = CoresetNodeServiceGrpc.newStub(cluster.channels.get(0)).ingest(acks);
        requests.onNext(batch("s", 0, TestData.batch(rng, 2, DIM)));
        assertEquals(AckStatus.ACK_STATUS_ACCEPTED, acks.next().getStatus());
        gated.awaitEntered();
        requests.onNext(batch("s", 1, TestData.batch(rng, 2, DIM)));
        requests.onNext(batch("s", 2, TestData.batch(rng, 2, DIM)));
        Batch third = batch("s", 3, TestData.batch(rng, 2, DIM));
        requests.onNext(third);
        assertEquals(AckStatus.ACK_STATUS_ACCEPTED, acks.next().getStatus());
        assertEquals(AckStatus.ACK_STATUS_ACCEPTED, acks.next().getStatus());
        Ack busy = acks.next();
        assertEquals(3, busy.getSeq());
        assertEquals(AckStatus.ACK_STATUS_BUSY, busy.getStatus());

        requests.onNext(third);
        assertEquals(AckStatus.ACK_STATUS_BUSY, acks.next().getStatus());
        gated.release();
        Ack last = assertTimeoutPreemptively(Duration.ofSeconds(30), () -> {
            for (;;) {
                requests.onNext(third);
                Ack a = acks.next();
                if (a.getStatus() != AckStatus.ACK_STATUS_BUSY) {
                    return a;
                }
            }
        });
        assertEquals(AckStatus.ACK_STATUS_ACCEPTED, last.getStatus());
        requests.onCompleted();
        SnapshotResponse snap = stub.snapshot(SnapshotRequest.newBuilder().setSessionId("s")
                .setAwaitAppliedMs(30_000).build());
        assertEquals(4, snap.getAppliedBatches());
        assertTrue(snap.getComplete());
    }

    @Test
    void snapshotReportsFailuresAndUnknownSessions() throws Exception {
        start(Map.of(), 4);
        create("f", new SessionSpec("failing", Map.of("poison", "7"), 1L, 1, 2));
        WeightedPoint poison = new WeightedPoint(SparseVector.dense(new double[] {1, 0}), 7);
        assertEquals(io.github.c0rwin.coreset.node.IngestResult.Status.ACCEPTED,
                cluster.nodes.get(0).ingest("f", 0, List.of(poison)).status());
        SnapshotResponse snap = stub.snapshot(SnapshotRequest.newBuilder().setSessionId("f")
                .setAwaitAppliedMs(30_000).build());
        assertEquals(SessionState.SESSION_STATE_FAILED, snap.getState());
        assertTrue(snap.getFailure().contains("poisoned batch"), snap.getFailure());
        assertFalse(snap.getComplete());

        StatusRuntimeException e = assertThrows(StatusRuntimeException.class,
                () -> stub.snapshot(SnapshotRequest.newBuilder().setSessionId("nope").build()));
        assertEquals(Status.Code.NOT_FOUND, e.getStatus().getCode());
    }

    @Test
    void partialCostOnPointsAndOnSessionCoreset() throws Exception {
        start(Map.of(), 4);
        SplittableRandom rng = new SplittableRandom(3);
        List<WeightedPoint> points = TestData.batch(rng, 50, DIM);
        List<SparseVector> centers = List.of(points.get(0).vector(), points.get(1).vector());
        PartialCostResponse onPoints = stub.partialCost(PartialCostRequest.newBuilder()
                .addAllCenters(ProtoCodec.vectorsToProto(centers))
                .setPoints(PointList.newBuilder().addAllPoints(ProtoCodec.pointsToProto(points)))
                .build());
        assertEquals(Cost.cost(points, centers), onPoints.getCost());
        assertEquals(TestData.weight(points), onPoints.getWeight());
        assertEquals(50, onPoints.getPoints());

        create("s", new SessionSpec("identity", Map.of(), 1L, 1, 2));
        cluster.nodes.get(0).ingest("s", 0, points);
        assertTrue(cluster.nodes.get(0).awaitApplied("s", Duration.ofSeconds(30)));
        PartialCostResponse onSession = stub.partialCost(PartialCostRequest.newBuilder()
                .addAllCenters(ProtoCodec.vectorsToProto(centers)).setSessionId("s").build());
        assertEquals(Cost.cost(Coreset.of(points), centers), onSession.getCost());

        StatusRuntimeException e = assertThrows(StatusRuntimeException.class, () -> stub.partialCost(
                PartialCostRequest.newBuilder().addAllCenters(ProtoCodec.vectorsToProto(centers)).build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, e.getStatus().getCode());
    }

    @Test
    void statsHealthAndTransportMetrics() throws Exception {
        start(Map.of(), 4);
        create("b", new SessionSpec("identity", Map.of(), 1L, 1, 2));
        create("a", new SessionSpec("identity", Map.of(), 1L, 1, 2));
        StatsResponse stats = stub.stats(StatsRequest.getDefaultInstance());
        // Without authentication configured every caller is anonymous: session ids are not handed out.
        assertEquals(List.of(), stats.getSessionIdsList());
        assertTrue(stats.getMetricsJson().contains("\"node.sessions.created\":2"), stats.getMetricsJson());
        assertTrue(stats.getMetricsJson().contains("grpc.server.CreateSession.bytes_in"), stats.getMetricsJson());

        HealthCheckResponse health = HealthGrpc.newBlockingStub(cluster.channels.get(0))
                .check(HealthCheckRequest.newBuilder().setService("").build());
        assertEquals(HealthCheckResponse.ServingStatus.SERVING, health.getStatus());

        var counters = cluster.metrics.get(0).snapshot().counters();
        assertEquals(2, counters.get("grpc.server.CreateSession.messages_in"));
        assertTrue(counters.get("grpc.server.CreateSession.bytes_in") > 0);
        assertEquals(2, counters.get("grpc.server.CreateSession.status.OK"));
        assertNotNull(cluster.metrics.get(0).snapshot().distributions().get("grpc.server.CreateSession.latency_ns"));
    }
}
