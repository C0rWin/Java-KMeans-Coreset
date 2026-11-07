package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.AckStatus;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.PartialCostRequest;
import io.github.c0rwin.coreset.grpc.proto.PointList;
import io.github.c0rwin.coreset.grpc.proto.SnapshotRequest;
import io.github.c0rwin.coreset.grpc.proto.SparseVector;
import io.github.c0rwin.coreset.grpc.proto.WeightedPoint;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** Default node limits are enforced at the gRPC boundary with clear status codes (review H1-H3, H5, L1, M3). */
class NodeLimitsGrpcTest {

    private InProcessCluster cluster;
    private CoresetNodeServiceGrpc.CoresetNodeServiceBlockingStub stub;

    private void start(Map<String, io.github.c0rwin.coreset.core.CoresetAlgorithm> extra) throws Exception {
        cluster = new InProcessCluster(1, extra, 4);
        stub = CoresetNodeServiceGrpc.newBlockingStub(cluster.channels.get(0)).withDeadlineAfter(60, TimeUnit.SECONDS);
    }

    @AfterEach
    void stop() throws Exception {
        if (cluster != null) {
            cluster.close();
        }
    }

    private static CreateSessionRequest.Builder create(String id, int workers, int queueCapacity) {
        return CreateSessionRequest.newBuilder().setSessionId(id)
                .setSpec(ProtoCodec.toProto(new SessionSpec("identity", Map.of(), 1L, workers, queueCapacity)));
    }

    private static Status.Code code(Runnable call) {
        return assertThrows(StatusRuntimeException.class, call::run).getStatus().getCode();
    }

    private Ack ingestOne(Batch batch) throws InterruptedException {
        BlockingQueue<Ack> acks = new LinkedBlockingQueue<>();
        StreamObserver<Batch> requests = CoresetNodeServiceGrpc.newStub(cluster.channels.get(0)).ingest(
                new StreamObserver<>() {
                    @Override
                    public void onNext(Ack value) {
                        acks.add(value);
                    }

                    @Override
                    public void onError(Throwable t) {
                    }

                    @Override
                    public void onCompleted() {
                    }
                });
        requests.onNext(batch);
        Ack a = acks.poll(30, TimeUnit.SECONDS);
        requests.onCompleted();
        assertNotNull(a, "no ack");
        return a;
    }

    private static WeightedPoint point(int dim, int index) {
        return WeightedPoint.newBuilder().setWeight(1)
                .setVector(SparseVector.newBuilder().setDim(dim).addIndices(index).addValues(1)).build();
    }

    @Test
    void hugeDimensionIsRejectedBeforeDecoding() throws Exception {
        start(Map.of());
        stub.createSession(create("s", 1, 2).build());
        Ack ack = ingestOne(Batch.newBuilder().setSessionId("s").setSeq(0).addPoints(point(1 << 28, 5)).build());
        assertEquals(AckStatus.ACK_STATUS_FAILED, ack.getStatus());
        assertTrue(ack.getMessage().contains("dim 268435456 exceeds"), ack.getMessage());
    }

    @Test
    void oversizedBatchIsRejectedBeforeDecoding() throws Exception {
        start(Map.of());
        stub.createSession(create("s", 1, 2).build());
        Batch.Builder b = Batch.newBuilder().setSessionId("s").setSeq(0);
        WeightedPoint empty = WeightedPoint.newBuilder().setVector(SparseVector.getDefaultInstance()).build();
        for (int i = 0; i <= (1 << 18); i++) {
            b.addPoints(empty);
        }
        Ack ack = ingestOne(b.build());
        assertEquals(AckStatus.ACK_STATUS_FAILED, ack.getStatus());
        assertTrue(ack.getMessage().contains("262145 points exceed"), ack.getMessage());
    }

    @Test
    void sessionShapeLimits() throws Exception {
        start(Map.of());
        assertEquals(Status.Code.INVALID_ARGUMENT, code(() -> stub.createSession(create("q", 1, 1 << 20).build())));
        assertEquals(Status.Code.INVALID_ARGUMENT, code(() -> stub.createSession(create("x".repeat(129), 1, 2).build())));
        assertEquals(Status.Code.INVALID_ARGUMENT, code(() -> stub.createSession(create("a/b", 1, 2).build())));
        CreateSessionRequest.Builder manyParams = create("p", 1, 2);
        for (int i = 0; i < 33; i++) {
            manyParams.getSpecBuilder().putParams("k" + i, "v");
        }
        assertEquals(Status.Code.INVALID_ARGUMENT, code(() -> stub.createSession(manyParams.build())));
        CreateSessionRequest.Builder longValue = create("v", 1, 2);
        longValue.getSpecBuilder().putParams("k", "v".repeat(257));
        StatusRuntimeException e = assertThrows(StatusRuntimeException.class,
                () -> stub.createSession(longValue.build()));
        assertEquals(Status.Code.INVALID_ARGUMENT, e.getStatus().getCode());
        assertTrue(e.getStatus().getDescription().length() <= 300, e.getStatus().getDescription());
    }

    @Test
    void sessionCountIsBounded() throws Exception {
        start(Map.of());
        for (int i = 0; i < 64; i++) {
            assertTrue(stub.createSession(create("s" + i, 1, 2).build()).getCreated());
        }
        assertEquals(Status.Code.RESOURCE_EXHAUSTED, code(() -> stub.createSession(create("s64", 1, 2).build())));
    }

    @Test
    void partialCostWorkIsBounded() throws Exception {
        start(Map.of());
        PartialCostRequest.Builder tooManyCenters = PartialCostRequest.newBuilder()
                .setPoints(PointList.newBuilder().addPoints(point(1, 0)));
        for (int i = 0; i <= 100_000; i++) {
            tooManyCenters.addCenters(SparseVector.newBuilder().setDim(1));
        }
        assertEquals(Status.Code.INVALID_ARGUMENT, code(() -> stub.withDeadlineAfter(10, TimeUnit.SECONDS)
                .partialCost(tooManyCenters.build())));

        PartialCostRequest.Builder tooMuchWork = PartialCostRequest.newBuilder();
        PointList.Builder points = PointList.newBuilder();
        for (int i = 0; i < 50_000; i++) {
            tooMuchWork.addCenters(SparseVector.newBuilder().setDim(1).addIndices(0).addValues(i));
            points.addPoints(point(1, 0));
        }
        tooMuchWork.setPoints(points);
        assertEquals(Status.Code.RESOURCE_EXHAUSTED, code(() -> stub.withDeadlineAfter(10, TimeUnit.SECONDS)
                .partialCost(tooMuchWork.build())));
    }

    @Test
    void hugeAwaitIsClampedAndCancelledWaitsReturn() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(TestAlgorithms.IDENTITY);
        start(Map.of("gated", gated));
        stub.createSession(CreateSessionRequest.newBuilder().setSessionId("g")
                .setSpec(ProtoCodec.toProto(new SessionSpec("gated", Map.of(), 1L, 1, 2))).build());
        // Above Long.MAX_VALUE nanoseconds: must not surface as UNKNOWN.
        stub.createSession(create("idle", 1, 2).build());
        assertEquals(io.github.c0rwin.coreset.grpc.proto.SessionState.SESSION_STATE_ACTIVE, stub.snapshot(
                SnapshotRequest.newBuilder().setSessionId("idle").setAwaitAppliedMs(10_000_000_000_000_000L).build())
                .getState());

        assertEquals(AckStatus.ACK_STATUS_ACCEPTED,
                ingestOne(Batch.newBuilder().setSessionId("g").setSeq(0).addPoints(point(2, 1)).build()).getStatus());
        gated.awaitEntered();
        assertEquals(Status.Code.DEADLINE_EXCEEDED, code(() -> stub.withDeadlineAfter(200, TimeUnit.MILLISECONDS)
                .snapshot(SnapshotRequest.newBuilder().setSessionId("g").setAwaitAppliedMs(9_000_000_000_000L).build())));
        // The server-side handler notices the cancellation and finishes instead of parking until the gate opens.
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
        while (!cluster.metrics.get(0).snapshot().counters().keySet().stream()
                .anyMatch(k -> k.startsWith("grpc.server.Snapshot.status.") && !k.endsWith(".OK"))
                && System.nanoTime() < deadline) {
            LockSupport.parkNanos(10_000_000);
        }
        assertTrue(cluster.metrics.get(0).snapshot().counters().keySet().stream()
                        .anyMatch(k -> k.startsWith("grpc.server.Snapshot.status.") && !k.endsWith(".OK")),
                cluster.metrics.get(0).snapshot().counters().toString());
        gated.release();
    }
}
