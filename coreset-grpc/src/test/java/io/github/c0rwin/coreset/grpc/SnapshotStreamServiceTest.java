package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.grpc.proto.ReplicateRequest;
import io.github.c0rwin.coreset.grpc.proto.SnapshotChunk;
import io.github.c0rwin.coreset.grpc.proto.SnapshotStreamRequest;
import io.github.c0rwin.coreset.grpc.proto.StoreReplicaResponse;
import io.github.c0rwin.coreset.node.MessageSpec;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

/** Wire behaviour of SnapshotStream, Replicate and StoreReplica on one node. */
class SnapshotStreamServiceTest {

    private static void session(InProcessCluster cluster) throws Exception {
        cluster.nodes.get(0).createSession("s", new SessionSpec("identity", Map.of(), 1L, 1, 16,
                TreeMode.SINGLE_WRITER, MessageSpec.perDraw(40, 2, 16)));
        List<WeightedPoint> points = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            points.add(new WeightedPoint(SparseVector.dense(new double[] {i, i % 3}), 1.0));
        }
        cluster.nodes.get(0).ingest("s", 0, points);
        cluster.nodes.get(0).awaitApplied("s", Duration.ofSeconds(10));
    }

    @Test
    void streamsOneHeaderThenItsPacketsInOrder() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16)) {
            session(cluster);
            Iterator<SnapshotChunk> it = CoresetNodeServiceGrpc.newBlockingStub(cluster.channels.get(0))
                    .snapshotStream(SnapshotStreamRequest.newBuilder().setSessionId("s").setOwner(5).build());
            MessageHeader h = it.next().getHeader();
            assertEquals(5, h.getOwner());
            assertEquals(40, h.getDraws());
            assertEquals(3, h.getPackets());
            assertEquals(30.0, h.getNodeWeight());
            assertTrue(h.getComplete());
            int seq = 0;
            int records = 0;
            while (it.hasNext()) {
                Packet p = it.next().getPacket();
                assertEquals(seq++, p.getSeq());
                records += p.getRecordsCount();
                assertEquals(p.getRecordsCount(), p.getTagsCount());
            }
            assertEquals(3, seq);
            assertEquals(40, records);

            Iterator<SnapshotChunk> headerOnly = CoresetNodeServiceGrpc.newBlockingStub(cluster.channels.get(0))
                    .snapshotStream(SnapshotStreamRequest.newBuilder().setSessionId("s").setHeaderOnly(true).build());
            assertTrue(headerOnly.next().hasHeader());
            assertFalse(headerOnly.hasNext());
        }
    }

    @Test
    void aMissingReplicaIsNotFoundAndReplicateNeedsAPeerTransport() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16)) {
            session(cluster);
            CoresetNodeServiceGrpc.CoresetNodeServiceBlockingStub stub =
                    CoresetNodeServiceGrpc.newBlockingStub(cluster.channels.get(0));
            StatusRuntimeException e = org.junit.jupiter.api.Assertions.assertThrows(StatusRuntimeException.class,
                    () -> stub.snapshotStream(SnapshotStreamRequest.newBuilder().setSessionId("s").setOwner(3)
                            .setReplica(true).build()).next());
            assertEquals(Status.Code.NOT_FOUND, e.getStatus().getCode());
            e = org.junit.jupiter.api.Assertions.assertThrows(StatusRuntimeException.class,
                    () -> stub.replicate(ReplicateRequest.newBuilder().setSessionId("s").addPeers("x").build()));
            assertEquals(Status.Code.FAILED_PRECONDITION, e.getStatus().getCode());
        }
    }

    @Test
    void storeReplicaRefusesPacketsBeforeTheHeaderAndIncompleteMessages() throws Exception {
        try (InProcessCluster cluster = new InProcessCluster(1, Map.of(), 16)) {
            session(cluster);
            CoresetNodeServiceGrpc.CoresetNodeServiceStub stub = CoresetNodeServiceGrpc.newStub(cluster.channels.get(0));
            assertEquals(Status.Code.INVALID_ARGUMENT, store(stub, List.of(
                    SnapshotChunk.newBuilder().setPacket(Packet.newBuilder().setSeq(0)).build())));
            MessageHeader h = MessageHeader.newBuilder().setSessionId("s").setOwner(1).setRecords(20)
                    .setRecordsPerPacket(10).setPackets(2).setDraws(20).build();
            assertEquals(Status.Code.INVALID_ARGUMENT, store(stub, List.of(
                    SnapshotChunk.newBuilder().setHeader(h).build())));
            MessageHeader huge = h.toBuilder().setRecords(Integer.MAX_VALUE).build();
            assertEquals(Status.Code.INVALID_ARGUMENT, store(stub, List.of(
                    SnapshotChunk.newBuilder().setHeader(huge).build())));
        }
    }

    private static Status.Code store(CoresetNodeServiceGrpc.CoresetNodeServiceStub stub, List<SnapshotChunk> chunks)
            throws InterruptedException, java.util.concurrent.TimeoutException {
        CompletableFuture<Status.Code> done = new CompletableFuture<>();
        StreamObserver<SnapshotChunk> out = stub.storeReplica(new StreamObserver<StoreReplicaResponse>() {
            @Override
            public void onNext(StoreReplicaResponse r) {
            }

            @Override
            public void onError(Throwable t) {
                done.complete(Status.fromThrowable(t).getCode());
            }

            @Override
            public void onCompleted() {
                done.complete(Status.Code.OK);
            }
        });
        for (SnapshotChunk c : chunks) {
            out.onNext(c);
        }
        out.onCompleted();
        try {
            return done.get(10, TimeUnit.SECONDS);
        } catch (ExecutionException e) {
            throw new AssertionError(e);
        }
    }
}
