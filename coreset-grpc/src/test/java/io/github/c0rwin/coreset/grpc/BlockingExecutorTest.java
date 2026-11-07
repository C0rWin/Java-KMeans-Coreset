package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.CreateSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.SnapshotRequest;
import io.github.c0rwin.coreset.grpc.proto.SnapshotResponse;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.node.testing.TestData;
import io.github.c0rwin.coreset.node.testing.TestResolver;
import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.Status;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.junit.jupiter.api.Test;

/** Blocking calls run on a bounded executor; beyond its queue they are refused, not queued without bound (M4). */
class BlockingExecutorTest {

    @Test
    void blockingCallsBeyondTheQueueAreRefused() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(TestAlgorithms.IDENTITY);
        CoresetNode node = new CoresetNode(new TestResolver(Map.of("gated", gated)), MetricsSink.NOOP,
                Duration.ofMillis(200));
        CoresetNodeGrpcService service = new CoresetNodeGrpcService(node, () -> "{}",
                CoresetNodeGrpcService.Config.defaults().withBlocking(1, 1));
        String name = InProcessServerBuilder.generateName();
        Server server = InProcessServerBuilder.forName(name).addService(service).build().start();
        ManagedChannel channel = InProcessChannelBuilder.forName(name).build();
        try {
            CoresetNodeServiceGrpc.CoresetNodeServiceBlockingStub stub = CoresetNodeServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(30, TimeUnit.SECONDS);
            stub.createSession(CreateSessionRequest.newBuilder().setSessionId("g")
                    .setSpec(ProtoCodec.toProto(new SessionSpec("gated", Map.of(), 1L, 1, 2))).build());
            node.ingest("g", 0, TestData.batch(new SplittableRandom(1), 1, 2));
            gated.awaitEntered();
            SnapshotRequest waiting = SnapshotRequest.newBuilder().setSessionId("g").setAwaitAppliedMs(30_000).build();
            CoresetNodeServiceGrpc.CoresetNodeServiceStub async = CoresetNodeServiceGrpc.newStub(channel);
            List<CompletableFuture<SnapshotResponse>> calls = new ArrayList<>();
            for (int i = 0; i < 5; i++) {
                CompletableFuture<SnapshotResponse> f = new CompletableFuture<>();
                async.snapshot(waiting, observer(f));
                calls.add(f);
            }
            // One thread plus one queue slot: at most two of the five waits are admitted while the gate is shut.
            int refused = 0;
            for (CompletableFuture<SnapshotResponse> f : calls) {
                try {
                    f.get(200, TimeUnit.MILLISECONDS);
                } catch (ExecutionException e) {
                    assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(e.getCause()).getCode());
                    refused++;
                } catch (TimeoutException e) {
                    // admitted and waiting for the gate
                }
            }
            assertTrue(refused >= 3, "refused " + refused);
            gated.release();
            for (CompletableFuture<SnapshotResponse> f : calls) {
                if (!f.isCompletedExceptionally()) {
                    assertEquals(1, f.get(30, TimeUnit.SECONDS).getAppliedBatches());
                }
            }
        } finally {
            gated.release();
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            service.close();
            node.close();
        }
    }

    private static <T> StreamObserver<T> observer(CompletableFuture<T> f) {
        return new StreamObserver<>() {
            @Override
            public void onNext(T value) {
                f.complete(value);
            }

            @Override
            public void onError(Throwable t) {
                f.completeExceptionally(t);
            }

            @Override
            public void onCompleted() {
            }
        };
    }
}
