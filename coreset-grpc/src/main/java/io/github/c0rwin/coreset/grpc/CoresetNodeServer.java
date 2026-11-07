package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.grpc.Server;
import io.grpc.ServerInterceptors;
import io.grpc.health.v1.HealthCheckResponse.ServingStatus;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.protobuf.services.HealthStatusManager;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

/** Netty gRPC server exposing a {@link CoresetNode} plus the standard {@code grpc.health.v1.Health} service. */
public final class CoresetNodeServer implements AutoCloseable {

    /** Inbound message limit; snapshots and large batches exceed gRPC's 4 MiB default. */
    public static final int DEFAULT_MAX_MESSAGE_BYTES = 256 * 1024 * 1024;

    private final Server server;
    private final CoresetNodeGrpcService service;
    private final HealthStatusManager health;

    private CoresetNodeServer(Server server, CoresetNodeGrpcService service, HealthStatusManager health) {
        this.server = server;
        this.service = service;
        this.health = health;
    }

    /**
     * @param port         0 picks an ephemeral port
     * @param stats        recorder whose JSON snapshot the Stats RPC returns
     * @param transportSink receives per-method gRPC metrics
     */
    public static CoresetNodeServer start(int port, CoresetNode node, RecordingMetrics stats, MetricsSink transportSink)
            throws IOException {
        CoresetNodeGrpcService service = new CoresetNodeGrpcService(node, stats);
        HealthStatusManager health = new HealthStatusManager();
        Server server = NettyServerBuilder.forPort(port)
                .maxInboundMessageSize(DEFAULT_MAX_MESSAGE_BYTES)
                .addService(ServerInterceptors.intercept(service, GrpcMetrics.server(transportSink)))
                .addService(health.getHealthService())
                .build();
        try {
            server.start();
        } catch (IOException e) {
            service.close();
            throw e;
        }
        health.setStatus(HealthStatusManager.SERVICE_NAME_ALL_SERVICES, ServingStatus.SERVING);
        health.setStatus(CoresetNodeServiceGrpc.SERVICE_NAME, ServingStatus.SERVING);
        return new CoresetNodeServer(server, service, health);
    }

    public int port() {
        return server.getPort();
    }

    public void awaitTermination() throws InterruptedException {
        server.awaitTermination();
    }

    /** Marks the node NOT_SERVING, then stops accepting calls and waits briefly for in-flight ones. */
    @Override
    public void close() {
        health.enterTerminalState();
        server.shutdown();
        try {
            if (!server.awaitTermination(5, TimeUnit.SECONDS)) {
                server.shutdownNow();
                server.awaitTermination(5, TimeUnit.SECONDS);
            }
        } catch (InterruptedException e) {
            server.shutdownNow();
            Thread.currentThread().interrupt();
        } finally {
            service.close();
        }
    }
}
