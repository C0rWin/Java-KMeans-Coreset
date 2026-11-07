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

    /**
     * Inbound message limit. Only requests count (batches, centers, cost shards); snapshots flow server to client and
     * are bounded by the coordinator's own inbound limit. 16 MiB holds a batch of the default 2^18-point cap with a
     * few dozen non-zeros per point; decoded heap is several times the wire size, so keep it modest.
     */
    public static final int DEFAULT_MAX_INBOUND_MESSAGE_BYTES = 16 * 1024 * 1024;

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
     * @param maxInboundMessageBytes largest request message accepted
     */
    public static CoresetNodeServer start(int port, CoresetNode node, RecordingMetrics stats, MetricsSink transportSink,
                                          int maxInboundMessageBytes) throws IOException {
        CoresetNodeGrpcService service = new CoresetNodeGrpcService(node, stats);
        HealthStatusManager health = new HealthStatusManager();
        Server server = NettyServerBuilder.forPort(port)
                .maxInboundMessageSize(maxInboundMessageBytes)
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
