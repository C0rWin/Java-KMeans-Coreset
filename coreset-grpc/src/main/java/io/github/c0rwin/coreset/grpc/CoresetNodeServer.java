package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.grpc.Server;
import io.grpc.ServerCredentials;
import io.grpc.ServerInterceptor;
import io.grpc.ServerInterceptors;
import io.grpc.TlsServerCredentials;
import io.grpc.health.v1.HealthCheckResponse.ServingStatus;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;
import io.grpc.protobuf.services.HealthStatusManager;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/**
 * Netty gRPC server exposing a {@link CoresetNode} plus the standard {@code grpc.health.v1.Health} service.
 *
 * <p>Security posture: listens on loopback unless told otherwise, plaintext unless TLS files are given, and accepts
 * any caller unless a shared token or client certificates are required (see {@link ServerSecurity}). The health
 * service stays unauthenticated so load balancers can probe it.
 */
public final class CoresetNodeServer implements AutoCloseable {

    /**
     * Inbound message limit. Only requests count (batches, centers, cost shards); snapshots flow server to client and
     * are bounded by the coordinator's own inbound limit. 16 MiB holds a batch of the default 2^18-point cap with a
     * few dozen non-zeros per point; decoded heap is several times the wire size, so keep it modest.
     */
    public static final int DEFAULT_MAX_INBOUND_MESSAGE_BYTES = 16 * 1024 * 1024;

    /** Stats responses are served from a snapshot at most this old. */
    public static final Duration STATS_TTL = Duration.ofSeconds(1);

    private final Server server;
    private final CoresetNodeGrpcService service;
    private final HealthStatusManager health;
    private final ExecutorService handlers;

    private CoresetNodeServer(Server server, CoresetNodeGrpcService service, HealthStatusManager health,
                              ExecutorService handlers) {
        this.server = server;
        this.service = service;
        this.health = health;
        this.handlers = handlers;
    }

    /**
     * @param stats         recorder whose JSON snapshot the Stats RPC returns
     * @param transportSink receives per-method gRPC metrics
     */
    public static CoresetNodeServer start(NodeServerOptions options, CoresetNode node, RecordingMetrics stats,
                                          MetricsSink transportSink) throws IOException {
        ServerSecurity security = options.security();
        CoresetNodeGrpcService service = new CoresetNodeGrpcService(node, cached(() -> stats.snapshot().toJson()),
                CoresetNodeGrpcService.Config.defaults()
                        .withBlocking(options.blockingThreads(), options.blockingQueue())
                        .withSessionIdsInStats(security.authenticatesCallers()));
        HealthStatusManager health = new HealthStatusManager();
        ExecutorService handlers = Executors.newFixedThreadPool(options.handlerThreads(), daemonThreads());
        List<ServerInterceptor> interceptors = new ArrayList<>(2);
        if (security.authToken() != null) {
            interceptors.add(TokenAuth.server(security.authToken()));
        }
        interceptors.add(GrpcMetrics.server(transportSink)); // last added runs first: rejected calls are counted
        InetSocketAddress address = new InetSocketAddress(options.bindHost(), options.port());
        NettyServerBuilder builder = security.tls()
                ? NettyServerBuilder.forAddress(address, credentials(security))
                : NettyServerBuilder.forAddress(address);
        Server server = builder
                .executor(handlers)
                .maxInboundMessageSize(options.maxInboundMessageBytes())
                .maxConcurrentCallsPerConnection(options.maxConcurrentCallsPerConnection())
                .keepAliveTime(NodeServerOptions.KEEPALIVE_TIME.toNanos(), TimeUnit.NANOSECONDS)
                .keepAliveTimeout(NodeServerOptions.KEEPALIVE_TIMEOUT.toNanos(), TimeUnit.NANOSECONDS)
                .permitKeepAliveTime(NodeServerOptions.PERMIT_KEEPALIVE.toNanos(), TimeUnit.NANOSECONDS)
                .maxConnectionIdle(options.maxConnectionIdle().toNanos(), TimeUnit.NANOSECONDS)
                .maxConnectionAge(options.maxConnectionAge().toNanos(), TimeUnit.NANOSECONDS)
                .maxConnectionAgeGrace(options.maxConnectionAgeGrace().toNanos(), TimeUnit.NANOSECONDS)
                .addService(ServerInterceptors.intercept(service, interceptors))
                .addService(health.getHealthService())
                .build();
        try {
            server.start();
        } catch (IOException | RuntimeException e) {
            service.close();
            handlers.shutdownNow();
            throw e;
        }
        health.setStatus(HealthStatusManager.SERVICE_NAME_ALL_SERVICES, ServingStatus.SERVING);
        health.setStatus(CoresetNodeServiceGrpc.SERVICE_NAME, ServingStatus.SERVING);
        return new CoresetNodeServer(server, service, health, handlers);
    }

    /** Rebuilds the Stats JSON (a full copy of every histogram) at most once per {@link #STATS_TTL}. */
    static Supplier<String> cached(Supplier<String> json) {
        AtomicReference<Cached> last = new AtomicReference<>();
        return () -> {
            long now = System.nanoTime();
            Cached c = last.get();
            if (c == null || now - c.at() > STATS_TTL.toNanos()) {
                c = new Cached(now, json.get());
                last.set(c); // racing callers may both rebuild once; either result is fresh
            }
            return c.json();
        };
    }

    private record Cached(long at, String json) {
    }

    private static ServerCredentials credentials(ServerSecurity security) throws IOException {
        TlsServerCredentials.Builder tls = TlsServerCredentials.newBuilder()
                .keyManager(security.certChain().toFile(), security.privateKey().toFile());
        if (security.clientCa() != null) {
            tls.trustManager(security.clientCa().toFile()).clientAuth(TlsServerCredentials.ClientAuth.REQUIRE);
        }
        return tls.build();
    }

    private static ThreadFactory daemonThreads() {
        AtomicInteger n = new AtomicInteger();
        return r -> {
            Thread t = new Thread(r, "coreset-grpc-handler-" + n.incrementAndGet());
            t.setDaemon(true);
            return t;
        };
    }

    public int port() {
        return server.getPort();
    }

    /** Addresses the server listens on. */
    public List<? extends SocketAddress> listenAddresses() {
        return server.getListenSockets();
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
            handlers.shutdownNow();
        }
    }
}
