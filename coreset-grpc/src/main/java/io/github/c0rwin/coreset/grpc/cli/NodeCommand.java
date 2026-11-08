package io.github.c0rwin.coreset.grpc.cli;

import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.grpc.CoresetNodeServer;
import io.github.c0rwin.coreset.grpc.MetricsHttpServer;
import io.github.c0rwin.coreset.grpc.MicrometerMetricsSink;
import io.github.c0rwin.coreset.grpc.NodeServerOptions;
import io.github.c0rwin.coreset.grpc.ServerSecurity;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.NodeLimits;
import io.github.c0rwin.coreset.node.TeeMetricsSink;
import io.micrometer.prometheusmetrics.PrometheusConfig;
import io.micrometer.prometheusmetrics.PrometheusMeterRegistry;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * {@code coreset-node}: serves one {@link CoresetNode} over gRPC and Prometheus metrics over HTTP. Algorithms come
 * from {@link CoresetAlgorithms#load()}: the built-ins plus every provider jar on the class path.
 */
@Command(name = "coreset-node", mixinStandardHelpOptions = true,
        description = "Coreset ingestion node: gRPC service plus Prometheus /metrics.")
public final class NodeCommand implements Callable<Integer> {

    @Option(names = "--port", defaultValue = "7070", description = "gRPC port (0 = ephemeral). Default: ${DEFAULT-VALUE}")
    int port;

    @Option(names = "--bind", defaultValue = "127.0.0.1",
            description = "gRPC listen address; 0.0.0.0 exposes the node (use TLS and a token). Default: ${DEFAULT-VALUE}")
    String bind;

    @Option(names = "--metrics-port", defaultValue = "9464",
            description = "HTTP port for /metrics (0 = ephemeral, -1 = disabled). Default: ${DEFAULT-VALUE}")
    int metricsPort;

    @Option(names = "--metrics-bind", defaultValue = "127.0.0.1",
            description = "/metrics listen address; it has no authentication. Default: ${DEFAULT-VALUE}")
    String metricsBind;

    @Option(names = "--tls-cert", description = "PEM certificate chain; with --tls-key enables TLS.")
    Path tlsCert;

    @Option(names = "--tls-key", description = "PEM PKCS#8 private key of --tls-cert.")
    Path tlsKey;

    @Option(names = "--tls-client-ca", description = "PEM CAs for client certificates; requires them (mutual TLS).")
    Path tlsClientCa;

    @Option(names = "--auth-token-file",
            description = "File holding a shared token every call must present as 'authorization: Bearer <token>'.")
    Path authTokenFile;

    @Option(names = "--handler-threads", defaultValue = "0",
            description = "gRPC callback threads (0 = max(4, 2 x cores)). Default: ${DEFAULT-VALUE}")
    int handlerThreads;

    @Option(names = "--blocking-threads", defaultValue = "0",
            description = "Threads for snapshot, cost and session calls (0 = max(4, cores)). Default: ${DEFAULT-VALUE}")
    int blockingThreads;

    @Option(names = "--max-connection-idle-ms", defaultValue = "1800000",
            description = "Close client connections idle this long. Default: ${DEFAULT-VALUE}")
    long maxConnectionIdleMs;

    @Option(names = "--max-connection-age-ms", defaultValue = "7200000",
            description = "Recycle client connections after this long. Default: ${DEFAULT-VALUE}")
    long maxConnectionAgeMs;

    @Option(names = "--max-connection-age-grace-ms", defaultValue = "900000",
            description = "Time in-flight calls get when a connection is recycled. Default: ${DEFAULT-VALUE}")
    long maxConnectionAgeGraceMs;

    @Option(names = "--shutdown-grace-ms", defaultValue = "10000",
            description = "Wait for busy workers before interrupting them on shutdown. Default: ${DEFAULT-VALUE}")
    long shutdownGraceMs;

    @Option(names = "--max-message-bytes", defaultValue = "16777216",
            description = "Largest inbound gRPC request (batch, cost shard). Default: ${DEFAULT-VALUE}")
    int maxMessageBytes;

    @Option(names = "--max-dim", defaultValue = "16777216",
            description = "Largest vector dimension (dense buffers are O(dim)). Default: ${DEFAULT-VALUE}")
    int maxDim;

    @Option(names = "--max-points-per-batch", defaultValue = "262144",
            description = "Points per batch or raw cost shard. Default: ${DEFAULT-VALUE}")
    int maxPointsPerBatch;

    @Option(names = "--max-sessions", defaultValue = "64", description = "Open sessions. Default: ${DEFAULT-VALUE}")
    int maxSessions;

    @Option(names = "--max-total-workers", defaultValue = "4096",
            description = "Worker threads over all sessions. Default: ${DEFAULT-VALUE}")
    int maxTotalWorkers;

    @Option(names = "--max-queue-capacity", defaultValue = "4096",
            description = "Per-worker queue capacity (power of two). Default: ${DEFAULT-VALUE}")
    int maxQueueCapacity;

    @Option(names = "--max-cost-work", defaultValue = "2147483648",
            description = "Points x centers per cost evaluation. Default: ${DEFAULT-VALUE}")
    long maxCostWork;

    @Option(names = "--max-snapshot-points", defaultValue = "4194304",
            description = "Points in one session snapshot. Default: ${DEFAULT-VALUE}")
    int maxSnapshotPoints;

    @Option(names = "--max-await-ms", defaultValue = "600000",
            description = "Longest awaitApplied wait a client may request. Default: ${DEFAULT-VALUE}")
    long maxAwaitMs;

    @Option(names = "--max-seq-window", defaultValue = "4096",
            description = "Dedupe window per producer; must exceed the senders' in-flight batches. "
                    + "Default: ${DEFAULT-VALUE}")
    int maxSeqWindow;

    @Option(names = "--session-idle-timeout-ms", defaultValue = "3600000",
            description = "Close sessions without ingest progress or reads for this long (0 = never). "
                    + "Default: ${DEFAULT-VALUE}")
    long sessionIdleTimeoutMs;

    NodeLimits limits() {
        return NodeLimits.builder().maxDim(maxDim).maxPointsPerBatch(maxPointsPerBatch).maxSessions(maxSessions)
                .maxTotalWorkers(maxTotalWorkers).maxQueueCapacity(maxQueueCapacity).maxCostWork(maxCostWork)
                .maxSnapshotPoints(maxSnapshotPoints).maxAwait(Duration.ofMillis(maxAwaitMs))
                .maxSeqWindow(maxSeqWindow).sessionIdleTimeout(Duration.ofMillis(sessionIdleTimeoutMs)).build();
    }

    /** A started node; {@link #close()} stops servers, then sessions. */
    public static final class Running implements AutoCloseable {
        private final CoresetNode node;
        private final CoresetNodeServer server;
        private final MetricsHttpServer metrics;

        Running(CoresetNode node, CoresetNodeServer server, MetricsHttpServer metrics) {
            this.node = node;
            this.server = server;
            this.metrics = metrics;
        }

        public int port() {
            return server.port();
        }

        /** Addresses the gRPC server listens on. */
        public List<? extends SocketAddress> listenAddresses() {
            return server.listenAddresses();
        }

        /** HTTP metrics port, or -1 when disabled. */
        public int metricsPort() {
            return metrics == null ? -1 : metrics.port();
        }

        /** HTTP metrics listen address, or null when disabled. */
        public InetSocketAddress metricsAddress() {
            return metrics == null ? null : metrics.address();
        }

        public void awaitTermination() throws InterruptedException {
            server.awaitTermination();
        }

        @Override
        public void close() {
            server.close();
            if (metrics != null) {
                metrics.close();
            }
            node.close();
        }
    }

    NodeServerOptions serverOptions() throws IOException {
        String token = authTokenFile == null ? null : Files.readString(authTokenFile, StandardCharsets.UTF_8).strip();
        NodeServerOptions.Builder b = NodeServerOptions.builder().bindHost(bind).port(port)
                .maxInboundMessageBytes(maxMessageBytes)
                .security(new ServerSecurity(tlsCert, tlsKey, tlsClientCa, token))
                .maxConnectionIdle(Duration.ofMillis(maxConnectionIdleMs))
                .maxConnectionAge(Duration.ofMillis(maxConnectionAgeMs))
                .maxConnectionAgeGrace(Duration.ofMillis(maxConnectionAgeGraceMs));
        if (handlerThreads > 0) {
            b.handlerThreads(handlerThreads);
        }
        if (blockingThreads > 0) {
            b.blockingThreads(blockingThreads);
        }
        return b.build();
    }

    /** Starts servers without blocking. */
    public Running start(CoresetAlgorithms algorithms) throws IOException {
        NodeServerOptions serverOptions = serverOptions(); // validates the security flags before anything starts
        PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        RecordingMetrics recorder = new RecordingMetrics();
        MetricsSink sink = TeeMetricsSink.of(recorder, new MicrometerMetricsSink(registry));
        CoresetNode node = new CoresetNode(algorithms, sink, Duration.ofMillis(shutdownGraceMs), limits());
        CoresetNodeServer server = null;
        try {
            server = CoresetNodeServer.start(serverOptions, node, recorder, sink);
            MetricsHttpServer http = metricsPort < 0 ? null
                    : MetricsHttpServer.start(new InetSocketAddress(InetAddress.getByName(metricsBind), metricsPort),
                            registry);
            return new Running(node, server, http);
        } catch (IOException | RuntimeException e) {
            if (server != null) {
                server.close();
            }
            node.close();
            throw e;
        }
    }

    @Override
    public Integer call() throws Exception {
        Running running = start(CoresetAlgorithms.load());
        Runtime.getRuntime().addShutdownHook(new Thread(running::close, "coreset-node-shutdown"));
        PrintWriter out = new PrintWriter(System.out, true);
        out.printf("coreset-node listening on %s, metrics on %s%n", running.listenAddresses(),
                running.metricsAddress() == null ? "-" : running.metricsAddress());
        running.awaitTermination();
        return 0;
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new NodeCommand()).execute(args));
    }
}
