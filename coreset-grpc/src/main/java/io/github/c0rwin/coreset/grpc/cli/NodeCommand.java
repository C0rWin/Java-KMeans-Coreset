package io.github.c0rwin.coreset.grpc.cli;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.grpc.CoresetNodeServer;
import io.github.c0rwin.coreset.grpc.MetricsHttpServer;
import io.github.c0rwin.coreset.grpc.MicrometerMetricsSink;
import io.github.c0rwin.coreset.node.AlgorithmResolver;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.ServiceLoaderAlgorithmResolver;
import io.github.c0rwin.coreset.node.TeeMetricsSink;
import io.micrometer.prometheusmetrics.PrometheusConfig;
import io.micrometer.prometheusmetrics.PrometheusMeterRegistry;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

/**
 * {@code coreset-node}: serves one {@link CoresetNode} over gRPC and Prometheus metrics over HTTP. Algorithms come
 * from {@link java.util.ServiceLoader} providers of {@link AlgorithmResolver} on the classpath.
 */
@Command(name = "coreset-node", mixinStandardHelpOptions = true,
        description = "Coreset ingestion node: gRPC service plus Prometheus /metrics.")
public final class NodeCommand implements Callable<Integer> {

    @Option(names = "--port", defaultValue = "7070", description = "gRPC port (0 = ephemeral). Default: ${DEFAULT-VALUE}")
    int port;

    @Option(names = "--metrics-port", defaultValue = "9464",
            description = "HTTP port for /metrics (0 = ephemeral, -1 = disabled). Default: ${DEFAULT-VALUE}")
    int metricsPort;

    @Option(names = "--shutdown-grace-ms", defaultValue = "10000",
            description = "Wait for busy workers before interrupting them on shutdown. Default: ${DEFAULT-VALUE}")
    long shutdownGraceMs;

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

        /** HTTP metrics port, or -1 when disabled. */
        public int metricsPort() {
            return metrics == null ? -1 : metrics.port();
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

    /** Starts servers without blocking. */
    public Running start(AlgorithmResolver resolver) throws IOException {
        PrometheusMeterRegistry registry = new PrometheusMeterRegistry(PrometheusConfig.DEFAULT);
        RecordingMetrics recorder = new RecordingMetrics();
        MetricsSink sink = TeeMetricsSink.of(recorder, new MicrometerMetricsSink(registry));
        CoresetNode node = new CoresetNode(resolver, sink, Duration.ofMillis(shutdownGraceMs));
        CoresetNodeServer server = null;
        try {
            server = CoresetNodeServer.start(port, node, recorder, sink);
            MetricsHttpServer http = metricsPort < 0 ? null
                    : MetricsHttpServer.start(new InetSocketAddress(metricsPort), registry);
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
        Running running = start(ServiceLoaderAlgorithmResolver.load());
        Runtime.getRuntime().addShutdownHook(new Thread(running::close, "coreset-node-shutdown"));
        PrintWriter out = new PrintWriter(System.out, true);
        out.printf("coreset-node listening on %d, metrics on %d%n", running.port(), running.metricsPort());
        running.awaitTermination();
        return 0;
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new NodeCommand()).execute(args));
    }
}
