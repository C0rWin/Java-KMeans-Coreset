package io.github.c0rwin.coreset.grpc.cli;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.ClientSecurity;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Spec;

/**
 * {@code coreset-coordinator}: streams a points file to the nodes, gathers their coresets, clusters the union with
 * k-means and reports the centers and the distributed coreset cost.
 */
@Command(name = "coreset-coordinator", mixinStandardHelpOptions = true,
        description = "Distributes a points file over coreset nodes and clusters the gathered coreset.")
public final class CoordinatorCommand implements Callable<Integer> {

    @Spec
    CommandLine.Model.CommandSpec spec;

    @Option(names = "--nodes", required = true, split = ",", description = "Node targets host:port, comma separated.")
    List<String> nodes;

    @Option(names = "--input", required = true, description = "Points file (dense CSV/whitespace, or sparse).")
    Path input;

    @Option(names = "--sparse-dim", defaultValue = "0",
            description = "Read index:value lines of this dimension; 0 = dense lines. Default: ${DEFAULT-VALUE}")
    int sparseDim;

    @Option(names = "--algorithm", required = true, description = "Node-side coreset algorithm id.")
    String algorithm;

    @Option(names = "--param", description = "Algorithm parameter key=value (repeatable).")
    Map<String, String> params = new LinkedHashMap<>();

    @Option(names = "--session", description = "Session id. Default: random UUID.")
    String session;

    @Option(names = "--seed", defaultValue = "1", description = "Session and clustering seed. Default: ${DEFAULT-VALUE}")
    long seed;

    @Option(names = "--workers", defaultValue = "4", description = "Workers per node. Default: ${DEFAULT-VALUE}")
    int workers;

    @Option(names = "--queue-capacity", defaultValue = "64",
            description = "Per-worker queue capacity (power of two). Default: ${DEFAULT-VALUE}")
    int queueCapacity;

    @Option(names = "--tree-mode", defaultValue = "single-writer",
            description = "single-writer or cas. Default: ${DEFAULT-VALUE}")
    String treeMode;

    @Option(names = "--batch-size", defaultValue = "1000", description = "Points per batch. Default: ${DEFAULT-VALUE}")
    int batchSize;

    @Option(names = "--max-in-flight", defaultValue = "16",
            description = "Unacknowledged batches per node. Default: ${DEFAULT-VALUE}")
    int maxInFlight;

    @Option(names = "--k", required = true, description = "Number of centers.")
    int k;

    @Option(names = "--iterations", defaultValue = "50", description = "Lloyd iterations. Default: ${DEFAULT-VALUE}")
    int iterations;

    @Option(names = "--restarts", defaultValue = "10",
            description = "k-means++ and Lloyd restarts; the cheapest is kept. Default: ${DEFAULT-VALUE}")
    int restarts;

    @Option(names = "--final-algorithm", description = "Compress the gathered union once more with this algorithm.")
    String finalAlgorithm;

    @Option(names = "--final-param", description = "Final algorithm parameter key=value (repeatable).")
    Map<String, String> finalParams = new LinkedHashMap<>();

    @Option(names = "--timeout-ms", defaultValue = "600000",
            description = "Bound on flushing ingest and on awaiting applied batches. Default: ${DEFAULT-VALUE}")
    long timeoutMs;

    @Option(names = "--keep-session", description = "Leave the session open on the nodes.")
    boolean keepSession;

    @Option(names = "--producer-id",
            description = "Producer id of this run's batches; fixed ids make runs reproducible. Default: random.")
    String producerId;

    @Option(names = "--tls-ca", description = "PEM CAs trusted for node certificates; enables TLS.")
    Path tlsCa;

    @Option(names = "--tls-cert", description = "PEM client certificate chain for mutual TLS (with --tls-key).")
    Path tlsCert;

    @Option(names = "--tls-key", description = "PEM PKCS#8 private key of --tls-cert.")
    Path tlsKey;

    @Option(names = "--auth-token-file", description = "File holding the nodes' shared auth token.")
    Path authTokenFile;

    @Override
    public Integer call() throws Exception {
        PrintWriter out = spec.commandLine().getOut();
        CoresetAlgorithm finalCompress = finalAlgorithm == null ? null
                : CoresetAlgorithms.load().create(finalAlgorithm, finalParams);
        String sessionId = session != null ? session : UUID.randomUUID().toString();
        SessionSpec sessionSpec = new SessionSpec(algorithm, params, seed, workers, queueCapacity,
                TreeMode.fromId(treeMode));
        Duration timeout = Duration.ofMillis(timeoutMs);
        CoordinatorOptions options = CoordinatorOptions.defaults().withMaxInFlightPerNode(maxInFlight);
        String token = authTokenFile == null ? null : Files.readString(authTokenFile, StandardCharsets.UTF_8).strip();
        ClientSecurity security = new ClientSecurity(tlsCa, tlsCert, tlsKey, token);
        try (CoresetCoordinator coordinator = CoresetCoordinator.connect(nodes, options, security, MetricsSink.NOOP);
             SessionLease ignored = SessionLease.create(coordinator, sessionId, sessionSpec, keepSession)) {
            long points = 0;
            try (IngestStream stream = producerId == null ? coordinator.openIngest(sessionId)
                    : coordinator.openIngest(sessionId, producerId);
                 BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
                PointsReader pr = new PointsReader(reader, sparseDim);
                for (List<WeightedPoint> batch = pr.next(batchSize); !batch.isEmpty(); batch = pr.next(batchSize)) {
                    stream.submit(batch);
                    points += batch.size();
                }
                stream.flush(timeout);
            }
            GatheredSnapshot snapshot = coordinator.gather(sessionId, timeout);
            if (!snapshot.complete()) {
                spec.commandLine().getErr().printf("warning: snapshot incomplete (%d of %d batches applied)%n",
                        snapshot.appliedBatches(), snapshot.acceptedBatches());
            }
            KMeansResult result = CoresetCoordinator.cluster(snapshot.coreset(), k, iterations, restarts, seed,
                    finalCompress);
            double cost = coordinator.cost(sessionId, result.centers());
            out.printf("# session=%s points=%d batches=%d coreset_points=%d cost=%s%n", sessionId, points,
                    snapshot.acceptedBatches(), snapshot.coreset().size(), cost);
            for (SparseVector c : result.centers()) {
                double[] d = c.toDense();
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < d.length; i++) {
                    if (i > 0) {
                        sb.append(',');
                    }
                    sb.append(d[i]);
                }
                out.println(sb);
            }
            out.flush();
        }
        return 0;
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new CoordinatorCommand()).execute(args));
    }

    /** Creates the session and closes it again on every exit path, unless it is to be kept. */
    private record SessionLease(CoresetCoordinator coordinator, String sessionId, boolean keep)
            implements AutoCloseable {

        static SessionLease create(CoresetCoordinator coordinator, String sessionId, SessionSpec spec, boolean keep)
                throws InterruptedException {
            try {
                coordinator.createSession(sessionId, spec);
            } catch (RuntimeException e) {
                if (!keep) { // some nodes may have created it before another refused
                    try {
                        coordinator.closeSession(sessionId);
                    } catch (RuntimeException suppressed) {
                        e.addSuppressed(suppressed);
                    }
                }
                throw e;
            }
            return new SessionLease(coordinator, sessionId, keep);
        }

        @Override
        public void close() throws InterruptedException {
            if (!keep) {
                coordinator.closeSession(sessionId);
            }
        }
    }
}
