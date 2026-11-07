package io.github.c0rwin.coreset.bench.perf;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.Distribution;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.MetricsSnapshot;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.CoresetNodeServer;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.node.CoreAlgorithmsResolver;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.NodeMetricNames;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;

/**
 * Ingestion throughput and latency of {@link CoresetNode}: in process with W workers per tree mode, and through gRPC
 * over Netty loopback with 1..N nodes (client and nodes share this JVM). Every configuration runs once to warm up and
 * then {@code --repeats} times; tables report the median with the min-max range of points/s.
 *
 * <p>Producers cycle over a pool of distinct pre-generated batches with unique seqs, so the data set stays small while
 * every batch is compressed with its own seq-derived generator. A BUSY ingest is retried after {@code --busy-backoff-us}.
 */
public final class NodeIngestHarness {

    private static final String SESSION = "bench";

    private final Map<String, String> opts;
    private final List<List<WeightedPoint>> pool;
    private final int totalBatches;
    private final long totalPoints;
    private final int repeats;

    private NodeIngestHarness(Map<String, String> opts) {
        this.opts = opts;
        int batch = Integer.parseInt(opts.get("batch"));
        int poolBatches = Integer.parseInt(opts.get("pool-batches"));
        this.pool = Workloads.batches(Workloads.gaussianMixture(poolBatches * batch, Integer.parseInt(opts.get("d")),
                Integer.parseInt(opts.get("clusters")), 21L), batch);
        this.totalBatches = (int) (Long.parseLong(opts.get("points")) / batch);
        this.totalPoints = (long) totalBatches * batch;
        this.repeats = Integer.parseInt(opts.get("repeats"));
    }

    public static void main(String[] argv) throws Exception {
        Map<String, String> opts = HarnessSupport.args(argv, Map.ofEntries(
                Map.entry("points", "1000000"),
                Map.entry("batch", "1000"),
                Map.entry("pool-batches", "256"),
                Map.entry("d", "16"),
                Map.entry("clusters", "10"),
                Map.entry("algorithm", "sensitivity"),
                Map.entry("k", "10"),
                Map.entry("m", "200"),
                Map.entry("workers", "1,2,4,8"),
                Map.entry("modes", "single-writer,cas"),
                Map.entry("producers", "2"),
                Map.entry("queue", "64"),
                Map.entry("busy-backoff-us", "50"),
                Map.entry("grpc-nodes", "1,2,3,4"),
                Map.entry("grpc-workers", "2"),
                Map.entry("repeats", "3")));
        HarnessSupport.quietLogs();
        HarnessSupport.banner("NodeIngestHarness " + opts);
        NodeIngestHarness h = new NodeIngestHarness(opts);
        h.inProcess();
        if (!opts.get("grpc-nodes").isBlank() && !opts.get("grpc-nodes").equals("0")) {
            h.grpc();
        }
    }

    Map<String, String> algorithmParams() {
        String alg = opts.get("algorithm");
        return alg.equals("uniform") ? Map.of("m", opts.get("m")) : Map.of("k", opts.get("k"), "m", opts.get("m"));
    }

    // ---------------------------------------------------------------- in process

    record InProcessResult(double pointsPerSec, double ingestP50Ns, double ingestP99Ns, double compressP50Us,
                           double compressP99Us, double busyRate, ResourceProbe.Usage usage, int coresetSize,
                           double weightError) {
    }

    private void inProcess() throws InterruptedException {
        System.out.println();
        System.out.printf("## In-process CoresetNode: %d points in batches of %d, %s %s, %s producers, queue %s%n",
                totalPoints, pool.get(0).size(), opts.get("algorithm"), algorithmParams(), opts.get("producers"),
                opts.get("queue"));
        System.out.println();
        System.out.println("| mode | W | points/s (median) | min-max | ingest p50 / p99 (us) | batch compress p50 / p99 (us)"
                + " | BUSY rate | alloc MB/s | GC n / ms | peak heap MiB (baseline) | snapshot pts | weight err |");
        System.out.println("|---|---|---|---|---|---|---|---|---|---|---|---|");
        for (String modeId : opts.get("modes").split(",")) {
            TreeMode mode = TreeMode.fromId(modeId.trim());
            for (int w : HarnessSupport.ints(opts.get("workers"))) {
                runInProcess(mode, w); // warmup
                List<Double> rates = new ArrayList<>();
                List<InProcessResult> results = new ArrayList<>();
                for (int r = 0; r < repeats; r++) {
                    InProcessResult res = runInProcess(mode, w);
                    rates.add(res.pointsPerSec());
                    results.add(res);
                }
                InProcessResult med = closestToMedian(results, rates);
                ResourceProbe.Usage u = med.usage();
                System.out.printf("| %s | %d | %.0f | %.0f-%.0f | %.1f / %.1f | %.0f / %.0f | %.4f | %.0f | %d / %d | %s (%s) | %d"
                                + " | %.1e |%n",
                        mode.id(), w, HarnessSupport.median(rates), HarnessSupport.min(rates),
                        HarnessSupport.max(rates), med.ingestP50Ns() / 1e3, med.ingestP99Ns() / 1e3,
                        med.compressP50Us(), med.compressP99Us(), med.busyRate(), u.allocRateMBs(), u.gcCount(),
                        u.gcMillis(), HarnessSupport.mib(u.peakHeapBytes()), HarnessSupport.mib(u.baselineHeapBytes()),
                        med.coresetSize(), med.weightError());
            }
        }
    }

    private static <T> T closestToMedian(List<T> results, List<Double> rates) {
        double med = HarnessSupport.median(rates);
        int best = 0;
        for (int i = 1; i < rates.size(); i++) {
            if (Math.abs(rates.get(i) - med) < Math.abs(rates.get(best) - med)) {
                best = i;
            }
        }
        return results.get(best);
    }

    InProcessResult runInProcess(TreeMode mode, int workers) throws InterruptedException {
        RecordingMetrics metrics = new RecordingMetrics();
        int queue = Integer.parseInt(opts.get("queue"));
        int producers = Integer.parseInt(opts.get("producers"));
        long backoffNs = Long.parseLong(opts.get("busy-backoff-us")) * 1000L;
        try (CoresetNode node = new CoresetNode(new CoreAlgorithmsResolver(), metrics)) {
            node.createSession(SESSION, new SessionSpec(opts.get("algorithm"), algorithmParams(), 99L, workers, queue,
                    mode));
            AtomicLong busy = new AtomicLong();
            ResourceProbe probe = ResourceProbe.start();
            long t0 = System.nanoTime();
            Thread[] threads = new Thread[producers];
            for (int p = 0; p < producers; p++) {
                final int first = p;
                threads[p] = new Thread(() -> {
                    long localBusy = 0;
                    for (int b = first; b < totalBatches; b += producers) {
                        List<WeightedPoint> batch = pool.get(b % pool.size());
                        for (;;) {
                            IngestResult r = node.ingest(SESSION, b, batch);
                            if (r.status() == IngestResult.Status.ACCEPTED) {
                                break;
                            }
                            if (r.status() != IngestResult.Status.BUSY) {
                                throw new IllegalStateException("ingest " + b + ": " + r);
                            }
                            localBusy++;
                            LockSupport.parkNanos(backoffNs);
                        }
                    }
                    busy.addAndGet(localBusy);
                }, "producer-" + p);
                threads[p].start();
            }
            for (Thread t : threads) {
                t.join();
            }
            if (!node.awaitApplied(SESSION, Duration.ofMinutes(5))) {
                throw new IllegalStateException("batches not applied within 5 minutes");
            }
            double seconds = (System.nanoTime() - t0) / 1e9;
            ResourceProbe.Usage usage = probe.stop();
            NodeSnapshot snap = node.snapshot(SESSION);
            if (snap.appliedBatches() != totalBatches) {
                throw new IllegalStateException("applied " + snap.appliedBatches() + " of " + totalBatches);
            }
            MetricsSnapshot ms = metrics.snapshot();
            Distribution ingest = ms.distributions().get(NodeMetricNames.INGEST_LATENCY);
            Distribution compress = ms.distributions().get(NodeMetricNames.BATCH_COMPRESS + MetricNames.WALL_NS_SUFFIX);
            long b = busy.get();
            return new InProcessResult(totalPoints / seconds, ingest.p50(), ingest.p99(), compress.p50() / 1e3,
                    compress.p99() / 1e3, b / (double) (b + totalBatches), usage, snap.coreset().size(),
                    weightError(snap.coreset(), totalPoints));
        }
    }

    static double weightError(Coreset c, double expected) {
        return Math.abs(c.totalWeight() - expected) / expected;
    }

    // ---------------------------------------------------------------- gRPC

    private void grpc() throws Exception {
        int workers = Integer.parseInt(opts.get("grpc-workers"));
        System.out.println();
        System.out.printf("## gRPC over Netty loopback: %d points, %d workers per node (single-writer), %s producers,"
                + " client and nodes in one JVM%n", totalPoints, workers, opts.get("producers"));
        System.out.println();
        System.out.println("| nodes | points/s (median) | min-max | node ingest p50 / p99 (us) | BUSY resends | alloc MB/s"
                + " | GC n / ms | peak heap MiB (baseline) | proto bytes out / point | snapshot pts | weight err |");
        System.out.println("|---|---|---|---|---|---|---|---|---|---|---|");
        for (int nodes : HarnessSupport.ints(opts.get("grpc-nodes"))) {
            runGrpc(nodes, workers); // warmup
            List<Double> rates = new ArrayList<>();
            List<GrpcResult> results = new ArrayList<>();
            for (int r = 0; r < repeats; r++) {
                GrpcResult res = runGrpc(nodes, workers);
                rates.add(res.pointsPerSec());
                results.add(res);
            }
            GrpcResult med = closestToMedian(results, rates);
            ResourceProbe.Usage u = med.usage();
            System.out.printf("| %d | %.0f | %.0f-%.0f | %.1f / %.1f | %d | %.0f | %d / %d | %s (%s) | %.1f | %d | %.1e |%n",
                    nodes, HarnessSupport.median(rates), HarnessSupport.min(rates), HarnessSupport.max(rates),
                    med.ingestP50Ns() / 1e3, med.ingestP99Ns() / 1e3, med.busyResends(), u.allocRateMBs(),
                    u.gcCount(), u.gcMillis(), HarnessSupport.mib(u.peakHeapBytes()),
                    HarnessSupport.mib(u.baselineHeapBytes()), med.bytesOut() / (double) totalPoints,
                    med.coresetSize(), med.weightError());
        }
    }

    record GrpcResult(double pointsPerSec, double ingestP50Ns, double ingestP99Ns, long busyResends,
                      ResourceProbe.Usage usage, long bytesOut, int coresetSize, double weightError) {
    }

    GrpcResult runGrpc(int nodes, int workers) throws Exception {
        int queue = Integer.parseInt(opts.get("queue"));
        int producers = Integer.parseInt(opts.get("producers"));
        List<CoresetNode> coreNodes = new ArrayList<>();
        List<RecordingMetrics> nodeMetrics = new ArrayList<>();
        List<CoresetNodeServer> servers = new ArrayList<>();
        List<String> targets = new ArrayList<>();
        try {
            for (int i = 0; i < nodes; i++) {
                RecordingMetrics m = new RecordingMetrics();
                CoresetNode node = new CoresetNode(new CoreAlgorithmsResolver(), m);
                coreNodes.add(node);
                nodeMetrics.add(m);
                CoresetNodeServer server = CoresetNodeServer.start(
                        io.github.c0rwin.coreset.grpc.NodeServerOptions.builder().port(0).build(), node, m,
                        MetricsSink.NOOP);
                servers.add(server);
                targets.add("127.0.0.1:" + server.port());
            }
            RecordingMetrics client = new RecordingMetrics();
            try (CoresetCoordinator coord = CoresetCoordinator.connect(targets, CoordinatorOptions.defaults(),
                    io.github.c0rwin.coreset.grpc.ClientSecurity.plaintext(), client)) {
                coord.createSession(SESSION, new SessionSpec(opts.get("algorithm"), algorithmParams(), 99L, workers,
                        queue, TreeMode.SINGLE_WRITER));
                ResourceProbe probe = ResourceProbe.start();
                long t0 = System.nanoTime();
                GatheredSnapshot gathered;
                long busyResends;
                try (IngestStream stream = coord.openIngest(SESSION)) {
                    Thread[] threads = new Thread[producers];
                    AtomicLong errors = new AtomicLong();
                    for (int p = 0; p < producers; p++) {
                        final int first = p;
                        threads[p] = new Thread(() -> {
                            try {
                                for (int b = first; b < totalBatches; b += producers) {
                                    stream.submit(pool.get(b % pool.size()));
                                }
                            } catch (Exception e) {
                                errors.incrementAndGet();
                                e.printStackTrace();
                            }
                        }, "grpc-producer-" + p);
                        threads[p].start();
                    }
                    for (Thread t : threads) {
                        t.join();
                    }
                    if (errors.get() > 0) {
                        throw new IllegalStateException(errors.get() + " producers failed");
                    }
                    stream.flush(Duration.ofMinutes(5));
                    busyResends = stream.busyRetries();
                }
                gathered = coord.gather(SESSION, Duration.ofMinutes(5));
                double seconds = (System.nanoTime() - t0) / 1e9;
                ResourceProbe.Usage usage = probe.stop();
                if (!gathered.complete()) {
                    throw new IllegalStateException("gathered snapshot incomplete");
                }
                double p50 = 0.0;
                double p99 = 0.0;
                for (RecordingMetrics m : nodeMetrics) {
                    Distribution d = m.snapshot().distributions().get(NodeMetricNames.INGEST_LATENCY);
                    p50 = Math.max(p50, d.p50());
                    p99 = Math.max(p99, d.p99());
                }
                long bytesOut = 0;
                for (Map.Entry<String, Long> e : client.snapshot().counters().entrySet()) {
                    if (e.getKey().startsWith("grpc.client.Ingest.bytes_out")) {
                        bytesOut += e.getValue();
                    }
                }
                coord.closeSession(SESSION);
                return new GrpcResult(totalPoints / seconds, p50, p99, busyResends, usage, bytesOut,
                        gathered.coreset().size(), weightError(gathered.coreset(), totalPoints));
            }
        } finally {
            for (CoresetNodeServer s : servers) {
                s.close();
            }
            for (CoresetNode n : coreNodes) {
                n.close();
            }
        }
    }
}
