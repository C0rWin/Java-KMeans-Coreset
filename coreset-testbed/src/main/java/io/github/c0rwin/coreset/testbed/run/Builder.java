package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.SampledCoreset;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.MetricsSnapshot;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.CoresetNodeServer;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.node.AlgorithmResolver;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.github.c0rwin.coreset.testbed.data.Dataset;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.locks.LockSupport;

/** Builds one coreset in one mode and measures the construction. */
final class Builder {

    private static final String SESSION = "testbed";
    private static final Duration TIMEOUT = Duration.ofMinutes(30);
    private static final long FINAL_REDUCE_SALT = 0xF1A1L;
    private static final long PRODUCER_SALT = 0x9D0CL;

    private final AlgorithmResolver resolver;
    private final boolean exclusive;

    /** @param exclusive runs execute one at a time, so heap peaks and process CPU are attributable */
    Builder(AlgorithmResolver resolver, boolean exclusive) {
        this.resolver = resolver;
        this.exclusive = exclusive;
    }

    /** Stream mode: the tree's only level right after {@code 2^level} blocks, i.e. the summary of those blocks. */
    record LevelCapture(int level, Coreset coreset, int inputEnd) {
    }

    record NodeOutput(int node, Coreset coreset, long batches) {
    }

    /**
     * @param sampled     provenance of an offline sensitivity sample, else null
     * @param bytesOnWire gRPC payload bytes, else null
     */
    record Built(Coreset coreset, SampledCoreset sampled, ResourceProbe.Sample sample, MetricsSnapshot metrics,
                 int levels, List<LevelCapture> captures, List<NodeOutput> nodes, Long bytesOnWire) {

        long distanceEvaluations() {
            long s = 0;
            for (Map.Entry<String, Long> e : metrics.counters().entrySet()) {
                if (e.getKey().endsWith("distance_evaluations")) {
                    s += e.getValue();
                }
            }
            return s;
        }

        long reduceOps() {
            long s = 0;
            for (var e : metrics.distributions().entrySet()) {
                if (e.getKey().endsWith(".compress" + MetricNames.WALL_NS_SUFFIX)) {
                    s += e.getValue().count();
                }
            }
            return s;
        }

        long mergeOps() {
            return metrics.counters().getOrDefault(MetricNames.MRTREE_MERGES, 0L);
        }
    }

    Built build(Dataset ds, String algorithmId, Map<String, String> params, ModeSpec mode, long seed)
            throws Exception {
        return switch (mode.type()) {
            case ModeSpec.OFFLINE -> offline(ds, resolver.resolve(algorithmId, params), seed);
            case ModeSpec.STREAM -> stream(ds, resolver.resolve(algorithmId, params), mode, seed);
            case ModeSpec.DISTRIBUTED -> distributed(ds, algorithmId, params, mode, seed);
            default -> throw new IllegalArgumentException("unknown mode " + mode.type());
        };
    }

    /** End (exclusive) of block {@code i} of {@code b} equal blocks over {@code n} points. */
    static int blockEnd(int n, int b, int i) {
        return (int) ((long) n * (i + 1) / b);
    }

    private Built offline(Dataset ds, CoresetAlgorithm algorithm, long seed) {
        RecordingMetrics metrics = new RecordingMetrics();
        SplittableRandom rng = new SplittableRandom(seed);
        ResourceProbe probe = ResourceProbe.start(ResourceProbe.Scope.THREAD, exclusive);
        Coreset out;
        SampledCoreset sampled = null;
        if (algorithm instanceof SensitivityCoreset s) {
            sampled = s.compressWithProvenance(ds.coreset(), rng, metrics);
            out = sampled.coreset();
        } else {
            out = algorithm.compress(ds.coreset(), rng, metrics);
        }
        ResourceProbe.Sample sample = probe.stop();
        return new Built(out, sampled, sample, metrics.snapshot(), 1, List.of(), List.of(), null);
    }

    private Built stream(Dataset ds, CoresetAlgorithm algorithm, ModeSpec mode, long seed) {
        final List<WeightedPoint> points = ds.points();
        final int n = points.size();
        final int b = Math.min(mode.blockCount(), n);
        RecordingMetrics metrics = new RecordingMetrics();
        SplittableRandom rng = new SplittableRandom(seed);
        List<LevelCapture> captures = new ArrayList<>();
        ResourceProbe probe = ResourceProbe.start(ResourceProbe.Scope.THREAD, exclusive);
        MergeReduceTree tree = new MergeReduceTree(algorithm, rng, metrics);
        int start = 0;
        for (int i = 0; i < b; i++) {
            int end = blockEnd(n, b, i);
            tree.add(algorithm.compress(Coreset.of(points.subList(start, end)), rng, metrics));
            long added = tree.batchesAdded();
            if (Long.bitCount(added) == 1) {
                captures.add(new LevelCapture(Long.numberOfTrailingZeros(added), tree.levelsView().get(0), end));
            }
            start = end;
        }
        Coreset out = mode.finalReduce() ? tree.compressedSnapshot() : tree.snapshot();
        ResourceProbe.Sample sample = probe.stop();
        int levels = 1 + (31 - Integer.numberOfLeadingZeros(b)) + (mode.finalReduce() ? 1 : 0);
        return new Built(out, null, sample, metrics.snapshot(), levels, captures, List.of(), null);
    }

    private Built distributed(Dataset ds, String algorithmId, Map<String, String> params, ModeSpec mode, long seed)
            throws Exception {
        final List<WeightedPoint> points = ds.points();
        final int n = points.size();
        final int b = Math.min(mode.blockCount(), n);
        final int nodesCount = mode.nodes();
        CoresetAlgorithm finalAlgorithm = resolver.resolve(algorithmId, params);
        // Same producer id and per-node partitions on both transports, so they build the identical coreset.
        String producer = "testbed-" + Long.toHexString(Seeds.mix(seed, PRODUCER_SALT));
        SessionSpec spec = new SessionSpec(algorithmId, params, seed, mode.workers(),
                Integer.highestOneBit(Math.max(2, b)) * 2, TreeMode.fromId(mode.treeMode()));
        RecordingMetrics metrics = new RecordingMetrics();
        List<CoresetNode> nodes = new ArrayList<>(nodesCount);
        List<CoresetNodeServer> servers = new ArrayList<>();
        CoresetCoordinator coordinator = null;
        RecordingMetrics wire = new RecordingMetrics();
        try {
            for (int i = 0; i < nodesCount; i++) {
                nodes.add(new CoresetNode(resolver, metrics));
            }
            boolean grpc = mode.transport().equals("grpc");
            if (grpc) {
                List<String> targets = new ArrayList<>();
                for (CoresetNode node : nodes) {
                    CoresetNodeServer s = CoresetNodeServer.start(
                            io.github.c0rwin.coreset.grpc.NodeServerOptions.builder().port(0).build(), node,
                            new RecordingMetrics(), MetricsSink.NOOP);
                    servers.add(s);
                    targets.add("127.0.0.1:" + s.port());
                }
                coordinator = CoresetCoordinator.connect(targets, CoordinatorOptions.defaults(),
                        io.github.c0rwin.coreset.grpc.ClientSecurity.plaintext(), wire);
            }
            ResourceProbe probe = ResourceProbe.start(ResourceProbe.Scope.PROCESS, exclusive);
            Coreset union;
            List<NodeSnapshot> snapshots;
            if (grpc) {
                coordinator.createSession(SESSION, spec);
                try (IngestStream stream = coordinator.openIngest(SESSION, producer)) {
                    int start = 0;
                    for (int i = 0; i < b; i++) {
                        int end = blockEnd(n, b, i);
                        stream.submit(points.subList(start, end));
                        start = end;
                    }
                    stream.flush(TIMEOUT);
                }
                GatheredSnapshot g = coordinator.gather(SESSION, TIMEOUT);
                if (!g.complete()) {
                    throw new IllegalStateException("gathered snapshot incomplete: " + g.appliedBatches() + " of "
                            + g.acceptedBatches() + " batches applied");
                }
                union = g.coreset();
                snapshots = g.nodes();
            } else {
                for (int i = 0; i < nodesCount; i++) {
                    nodes.get(i).createSession(SESSION, spec, i); // partition i, as the coordinator assigns
                }
                long[] nextSeq = new long[nodesCount];
                int start = 0;
                for (int i = 0; i < b; i++) {
                    int end = blockEnd(n, b, i);
                    int target = i % nodesCount; // round-robin with per-node dense seqs, as the gRPC coordinator does
                    ingest(nodes.get(target), producer, nextSeq[target]++, points.subList(start, end));
                    start = end;
                }
                snapshots = new ArrayList<>(nodesCount);
                int total = 0;
                double offset = 0.0;
                for (CoresetNode node : nodes) {
                    if (!node.awaitApplied(SESSION, TIMEOUT)) {
                        throw new IllegalStateException("node did not apply its batches within " + TIMEOUT);
                    }
                    NodeSnapshot s = node.snapshot(SESSION);
                    if (!s.complete()) {
                        throw new IllegalStateException("node snapshot incomplete: " + s);
                    }
                    snapshots.add(s);
                    total += s.coreset().size();
                    offset += s.coreset().offset();
                }
                // One pre-sized concatenation, as the coordinator's gather does, instead of re-copying per node.
                WeightedPoint[] all = new WeightedPoint[total];
                int at = 0;
                for (NodeSnapshot s : snapshots) {
                    for (WeightedPoint p : s.coreset().points()) {
                        all[at++] = p;
                    }
                }
                union = new Coreset(List.of(all), offset);
            }
            Coreset out = mode.finalReduce() && !union.isEmpty()
                    ? finalAlgorithm.compress(union, new SplittableRandom(Seeds.mix(seed, FINAL_REDUCE_SALT)),
                    metrics)
                    : union;
            ResourceProbe.Sample sample = probe.stop();
            List<NodeOutput> perNode = new ArrayList<>(snapshots.size());
            for (int i = 0; i < snapshots.size(); i++) {
                perNode.add(new NodeOutput(i, snapshots.get(i).coreset(), snapshots.get(i).appliedBatches()));
            }
            int perWorker = (int) Math.ceil((double) b / ((long) nodesCount * mode.workers()));
            int levels = 1 + (31 - Integer.numberOfLeadingZeros(Math.max(1, perWorker)))
                    + (mode.finalReduce() ? 1 : 0);
            Long wireBytes = grpc ? wireBytes(wire.snapshot()) : null;
            if (grpc) {
                coordinator.closeSession(SESSION);
            }
            return new Built(out, null, sample, metrics.snapshot(), levels, List.of(), perNode, wireBytes);
        } finally {
            if (coordinator != null) {
                coordinator.close();
            }
            for (CoresetNodeServer s : servers) {
                s.close();
            }
            for (CoresetNode node : nodes) {
                node.close();
            }
        }
    }

    private static void ingest(CoresetNode node, String producer, long seq, List<WeightedPoint> batch) {
        for (;;) {
            IngestResult r = node.ingest(SESSION, producer, seq, batch);
            switch (r.status()) {
                case ACCEPTED, DUPLICATE -> {
                    return;
                }
                case BUSY -> LockSupport.parkNanos(50_000L); // backpressure: retry the same seq
                case FAILED -> throw new IllegalStateException("ingest failed: " + r.reason());
            }
        }
    }

    private static long wireBytes(MetricsSnapshot wire) {
        long s = 0;
        for (Map.Entry<String, Long> e : wire.counters().entrySet()) {
            if (e.getKey().endsWith(".bytes_in") || e.getKey().endsWith(".bytes_out")) {
                s += e.getValue();
            }
        }
        return s;
    }
}
