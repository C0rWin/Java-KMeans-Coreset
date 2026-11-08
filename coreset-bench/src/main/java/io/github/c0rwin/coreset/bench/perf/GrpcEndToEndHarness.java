package io.github.c0rwin.coreset.bench.perf;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.bench.jmh.CodecBenchmark;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.CoresetNodeServer;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * One gRPC {@link CoresetNode} on loopback, driven by a {@link CoresetCoordinator}: end-to-end time and throughput
 * (client conversion, transport, server-side streaming merge-and-reduce, final snapshot gather), bytes on the wire
 * (measured by a counting TCP relay in a separate pass), weight conservation and empirical distortion on a shared
 * candidate set, for {@code sensitivity} and {@code uniform}.
 */
public final class GrpcEndToEndHarness {

    private final Map<String, String> opts;
    private final List<WeightedPoint> points;
    private final List<List<WeightedPoint>> batches;
    private final int k;
    private final int m;
    private final int repeats;

    private GrpcEndToEndHarness(Map<String, String> opts) {
        this.opts = opts;
        this.k = Integer.parseInt(opts.get("k"));
        this.m = Integer.parseInt(opts.get("m"));
        this.repeats = Integer.parseInt(opts.get("repeats"));
        this.points = Workloads.gaussianMixture(Integer.parseInt(opts.get("points")), Integer.parseInt(opts.get("d")),
                Integer.parseInt(opts.get("clusters")), 31L);
        this.batches = Workloads.batches(points, Integer.parseInt(opts.get("batch")));
    }

    public static void main(String[] argv) throws Exception {
        Map<String, String> opts = HarnessSupport.args(argv, Map.of(
                "points", "100000",
                "batch", "1000",
                "d", "16",
                "clusters", "10",
                "k", "10",
                "m", "200",
                "workers", "1,4",
                "repeats", "3",
                "candidates", "3"));
        HarnessSupport.quietLogs();
        HarnessSupport.banner("GrpcEndToEndHarness " + opts);
        codecSizes();
        new GrpcEndToEndHarness(opts).run();
    }

    /** Serialized protobuf bytes of one batch (payload only; framing is in the wire numbers). */
    static void codecSizes() {
        System.out.println();
        System.out.printf("## Serialized size of one %d-point batch (dim %d)%n", CodecBenchmark.BATCH, CodecBenchmark.DIM);
        System.out.println();
        System.out.println("| nnz | protobuf bytes | bytes/point |");
        System.out.println("|---|---|---|");
        for (int nnz : new int[] {16, 128}) {
            List<WeightedPoint> b = CodecBenchmark.batch(nnz, 13L);
            int proto = Batch.newBuilder().setSessionId("bench").setSeq(1).addAllPoints(ProtoCodec.pointsToProto(b))
                    .build().getSerializedSize();
            System.out.printf("| %d | %d | %.1f |%n", nnz, proto, proto / (double) b.size());
        }
    }

    record Outcome(double seconds, Coreset coreset, long bytesToServer, long bytesToClient) {
    }

    private void run() throws Exception {
        Coreset full = Coreset.of(points);
        // Candidates from P are shared by every coreset; each coreset adds the Lloyd solutions deployed from it.
        List<Candidate> common = new ArrayList<>();
        int count = Integer.parseInt(opts.get("candidates"));
        common.addAll(CandidateSolutions.kMeansPlusPlusOnFull(full, k, count, new SplittableRandom(41)));
        common.addAll(CandidateSolutions.lloydOnFull(full, k, 20, count, new SplittableRandom(42)));
        common.addAll(CandidateSolutions.convexCombinationsOfBicriteria(full, k, count, new SplittableRandom(43)));

        System.out.println();
        System.out.printf("## gRPC node end to end: %d points (d=%s, %s clusters) in %d batches, k=%d, m=%d%n",
                points.size(), opts.get("d"), opts.get("clusters"), batches.size(), k, m);
        System.out.println();
        System.out.println("| algorithm | workers | end-to-end s (median) | min-max | points/s | wire bytes"
                + " client->server | server->client | coreset pts | weight err | max distortion | mean distortion |");
        System.out.println("|---|---|---|---|---|---|---|---|---|---|---|");
        for (String algorithm : new String[] {"sensitivity", "uniform"}) {
            for (int w : HarnessSupport.ints(opts.get("workers"))) {
                report(algorithm, w, () -> runGrpc(algorithm, w, false), () -> runGrpc(algorithm, w, true), full,
                        common);
            }
        }
    }

    interface Run {
        Outcome run() throws Exception;
    }

    private void report(String algorithm, int workers, Run timed, Run counted, Coreset full, List<Candidate> common)
            throws Exception {
        timed.run(); // warmup
        List<Double> secs = new ArrayList<>();
        Outcome last = null;
        for (int r = 0; r < repeats; r++) {
            last = timed.run();
            secs.add(last.seconds());
        }
        Outcome wire = counted.run();
        Coreset c = last.coreset();
        List<Candidate> candidates = new ArrayList<>(common);
        candidates.addAll(CandidateSolutions.lloydOnCoreset(c, k, 20, Integer.parseInt(opts.get("candidates")),
                new SplittableRandom(44)));
        Distortion.Report d = Distortion.evaluate(full, c, candidates);
        double med = HarnessSupport.median(secs);
        System.out.printf("| %s | %d | %.3f | %.3f-%.3f | %.0f | %d | %d | %d | %.1e | %.4f | %.4f |%n",
                algorithm, workers, med, HarnessSupport.min(secs), HarnessSupport.max(secs),
                points.size() / med, wire.bytesToServer(), wire.bytesToClient(), c.size(),
                Math.abs(c.totalWeight() - points.size()) / points.size(), d.max(), d.mean());
    }

    private Outcome runGrpc(String algorithm, int workers, boolean countBytes) throws Exception {
        Map<String, String> params = algorithm.equals("uniform")
                ? Map.of("m", Integer.toString(m))
                : Map.of("k", Integer.toString(k), "m", Integer.toString(m));
        try (CoresetNode node = new CoresetNode(CoresetAlgorithms.load(), MetricsSink.NOOP);
             CoresetNodeServer server = CoresetNodeServer.start(
                     io.github.c0rwin.coreset.grpc.NodeServerOptions.builder().port(0).build(), node,
                     new RecordingMetrics(), MetricsSink.NOOP);
             CountingProxy proxy = countBytes ? new CountingProxy(server.port()) : null;
             CoresetCoordinator coord = CoresetCoordinator.connect(
                     List.of("127.0.0.1:" + (proxy == null ? server.port() : proxy.port())),
                     CoordinatorOptions.defaults(), io.github.c0rwin.coreset.grpc.ClientSecurity.plaintext(),
                     MetricsSink.NOOP)) {
            long t0 = System.nanoTime();
            coord.createSession("e2e", new SessionSpec(algorithm, params, 7L, workers, 64, TreeMode.SINGLE_WRITER));
            try (IngestStream stream = coord.openIngest("e2e")) {
                for (List<WeightedPoint> b : batches) {
                    stream.submit(b);
                }
                stream.flush(Duration.ofMinutes(5));
            }
            GatheredSnapshot g = coord.gather("e2e", Duration.ofMinutes(5));
            double seconds = (System.nanoTime() - t0) / 1e9;
            if (!g.complete()) {
                throw new IllegalStateException("gRPC snapshot incomplete");
            }
            return new Outcome(seconds, g.coreset(), proxy == null ? 0 : proxy.bytesToServer(),
                    proxy == null ? 0 : proxy.bytesToClient());
        }
    }
}
