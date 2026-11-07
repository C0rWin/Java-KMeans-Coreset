package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.SampledCoreset;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.CostConfidenceInterval;
import io.github.c0rwin.coreset.core.validate.IntegrityCheck;
import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.node.AlgorithmResolver;
import io.github.c0rwin.coreset.testbed.data.Dataset;
import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.data.Datasets;
import io.github.c0rwin.coreset.testbed.data.Esa22Benchmark;
import io.github.c0rwin.coreset.testbed.eval.CostEvaluator;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import io.github.c0rwin.coreset.testbed.report.ValidityReport;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Consumer;

/**
 * Runs a {@link TestbedConfig}: for every dataset, replicate, algorithm, size and mode it builds the coreset once
 * (once for all k when the algorithm does not take k) and evaluates it for every k, emitting one
 * {@link ValidityReport} per grid point. Runs execute one at a time so construction measurements are attributable;
 * full-data cost passes use all evaluation threads.
 */
public final class Testbed implements AutoCloseable {

    /** Receives reports and shared-candidate summaries as they are produced. */
    public interface Sink {
        void report(ValidityReport report) throws IOException;

        void pool(PoolInfo pool) throws IOException;
    }

    /** Cost of the full-data candidates shared by every run on (dataset, k). */
    public record PoolInfo(String dataset, int k, int candidates, double generationMs, double fullSolveMs,
                           double fullSolutionCost, long distanceEvals) {
    }

    private static final long POOL_SALT = 0x9001L;
    private static final long EVAL_SALT = 0xE7A1L;
    private static final double ONE_SIDED_TOLERANCE = 1e-9;

    private final TestbedConfig config;
    private final Datasets datasets;
    private final Builder builder;
    private final Provenance provenance;
    private final Consumer<String> log;
    private final Problem problem;
    private final ForkJoinPool evaluationPool;
    private final CostEvaluator evaluator;

    public Testbed(TestbedConfig config, Datasets datasets, AlgorithmResolver resolver, Provenance provenance,
                   Consumer<String> log) {
        this.config = config;
        this.datasets = datasets;
        this.builder = new Builder(resolver, true);
        this.provenance = provenance;
        this.log = log;
        this.problem = Problem.of(config.problem());
        int threads = config.evaluation().evaluationThreads() == 0
                ? Runtime.getRuntime().availableProcessors() : config.evaluation().evaluationThreads();
        this.evaluationPool = new ForkJoinPool(threads);
        this.evaluator = new CostEvaluator(problem, evaluationPool);
        for (AlgorithmSpec a : config.algorithms()) {
            if (problem.z() != 2.0 && a.oneSided()) {
                throw new IllegalArgumentException(a.label() + ": the mean coreset's offset is a sum of squared "
                        + "distances and cannot be evaluated for problem " + problem.id());
            }
        }
    }

    /** Number of reports {@link #run} will produce for a dataset of {@code n} points. */
    public int plannedRuns(DatasetSpec spec, int n) {
        int count = 0;
        for (AlgorithmSpec a : config.algorithms()) {
            for (int m : config.sizes()) {
                for (ModeSpec mode : config.modes()) {
                    count += ks(spec.id(), n, a, mode, m).size();
                }
            }
        }
        return count * config.replicates();
    }

    private List<Integer> ks(String datasetId, int n, AlgorithmSpec a, ModeSpec mode, int m) {
        List<Integer> out = new ArrayList<>();
        for (int k : config.ks()) {
            if (k <= n && !config.excluded(datasetId, n, a.label(), mode.label(), k, m)) {
                out.add(k);
            }
        }
        return out;
    }

    public void run(Sink sink) throws Exception {
        for (DatasetSpec spec : config.datasets()) {
            long t0 = System.nanoTime();
            Dataset ds = datasets.load(spec);
            log.accept(String.format(Locale.ROOT, "dataset %s: n=%d d=%d nnz=%d loaded in %.1fs", ds.id(), ds.n(),
                    ds.dim(), ds.nnz(), (System.nanoTime() - t0) / 1e9));
            int total = plannedRuns(spec, ds.n());
            Map<Integer, CandidatePool> pools = new HashMap<>();
            int done = 0;
            for (int r = 0; r < config.replicates(); r++) {
                long seed = Seeds.mix(config.seed(), r);
                for (AlgorithmSpec a : config.algorithms()) {
                    for (int m : config.sizes()) {
                        for (ModeSpec mode : config.modes()) {
                            List<Integer> ks = ks(ds.id(), ds.n(), a, mode, m);
                            if (ks.isEmpty()) {
                                continue;
                            }
                            Builder.Built shared = null;
                            for (int i = 0; i < ks.size(); i++) {
                                int k = ks.get(i);
                                CandidatePool pool = pool(ds, k, pools, sink);
                                Map<String, String> params = a.paramsFor(k, m);
                                Builder.Built built = a.takesK() || shared == null
                                        ? builder.build(ds, a.id(), params, mode, seed) : shared;
                                if (!a.takesK()) {
                                    shared = built;
                                }
                                ValidityReport report = evaluate(ds, pool, a, params, mode, k, m, r, seed, built,
                                        !a.takesK() && i > 0);
                                sink.report(report);
                                done++;
                                log.accept(String.format(Locale.ROOT,
                                        "[%s %d/%d] %s k=%d m=%d %s r=%d: size=%d distortion=%.4f build=%.0fms "
                                                + "eval=%.0fms%s",
                                        ds.id(), done, total, a.label(), k, m, mode.label(), r,
                                        report.construction().outputSize(), report.distortion().max(),
                                        report.construction().wallMs(), report.validation().wallMs(),
                                        report.verdict().pass() ? "" : " FAIL"));
                            }
                        }
                    }
                }
            }
            datasets.evict(spec);
        }
    }

    private CandidatePool pool(Dataset ds, int k, Map<Integer, CandidatePool> pools, Sink sink) throws IOException {
        CandidatePool p = pools.get(k);
        if (p == null) {
            p = CandidatePool.generate(ds, k, config.evaluation(),
                    Seeds.mix(config.seed(), POOL_SALT, Seeds.hash(ds.id()), k), evaluator);
            pools.put(k, p);
            sink.pool(new PoolInfo(ds.id(), k, p.candidates.size(), p.generationMs, p.fullSolveMs,
                    p.fullSolutionCost(), p.distanceEvals));
            log.accept(String.format(Locale.ROOT, "dataset %s k=%d: %d shared candidates in %.1fs (full solve %.1fs)",
                    ds.id(), k, p.candidates.size(), p.generationMs / 1e3, p.fullSolveMs / 1e3));
        }
        return p;
    }

    ValidityReport evaluate(Dataset ds, CandidatePool pool, AlgorithmSpec a, Map<String, String> params,
                            ModeSpec mode, int k, int m, int replicate, long seed, Builder.Built built,
                            boolean reused) throws Exception {
        EvaluationSpec ev = config.evaluation();
        Coreset c = built.coreset();
        long evals0 = evaluator.distanceEvaluations();
        long scanned0 = evaluator.pointsScanned();
        long t0 = System.nanoTime();
        SplittableRandom rng = new SplittableRandom(Seeds.mix(seed, EVAL_SALT, k));

        List<Candidate> runCands = new ArrayList<>(
                CandidateSolutions.lloydOnCoreset(c, k, ev.lloydIterations(), ev.lloydOnCoreset(), rng));
        long s0 = System.nanoTime();
        KMeansResult solved = WeightedLloyd.bestOf(c.points(), k, ev.lloydIterations(), ev.downstreamRestarts(), rng,
                MetricsSink.NOOP);
        double solveMs = (System.nanoTime() - s0) / 1e6;
        runCands.add(new Candidate("bestOf@C", solved.centers()));
        List<Centers> runCenters = new ArrayList<>(runCands.size());
        for (Candidate cand : runCands) {
            runCenters.add(Centers.of(cand.centers()));
        }
        double[] runFull = evaluator.costs(ds.coreset(), runCenters);

        List<Centers> all = new ArrayList<>(pool.centers);
        all.addAll(runCenters);
        double[] onCoreset = evaluator.costs(c, all);
        int poolSize = pool.centers.size();

        SampledCoreset sampled = built.sampled();
        boolean withCi = sampled != null && !sampled.exact() && problem.z() == 2.0;
        List<ValidityReport.Candidate> entries = new ArrayList<>(all.size());
        double max = 0.0;
        double sum = 0.0;
        String argmax = null;
        int covered = 0;
        int intervals = 0;
        int underestimates = 0;
        for (int j = 0; j < all.size(); j++) {
            boolean shared = j < poolSize;
            String generator = shared ? pool.candidates.get(j).generator() : runCands.get(j - poolSize).generator();
            double full = shared ? pool.costFull[j] : runFull[j - poolSize];
            double cc = onCoreset[j];
            double ratio = full > 0.0 ? cc / full : (cc == 0.0 ? 1.0 : Double.POSITIVE_INFINITY);
            double distortion = distortion(ratio);
            List<Double> ci = null;
            if (withCi && shared) { // coreset-optimised candidates are not independent of the sample
                CostConfidenceInterval.Interval iv = CostConfidenceInterval.normal(sampled,
                        pool.candidates.get(j).centers(), ev.confidence());
                ci = List.of(iv.lower(), iv.upper());
                intervals++;
                covered += iv.contains(full) ? 1 : 0;
            }
            if (cc < (1.0 - ONE_SIDED_TOLERANCE) * full) {
                underestimates++;
            }
            entries.add(new ValidityReport.Candidate(generator, full, cc, ratio, distortion, ci));
            sum += distortion;
            if (argmax == null || distortion > max) {
                max = distortion;
                argmax = generator;
            }
        }

        IntegrityCheck.Report integrity = IntegrityCheck.check(ds.moments(), c); // same moments, no input pass
        Optional<Esa22Benchmark.DeltaSweep> sweep = problem.z() == 2.0
                ? ds.benchmark().flatMap(b -> b.deltaSweep(c, Esa22Benchmark.deltaGrid(ev.deltaSteps())))
                : Optional.empty();

        List<ValidityReport.Level> levels = new ArrayList<>();
        if (!built.captures().isEmpty()) {
            double[][] prefix = pool.prefixCosts(ds, Math.min(mode.blockCount(), ds.n()), evaluator);
            for (Builder.LevelCapture cap : built.captures()) {
                double[] costs = evaluator.costs(cap.coreset(), pool.centers);
                double lmax = 0.0;
                String larg = null;
                for (int j = 0; j < costs.length; j++) {
                    double full = prefix[cap.level()][j];
                    double d = distortion(full > 0.0 ? costs[j] / full
                            : (costs[j] == 0.0 ? 1.0 : Double.POSITIVE_INFINITY));
                    if (larg == null || d > lmax) {
                        lmax = d;
                        larg = pool.candidates.get(j).generator();
                    }
                }
                levels.add(new ValidityReport.Level(cap.level(), cap.coreset().size(), serializedBytes(cap.coreset()),
                        cap.inputEnd(), lmax, larg));
            }
        }
        List<ValidityReport.Node> nodes = new ArrayList<>();
        for (Builder.NodeOutput no : built.nodes()) {
            nodes.add(new ValidityReport.Node(no.node(), no.coreset().size(), serializedBytes(no.coreset()),
                    no.batches()));
        }

        String hash = replayHash(c);
        Boolean deterministic = null;
        if (ev.verifyReplay() && !reused) {
            deterministic = hash.equals(replayHash(builder.build(ds, a.id(), params, mode, seed).coreset()));
        }
        double validationMs = (System.nanoTime() - t0) / 1e6 - solveMs;

        Integer oneSided = a.oneSided() ? underestimates : null;
        List<String> notes = new ArrayList<>();
        if (max > ev.distortionThreshold()) {
            notes.add(String.format(Locale.ROOT, "distortion %.4f exceeds threshold %.4f (%s)", max,
                    ev.distortionThreshold(), argmax));
        }
        if (max > 10.0) {
            notes.add("catastrophic failure: distortion > 10");
        } else if (max > 5.0) {
            notes.add("failure: distortion > 5");
        }
        if (integrity.nonFinite() > 0) {
            notes.add(integrity.nonFinite() + " non-finite coreset points");
        }
        if (integrity.negativeWeights() > 0) {
            notes.add(integrity.negativeWeights() + " negative weights");
        }
        if (oneSided != null && oneSided > 0) {
            notes.add(oneSided + " candidates underestimated by a one-sided coreset (missing offset?)");
        }
        if (Boolean.FALSE.equals(deterministic)) {
            notes.add("rebuilding with the same seed gave a different coreset");
        }
        boolean pass = max <= ev.distortionThreshold() && integrity.nonFinite() == 0
                && integrity.negativeWeights() == 0 && (oneSided == null || oneSided == 0)
                && !Boolean.FALSE.equals(deterministic);

        ResourceProbe.Sample s = built.sample();
        ValidityReport.Construction construction = new ValidityReport.Construction(c.size(), built.levels(),
                built.reduceOps(), built.mergeOps(), c.offset(), s.wallMs(), s.cpuMs(), s.processCpuMs(),
                s.allocBytes(), s.peakHeapBytes(), s.baselineHeapBytes(), s.gcCount(), s.gcMs(),
                built.distanceEvaluations(), ds.n(), serializedBytes(c), built.bytesOnWire(), c.estimatedBytes(),
                reused, levels, nodes);
        double fullSolution = pool.fullSolutionCost();
        double coresetSolution = runFull[runFull.length - 1];
        ValidityReport.Downstream downstream = new ValidityReport.Downstream("lloyd+kmeans++",
                ev.downstreamRestarts(), coresetSolution, fullSolution,
                fullSolution > 0.0 ? coresetSolution / fullSolution - 1.0 : 0.0, solveMs, pool.fullSolveMs);
        Integer partitions = ModeSpec.DISTRIBUTED.equals(mode.type()) ? mode.nodes() * mode.workers() : null;
        ValidityReport.Config cfg = new ValidityReport.Config(config.name(), problem.id(), problem.z(), k,
                new ValidityReport.Algorithm(a.label(), a.id(), params), m, epsilon(params), null, seed, replicate,
                mode.type(), mode.label(), mode.blockCount(), partitions, mode.transport(), mode.finalReduce());
        ValidityReport.DatasetInfo dsInfo = new ValidityReport.DatasetInfo(ds.id(), ds.sha256(), ds.n(), ds.dim(),
                ds.nnz(), ds.weightSum(), ds.preprocessing(), ds.license(), ds.source());
        return new ValidityReport(ValidityReport.SCHEMA_VERSION, provenance.run(1), cfg, dsInfo, construction,
                new ValidityReport.Integrity(integrity.weightRelErr(), integrity.firstMomentRelErr(),
                        integrity.secondMomentRelErr(), integrity.negativeWeights(), integrity.nonFinite(), oneSided,
                        hash, deterministic),
                new ValidityReport.DistortionInfo(true, entries, max, sum / all.size(), argmax,
                        intervals > 0 ? (double) covered / intervals : null,
                        sweep.map(d -> new ValidityReport.DeltaSweep(d.delta(), d.clustering(), d.deficientClusters(),
                                d.costFull(), d.costCoreset(), d.distortion())).orElse(null)),
                downstream, new ValidityReport.Validation(validationMs, evaluator.distanceEvaluations() - evals0,
                        (double) (evaluator.pointsScanned() - scanned0) / ds.n()),
                new ValidityReport.Verdict(max, ev.distortionThreshold(), pass, notes));
    }

    private static Double epsilon(Map<String, String> params) {
        String e = params.get("eps");
        return e == null ? null : Double.valueOf(e);
    }

    private static double distortion(double ratio) {
        if (ratio >= 1.0) {
            return ratio;
        }
        return ratio > 0.0 ? 1.0 / ratio : Double.POSITIVE_INFINITY;
    }

    static long serializedBytes(Coreset c) {
        return ProtoCodec.toProto(c).getSerializedSize();
    }

    /** SHA-256 over offset, then per point: weight, dim, nnz, indices, values (big-endian). */
    static String replayHash(Coreset c) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            ByteArrayOutputStream buf = new ByteArrayOutputStream(1 << 12);
            DataOutputStream out = new DataOutputStream(buf);
            out.writeDouble(c.offset());
            out.writeInt(c.size());
            for (WeightedPoint p : c.points()) {
                out.writeDouble(p.weight());
                out.writeInt(p.vector().dim());
                out.writeInt(p.vector().nnz());
                for (int t = 0; t < p.vector().nnz(); t++) {
                    out.writeInt(p.vector().indexAt(t));
                    out.writeDouble(p.vector().valueAt(t));
                }
                if (buf.size() > (1 << 16)) {
                    md.update(buf.toByteArray());
                    buf.reset();
                }
            }
            out.flush();
            md.update(buf.toByteArray());
            return HexFormat.of().formatHex(md.digest());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required of every JVM", e);
        }
    }

    @Override
    public void close() {
        evaluationPool.shutdownNow();
    }
}
