package io.github.c0rwin.coreset.experiments.quality;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.QualityConfig;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Quality of a run's output, computed in the orchestrator after the cell JVM exited (so it neither competes with the
 * system for CPU and memory nor depends on it): exact full-data cost of the deployed centers, ratio to a reference
 * solution, weight error and empirical distortion of the coreset.
 *
 * <p>Distortion follows {@link Distortion} exactly (entries, ratio {@code cost(C,Q)/cost(P,Q)}, {@code max(r, 1/r)}),
 * but full-data costs of the candidates shared by all coresets (k-means++ and Lloyd on P, the reference) are computed
 * once per dataset in one parallel pass instead of once per coreset; candidates specific to a coreset (Lloyd on C and
 * the deployed centers) share a second pass.
 */
public final class QualityEvaluator {

    private static final ObjectMapper JSON = new ObjectMapper();

    private final PointSource src;
    private final int k;
    private final QualityConfig q;
    private final long seed;
    private final Coreset full;
    private final Reference reference;
    private final List<Candidate> shared = new ArrayList<>();
    private final double[] sharedFullCosts;
    private final double totalWeight;

    /** Reference solution: centers, exact full cost, how it was obtained. */
    public record Reference(List<SparseVector> centers, double cost, Map<String, Object> info) {
    }

    public QualityEvaluator(PointSource src, int k, QualityConfig q, Path cacheDir, long seed) throws IOException {
        this.src = src;
        this.k = k;
        this.q = q;
        this.seed = seed;
        this.full = InMemory.load(src, q.maxInMemoryPoints(), seed);
        this.reference = loadOrComputeReference(cacheDir);
        if (q.distortion()) {
            SplittableRandom rnd = new SplittableRandom(seed ^ 0x5DEECE66DL);
            shared.addAll(CandidateSolutions.kMeansPlusPlusOnFull(full, k, q.candidates(), rnd));
            shared.addAll(CandidateSolutions.lloydOnFull(full, k, q.candidateIterations(), q.candidates(), rnd));
            shared.add(new Candidate("reference", reference.centers()));
        }
        List<List<SparseVector>> sets = new ArrayList<>();
        shared.forEach(c -> sets.add(c.centers()));
        if (sets.isEmpty()) { // still one pass, for the total weight
            sets.add(reference.centers());
        }
        FullCost.Result r = FullCost.evaluate(src, sets, q.evalThreads());
        this.sharedFullCosts = r.costs();
        this.totalWeight = r.weight();
    }

    public Reference reference() {
        return reference;
    }

    public double totalWeight() {
        return totalWeight;
    }

    /** Quality fields of one run: {@code coreset} is null for native (coreset-free) baselines. */
    public Map<String, Object> evaluate(Coreset coreset, List<SparseVector> centers) {
        Map<String, Object> out = new LinkedHashMap<>();
        long t0 = System.nanoTime();
        double costFull = FullCost.evaluate(src, List.of(centers), q.evalThreads()).costs()[0];
        out.put("eval_s", (System.nanoTime() - t0) / 1e9);
        out.put("eval_threads", q.evalThreads());
        out.put("cost_full", costFull);
        out.put("reference_cost", reference.cost());
        out.put("cost_ratio", costFull / reference.cost());
        if (coreset == null) {
            return out;
        }
        out.put("coreset_weight", coreset.totalWeight());
        out.put("weight_error", Math.abs(coreset.totalWeight() - totalWeight) / totalWeight);
        double deployedCoresetCost = Cost.cost(coreset, centers);
        out.put("cost_coreset", deployedCoresetCost);
        out.put("deployed_distortion", distortion(deployedCoresetCost, costFull));
        if (!q.distortion()) {
            return out;
        }
        List<Candidate> own = new ArrayList<>(CandidateSolutions.lloydOnCoreset(coreset, k, q.candidateIterations(),
                q.candidates(), new SplittableRandom(seed ^ 0x9E3779B97F4A7C15L)));
        List<List<SparseVector>> sets = new ArrayList<>();
        own.forEach(c -> sets.add(c.centers()));
        double[] ownFull = FullCost.evaluate(src, sets, q.evalThreads()).costs();
        own.add(new Candidate("deployed", centers));
        List<Distortion.Entry> entries = new ArrayList<>();
        for (int i = 0; i < shared.size(); i++) {
            entries.add(entry(shared.get(i), sharedFullCosts[i], coreset));
        }
        for (int i = 0; i < own.size(); i++) {
            entries.add(entry(own.get(i), i < ownFull.length ? ownFull[i] : costFull, coreset));
        }
        Distortion.Report report = report(entries);
        out.put("distortion_max", report.max());
        out.put("distortion_mean", report.mean());
        out.put("distortion_argmax", report.argmaxGenerator());
        out.put("distortion_candidates", entries.size());
        out.put("distortion_sampled_candidates", src.size() > q.maxInMemoryPoints());
        return out;
    }

    private static Distortion.Entry entry(Candidate c, double cp, Coreset coreset) {
        double cc = Cost.cost(coreset, c.centers());
        double ratio = cp > 0.0 ? cc / cp : (cc == 0.0 ? 1.0 : Double.POSITIVE_INFINITY);
        return new Distortion.Entry(c.generator(), cp, cc, ratio, ratio >= 1.0 ? ratio : 1.0 / ratio);
    }

    /** {@link Distortion#evaluate}'s aggregation over precomputed entries. */
    static Distortion.Report report(List<Distortion.Entry> entries) {
        double max = 0.0;
        double sum = 0.0;
        String argmax = null;
        for (Distortion.Entry e : entries) {
            sum += e.distortion();
            if (argmax == null || e.distortion() > max) {
                max = e.distortion();
                argmax = e.generator();
            }
        }
        return new Distortion.Report(entries, max, sum / entries.size(), argmax);
    }

    private static double distortion(double coresetCost, double fullCost) {
        double r = fullCost > 0.0 ? coresetCost / fullCost : (coresetCost == 0.0 ? 1.0 : Double.POSITIVE_INFINITY);
        return r >= 1.0 ? r : 1.0 / r;
    }

    /**
     * Cheapest of {@code referenceRuns} best-of-{@code referenceRestarts} Lloyd solutions on the full data (or its
     * sample), by exact full cost; cached per dataset checksum, k and settings.
     */
    private Reference loadOrComputeReference(Path cacheDir) throws IOException {
        Object sha = src.provenance().getOrDefault("sha256", "");
        String key = src.id() + "-" + String.valueOf(sha).substring(0, Math.min(12, String.valueOf(sha).length()))
                + "-k" + k + "-r" + q.referenceRuns() + "x" + q.referenceRestarts() + "-i" + q.referenceIterations()
                + (src.size() > q.maxInMemoryPoints() ? "-s" + q.maxInMemoryPoints() : "") + ".json";
        Path file = cacheDir.resolve("reference").resolve(key);
        if (Files.exists(file)) {
            Map<?, ?> m = JSON.readValue(file.toFile(), Map.class);
            return new Reference(decode(m.get("centers"), src.dim()), ((Number) m.get("cost")).doubleValue(),
                    castMap(m.get("info")));
        }
        long t0 = System.nanoTime();
        List<List<SparseVector>> runs = new ArrayList<>();
        SplittableRandom rnd = new SplittableRandom(seed ^ 0xC0FFEEL);
        for (int r = 0; r < q.referenceRuns(); r++) {
            KMeansResult res = WeightedLloyd.bestOf(full.pointSet(), k, q.referenceIterations(),
                    q.referenceRestarts(), rnd.split(), MetricsSink.NOOP);
            runs.add(res.centers());
        }
        double[] costs = FullCost.evaluate(src, runs, q.evalThreads()).costs();
        int best = 0;
        for (int r = 1; r < costs.length; r++) {
            if (costs[r] < costs[best]) {
                best = r;
            }
        }
        Map<String, Object> info = new LinkedHashMap<>();
        info.put("runs", q.referenceRuns());
        info.put("restarts", q.referenceRestarts());
        info.put("iterations", q.referenceIterations());
        info.put("runCosts", java.util.Arrays.stream(costs).boxed().toList());
        info.put("onSample", src.size() > q.maxInMemoryPoints());
        info.put("seconds", (System.nanoTime() - t0) / 1e9);
        Reference ref = new Reference(runs.get(best), costs[best], info);
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("dataset", src.provenance());
        m.put("k", k);
        m.put("cost", ref.cost());
        m.put("info", info);
        m.put("centers", encode(ref.centers()));
        Files.createDirectories(file.getParent());
        Path tmp = Files.createTempFile(file.getParent(), key, ".tmp");
        JSON.writerWithDefaultPrettyPrinter().writeValue(tmp.toFile(), m);
        Files.move(tmp, file, java.nio.file.StandardCopyOption.ATOMIC_MOVE);
        return ref;
    }

    /** Centers as {@code [[index, value, index, value, ...], ...]}. */
    public static List<double[]> encode(List<SparseVector> centers) {
        List<double[]> out = new ArrayList<>(centers.size());
        for (SparseVector c : centers) {
            double[] row = new double[2 * c.nnz()];
            for (int e = 0; e < c.nnz(); e++) {
                row[2 * e] = c.indexAt(e);
                row[2 * e + 1] = c.valueAt(e);
            }
            out.add(row);
        }
        return out;
    }

    public static List<SparseVector> decode(Object centers, int dim) {
        List<SparseVector> out = new ArrayList<>();
        for (Object o : (List<?>) centers) {
            List<?> row = (List<?>) o;
            int nnz = row.size() / 2;
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int e = 0; e < nnz; e++) {
                idx[e] = ((Number) row.get(2 * e)).intValue();
                val[e] = ((Number) row.get(2 * e + 1)).doubleValue();
            }
            out.add(SparseVector.of(dim, idx, val));
        }
        return out;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Object o) {
        return o == null ? Map.of() : (Map<String, Object>) o;
    }
}
