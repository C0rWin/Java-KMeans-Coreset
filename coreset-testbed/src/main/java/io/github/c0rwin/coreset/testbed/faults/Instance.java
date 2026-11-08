package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.algo.Clustering;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityBound;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.testbed.data.Dataset;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;

/**
 * A dataset prepared once for every fault experiment on it: the ESA'22 candidate pool plus the best-of-N Lloyd
 * solution on the full data (the downstream reference, also used to label clusters), the per-row cost table, the
 * point-level cost concentration {@code kappa_pt(Q)} per candidate and the k-means++ sensitivity bounds.
 */
final class Instance {

    static final String REFERENCE = "bestOf@P";

    final String id;
    final Dataset dataset;
    final PointSet points;
    final int n;
    final int k;
    final List<String> labels;
    final List<Centers> candidates;
    final int ref;
    final CostTable table;
    final double[] costFull;
    final double[] kappa;
    final double[] sens;
    final double sensTotal;
    final double sMax;
    final int[] cluster;
    final int[] clusterSize;
    final int smallestCluster;
    final double prepareMs;

    private Instance(String id, Dataset dataset, int k, List<String> labels, List<Centers> candidates,
                     CostTable table, double[] sens, double sensTotal, int[] cluster, double prepareMs) {
        this.id = id;
        this.dataset = dataset;
        this.points = dataset.coreset().pointSet();
        this.n = points.size();
        this.k = k;
        this.labels = List.copyOf(labels);
        this.candidates = List.copyOf(candidates);
        this.ref = candidates.size() - 1;
        this.table = table;
        this.costFull = table.weightedSums(points);
        this.kappa = new double[table.cols];
        for (int r = 0; r < n; r++) {
            int o = r * table.cols;
            double w = points.weight(r);
            for (int j = 0; j < table.cols; j++) {
                double c = w * table.cost[o + j];
                kappa[j] += c * c;
            }
        }
        for (int j = 0; j < table.cols; j++) {
            kappa[j] /= costFull[j] * costFull[j];
        }
        this.sens = sens;
        this.sensTotal = sensTotal;
        double max = 0.0;
        for (double s : sens) {
            max = Math.max(max, s);
        }
        this.sMax = max / sensTotal;
        this.cluster = cluster;
        int maxLabel = 0;
        for (int c : cluster) {
            maxLabel = Math.max(maxLabel, c);
        }
        this.clusterSize = new int[maxLabel + 1];
        for (int c : cluster) {
            clusterSize[c]++;
        }
        int small = 0;
        for (int c = 1; c < clusterSize.length; c++) {
            if (clusterSize[c] > 0 && (clusterSize[small] == 0 || clusterSize[c] < clusterSize[small])) {
                small = c;
            }
        }
        this.smallestCluster = small;
        this.prepareMs = prepareMs;
    }

    static Instance prepare(String id, Dataset ds, int k, int perGenerator, int lloydIterations, int restarts,
                            long seed, ForkJoinPool pool) {
        long t0 = System.nanoTime();
        SplittableRandom rng = new SplittableRandom(seed);
        List<Candidate> cands = new ArrayList<>();
        cands.addAll(CandidateSolutions.kMeansPlusPlusOnFull(ds.coreset(), k, perGenerator, rng));
        cands.addAll(CandidateSolutions.convexCombinationsOfBicriteria(ds.coreset(), k, perGenerator, rng));
        PointSet pts = ds.coreset().pointSet();
        KMeansResult best = WeightedLloyd.bestOf(pts, k, lloydIterations, restarts, rng, MetricsSink.NOOP);
        cands.add(new Candidate(REFERENCE, best.centers()));
        List<String> labels = new ArrayList<>();
        List<Centers> centers = new ArrayList<>();
        for (Candidate c : cands) {
            labels.add(c.generator());
            centers.add(Centers.of(c.centers()));
        }
        CostTable table = CostTable.of(pts, centers, pool);
        Clustering b = Seeding.kMeansPlusPlus(0).cluster(pts, k, rng, MetricsSink.NOOP);
        double[] s = SensitivityBound.standard(SensitivityBound.DEFAULT_ALPHA).sensitivities(pts, b);
        double total = 0.0;
        for (double x : s) {
            total += x;
        }
        return new Instance(id, ds, k, labels, centers, table, s, total, best.assignment(),
                (System.nanoTime() - t0) / 1e6);
    }

    int candidates() {
        return candidates.size();
    }

    int dim() {
        return points.dim();
    }

    List<SparseVector> referenceCenters() {
        return candidates.get(ref).vectors();
    }

    double referenceCost() {
        return costFull[ref];
    }

    /** {@code r_j = cost_j / cost(P, Q_j) - 1}. */
    double[] ratios(double[] cost, double scale) {
        double[] r = new double[cost.length];
        for (int j = 0; j < cost.length; j++) {
            r[j] = scale * cost[j] / costFull[j] - 1.0;
        }
        return r;
    }
}
