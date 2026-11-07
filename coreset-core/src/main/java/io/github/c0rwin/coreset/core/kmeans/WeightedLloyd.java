package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Weighted Lloyd iterations on sparse points.
 *
 * <p>Each iteration is an update step (centers become weighted centroids of their clusters) followed by an assignment
 * step; the loop stops when an assignment step changes nothing or after {@code maxIterations} updates. A cluster with
 * no positive weight is re-seeded at the positive-weight point farthest from its current center, so centers never
 * become NaN. Distances follow {@link SparseVector#squaredDistance(double[], double)}, which stays accurate far from
 * the origin, so assignments and the reported cost do not depend on where the data sits.
 *
 * <p>Centers are dense m x dim arrays while {@code m * dim <= max(2^20, 2 * total stored entries)}, and otherwise
 * sparse (sorted coordinates plus a hash index), so memory and update time follow the input's stored entries rather
 * than m * dim. Both layouts give bit-identical results. The points are streamed from their {@link PointSet} segments;
 * per iteration one assignment pass and one update pass (two with sparse centers); allocation O(n + stored center
 * entries). The {@code List} overloads first copy the points into one block.
 */
public final class WeightedLloyd {

    private WeightedLloyd() {
    }

    public static KMeansResult cluster(List<WeightedPoint> points, int k, int maxIterations, RandomGenerator rng) {
        return cluster(points, k, maxIterations, rng, MetricsSink.NOOP);
    }

    public static KMeansResult cluster(List<WeightedPoint> points, int k, int maxIterations, RandomGenerator rng,
                                       MetricsSink metrics) {
        return cluster(PointSet.of(PointBlock.of(points)), k, maxIterations, rng, metrics);
    }

    /**
     * Seeds with {@link KMeansPlusPlus} and runs Lloyd. Lloyd itself performs {@code n * c * (iterations + 1)} distance
     * evaluations for {@code n} points and {@code c} seeds. If no point has positive weight the result has no centers,
     * every assignment is -1 and the cost is 0.
     *
     * @throws IllegalArgumentException if {@code k < 1} or {@code maxIterations < 0}
     */
    public static KMeansResult cluster(PointSet points, int k, int maxIterations, RandomGenerator rng,
                                       MetricsSink metrics) {
        if (maxIterations < 0) {
            throw new IllegalArgumentException("maxIterations must be >= 0: " + maxIterations);
        }
        try (Span ignored = metrics.span(MetricNames.LLOYD_CLUSTER)) {
            List<SparseVector> seeds = KMeansPlusPlus.seed(points, k, rng, metrics);
            return run(points, seeds, maxIterations, metrics, null, null);
        }
    }

    public static KMeansResult refine(List<WeightedPoint> points, List<SparseVector> initialCenters, int maxIterations,
                                      MetricsSink metrics) {
        return refine(PointSet.of(PointBlock.of(points)), initialCenters, maxIterations, metrics, null);
    }

    public static KMeansResult refine(List<WeightedPoint> points, List<SparseVector> initialCenters, int maxIterations,
                                      MetricsSink metrics, double[] squaredDistances) {
        return refine(PointSet.of(PointBlock.of(points)), initialCenters, maxIterations, metrics, squaredDistances);
    }

    /**
     * Runs Lloyd from the given initial centers (which need not be data points). Same contract as
     * {@link #cluster(PointSet, int, int, RandomGenerator, MetricsSink)} minus seeding.
     *
     * @throws IllegalArgumentException if {@code initialCenters} is empty, {@code maxIterations < 0} or dimensions differ
     */
    public static KMeansResult refine(PointSet points, List<SparseVector> initialCenters, int maxIterations,
                                      MetricsSink metrics) {
        return refine(points, initialCenters, maxIterations, metrics, null);
    }

    /**
     * {@link #refine(PointSet, List, int, MetricsSink)} that also writes every point's squared distance to its
     * assigned final center into {@code squaredDistances} (when not null), so callers need not recompute them.
     *
     * @throws IllegalArgumentException as the four-argument form, or if {@code squaredDistances.length != n}
     */
    public static KMeansResult refine(PointSet points, List<SparseVector> initialCenters, int maxIterations,
                                      MetricsSink metrics, double[] squaredDistances) {
        if (initialCenters.isEmpty()) {
            throw new IllegalArgumentException("initialCenters must not be empty");
        }
        if (maxIterations < 0) {
            throw new IllegalArgumentException("maxIterations must be >= 0: " + maxIterations);
        }
        if (squaredDistances != null && squaredDistances.length != points.size()) {
            throw new IllegalArgumentException("squaredDistances has " + squaredDistances.length + " entries for "
                    + points.size() + " points");
        }
        try (Span ignored = metrics.span(MetricNames.LLOYD_CLUSTER)) {
            return run(points, initialCenters, maxIterations, metrics, null, squaredDistances);
        }
    }

    public static KMeansResult bestOf(List<WeightedPoint> points, int k, int maxIterations, int restarts,
                                      RandomGenerator rng, MetricsSink metrics) {
        return bestOf(PointSet.of(PointBlock.of(points)), k, maxIterations, restarts, rng, metrics);
    }

    /**
     * Runs {@link #cluster(PointSet, int, int, RandomGenerator, MetricsSink)} {@code restarts} times and returns the
     * cheapest result (ties: earliest restart). Restart {@code i} uses {@code new SplittableRandom(s_i)}, where
     * {@code s_i} is the {@code i}-th {@code rng.nextLong()}, so one restart can be replayed on its own. Lloyd only
     * finds a local optimum; independent k-means++ restarts are the standard guard against a bad one.
     *
     * @throws IllegalArgumentException if {@code restarts < 1}, or as {@code cluster}
     */
    public static KMeansResult bestOf(PointSet points, int k, int maxIterations, int restarts, RandomGenerator rng,
                                      MetricsSink metrics) {
        if (restarts < 1) {
            throw new IllegalArgumentException("restarts must be >= 1: " + restarts);
        }
        long[] seeds = new long[restarts];
        for (int i = 0; i < restarts; i++) {
            seeds[i] = rng.nextLong();
        }
        try (Span ignored = metrics.span(MetricNames.LLOYD_BEST_OF)) {
            KMeansResult best = null;
            for (long seed : seeds) {
                KMeansResult r = cluster(points, k, maxIterations, new SplittableRandom(seed), metrics);
                metrics.record(MetricNames.LLOYD_RESTART_COST, r.cost());
                if (best == null || r.cost() < best.cost()) {
                    best = r;
                }
            }
            return best;
        }
    }

    /**
     * {@code layout == null} lets {@link Centroids#layoutFor} choose (tests force one to compare the two);
     * {@code dist}, if not null, receives the final squared distances and serves as the working array.
     */
    static KMeansResult run(PointSet points, List<SparseVector> initial, int maxIterations, MetricsSink metrics,
                            Centroids.Layout layout, double[] dist) {
        final int n = points.size();
        final int m = initial.size();
        if (m == 0) {
            int[] none = new int[n];
            Arrays.fill(none, -1);
            metrics.record(MetricNames.LLOYD_ITERATIONS, 0);
            return new KMeansResult(List.of(), none, 0.0, 0);
        }
        final int dim = initial.get(0).dim();
        for (int j = 1; j < m; j++) {
            if (initial.get(j).dim() != dim) {
                throw new IllegalArgumentException("center " + j + " has dim " + initial.get(j).dim() + ", expected "
                        + dim);
            }
        }
        if (n > 0 && points.dim() != dim) {
            throw new IllegalArgumentException("points have dim " + points.dim() + ", expected " + dim);
        }
        double totalWeight = 0.0;
        double moment = 0.0;
        for (int t = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0, rows = b.size(); i < rows; i++) {
                double w = b.weight(i);
                totalWeight += w;
                moment += w * b.squaredNorm(i);
            }
        }
        Guards.requireSummable(totalWeight, moment);
        final long totalNnz = points.storedEntries();
        Centroids centers = Centroids.create(layout != null ? layout : Centroids.layoutFor(m, dim, totalNnz),
                initial, dim, points);
        double[] clusterWeight = new double[m];
        int[] assignment = new int[n];
        Arrays.fill(assignment, -1);
        if (dist == null) {
            dist = new double[n];
        }

        centers.assign(points, assignment, dist);
        metrics.record(MetricNames.LLOYD_ITERATION_COST, centers.cost);
        int iterations = 0;
        long reseeds = 0;
        while (iterations < maxIterations) {
            Guards.checkInterrupted("Lloyd iterations");
            centers.moveToMeans(points, assignment, clusterWeight);
            reseeds += reseedEmpty(centers, points, clusterWeight, dist);
            iterations++;
            boolean changed = centers.assign(points, assignment, dist);
            metrics.record(MetricNames.LLOYD_ITERATION_COST, centers.cost);
            if (!changed) {
                break;
            }
        }
        metrics.count(MetricNames.LLOYD_DISTANCE_EVALUATIONS, (long) n * m * (iterations + 1));
        metrics.count(MetricNames.LLOYD_EMPTY_CLUSTER_RESEEDS, reseeds);
        metrics.count(MetricNames.LLOYD_POINTS_SCANNED, (long) n * (1 + (iterations + 1L) * centers.assignPasses()
                + (long) iterations * centers.updatePasses()));
        if (centers instanceof Centroids.Sparse) {
            metrics.count(MetricNames.LLOYD_SPARSE_CENTERS, 1);
        }
        metrics.record(MetricNames.LLOYD_ITERATIONS, iterations);

        List<SparseVector> out = new ArrayList<>(m);
        for (int j = 0; j < m; j++) {
            out.add(centers.center(j));
        }
        return new KMeansResult(out, assignment, centers.cost, iterations);
    }

    /**
     * Moves each center whose cluster has no positive weight to the farthest positive-weight point not yet used for
     * this, or leaves it if every such point already sits on a center; returns the number of moved centers.
     */
    private static int reseedEmpty(Centroids centers, PointSet points, double[] clusterWeight, double[] dist) {
        int reseeds = 0;
        for (int j = 0; j < clusterWeight.length; j++) {
            if (clusterWeight[j] > 0.0) {
                continue;
            }
            int far = farthestUnused(points, dist);
            if (far < 0) {
                continue;
            }
            centers.set(j, points.vector(far));
            dist[far] = 0.0; // a second empty cluster must pick a different point
            reseeds++;
        }
        return reseeds;
    }

    private static int farthestUnused(PointSet points, double[] dist) {
        int best = -1;
        double bestD = 0.0;
        for (int t = 0, g = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                if (b.weight(i) > 0.0 && dist[g] > bestD) {
                    bestD = dist[g];
                    best = g;
                }
            }
        }
        return best;
    }

    public static Coreset meansOf(List<WeightedPoint> points, int[] assignment, int parts) {
        return meansOf(PointSet.of(PointBlock.of(points)), assignment, parts);
    }

    /**
     * Replaces every part of a partition by its weighted mean: the result holds, in part order, each part with positive
     * weight as (mean, part weight), and its offset is {@code sum_p w_p ||p - mean(part of p)||^2}. Zero-weight points
     * are ignored. The offset is a sum of non-negative terms, so it is exact up to rounding even far from the origin.
     * Two passes over the points plus one sort of each part's stored entries; no O(dim) buffer. The means are written
     * straight into the result's block.
     *
     * @throws IllegalArgumentException if a positive-weight point is assigned outside {@code [0, parts)} or the input
     *     overflows ({@link Guards#requireSummable})
     */
    public static Coreset meansOf(PointSet points, int[] assignment, int parts) {
        final int n = points.size();
        if (assignment.length != n) {
            throw new IllegalArgumentException("assignment has " + assignment.length + " entries for " + n + " points");
        }
        if (n == 0) {
            return Coreset.of(PointBlock.EMPTY);
        }
        double totalWeight = 0.0;
        double moment = 0.0;
        for (int t = 0, g = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                double w = b.weight(i);
                if (w > 0.0 && (assignment[g] < 0 || assignment[g] >= parts)) {
                    throw new IllegalArgumentException("point " + g + " assigned to " + assignment[g] + ", parts "
                            + parts);
                }
                totalWeight += w;
                moment += w * b.squaredNorm(i);
            }
        }
        Guards.requireSummable(totalWeight, moment);
        ClusterSums sums = new ClusterSums(points);
        double[] partWeight = sums.group(assignment, parts);
        PointBlock.Builder out = PointBlock.builder(points.dim(), parts, 0);
        double phi = 0.0;
        for (int j = 0; j < parts; j++) {
            if (partWeight[j] > 0.0) {
                ClusterSums.Mean mean = sums.mean(j, true);
                out.addRow(mean.indices, mean.values, 0, mean.indices.length, partWeight[j]);
                phi += mean.cost;
            }
        }
        return new Coreset(out.build(), phi);
    }
}
