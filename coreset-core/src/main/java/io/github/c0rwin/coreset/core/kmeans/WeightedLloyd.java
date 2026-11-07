package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Guards;
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
 * than m * dim. Both layouts give bit-identical results. Per call: one pass to read the points, then per iteration one
 * assignment pass and one update pass (two with sparse centers); allocation O(n + stored center entries).
 */
public final class WeightedLloyd {

    private WeightedLloyd() {
    }

    public static KMeansResult cluster(List<WeightedPoint> points, int k, int maxIterations, RandomGenerator rng) {
        return cluster(points, k, maxIterations, rng, MetricsSink.NOOP);
    }

    /**
     * Seeds with {@link KMeansPlusPlus} and runs Lloyd. Lloyd itself performs {@code n * c * (iterations + 1)} distance
     * evaluations for {@code n} points and {@code c} seeds. If no point has positive weight the result has no centers,
     * every assignment is -1 and the cost is 0.
     *
     * @throws IllegalArgumentException if {@code k < 1}, {@code maxIterations < 0} or points differ in dimension
     */
    public static KMeansResult cluster(List<WeightedPoint> points, int k, int maxIterations, RandomGenerator rng,
                                       MetricsSink metrics) {
        if (maxIterations < 0) {
            throw new IllegalArgumentException("maxIterations must be >= 0: " + maxIterations);
        }
        try (Span ignored = metrics.span(MetricNames.LLOYD_CLUSTER)) {
            List<SparseVector> seeds = KMeansPlusPlus.seed(points, k, rng, metrics);
            return run(points, seeds, maxIterations, metrics);
        }
    }

    /**
     * Runs Lloyd from the given initial centers (which need not be data points). Same contract as
     * {@link #cluster(List, int, int, RandomGenerator, MetricsSink)} minus seeding.
     *
     * @throws IllegalArgumentException if {@code initialCenters} is empty, {@code maxIterations < 0} or dimensions differ
     */
    public static KMeansResult refine(List<WeightedPoint> points, List<SparseVector> initialCenters, int maxIterations,
                                      MetricsSink metrics) {
        return refine(points, initialCenters, maxIterations, metrics, null);
    }

    /**
     * {@link #refine(List, List, int, MetricsSink)} that also writes every point's squared distance to its assigned
     * final center into {@code squaredDistances} (when not null), so callers need not recompute them.
     *
     * @throws IllegalArgumentException as the four-argument form, or if {@code squaredDistances.length != n}
     */
    public static KMeansResult refine(List<WeightedPoint> points, List<SparseVector> initialCenters, int maxIterations,
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

    /**
     * Runs {@link #cluster(List, int, int, RandomGenerator, MetricsSink)} {@code restarts} times and returns the
     * cheapest result (ties: earliest restart). Restart {@code i} uses {@code new SplittableRandom(s_i)}, where
     * {@code s_i} is the {@code i}-th {@code rng.nextLong()}, so one restart can be replayed on its own. Lloyd only
     * finds a local optimum; independent k-means++ restarts are the standard guard against a bad one.
     *
     * @throws IllegalArgumentException if {@code restarts < 1}, or as {@code cluster}
     */
    public static KMeansResult bestOf(List<WeightedPoint> points, int k, int maxIterations, int restarts,
                                      RandomGenerator rng, MetricsSink metrics) {
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

    private static KMeansResult run(List<WeightedPoint> points, List<SparseVector> initial, int maxIterations,
                                    MetricsSink metrics) {
        return run(points, initial, maxIterations, metrics, null, null);
    }

    /**
     * {@code layout == null} lets {@link Centroids#layoutFor} choose (tests force one to compare the two);
     * {@code dist}, if not null, receives the final squared distances and serves as the working array.
     */
    static KMeansResult run(List<WeightedPoint> points, List<SparseVector> initial, int maxIterations,
                            MetricsSink metrics, Centroids.Layout layout, double[] dist) {
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
        final SparseVector[] vectors = new SparseVector[n];
        final double[] weights = new double[n];
        long totalNnz = 0;
        double totalWeight = 0.0;
        double moment = 0.0;
        for (int i = 0; i < n; i++) {
            WeightedPoint p = points.get(i);
            if (p.vector().dim() != dim) {
                throw new IllegalArgumentException("point " + i + " has dim " + p.vector().dim() + ", expected " + dim);
            }
            vectors[i] = p.vector();
            weights[i] = p.weight();
            totalNnz += p.vector().nnz();
            totalWeight += p.weight();
            moment += p.weight() * p.vector().squaredNorm();
        }
        Guards.requireSummable(totalWeight, moment);
        Centroids centers = Centroids.create(layout != null ? layout : Centroids.layoutFor(m, dim, totalNnz),
                initial, dim, vectors, weights, totalNnz);
        double[] clusterWeight = new double[m];
        int[] assignment = new int[n];
        Arrays.fill(assignment, -1);
        if (dist == null) {
            dist = new double[n];
        }

        centers.assign(vectors, weights, assignment, dist);
        metrics.record(MetricNames.LLOYD_ITERATION_COST, centers.cost);
        int iterations = 0;
        long reseeds = 0;
        while (iterations < maxIterations) {
            Guards.checkInterrupted("Lloyd iterations");
            centers.moveToMeans(vectors, weights, assignment, clusterWeight);
            reseeds += reseedEmpty(centers, vectors, weights, clusterWeight, dist);
            iterations++;
            boolean changed = centers.assign(vectors, weights, assignment, dist);
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
    private static int reseedEmpty(Centroids centers, SparseVector[] vectors, double[] weights,
                                   double[] clusterWeight, double[] dist) {
        int reseeds = 0;
        for (int j = 0; j < clusterWeight.length; j++) {
            if (clusterWeight[j] > 0.0) {
                continue;
            }
            int far = farthestUnused(weights, dist);
            if (far < 0) {
                continue;
            }
            centers.set(j, vectors[far]);
            dist[far] = 0.0; // a second empty cluster must pick a different point
            reseeds++;
        }
        return reseeds;
    }

    private static int farthestUnused(double[] weights, double[] dist) {
        int best = -1;
        double bestD = 0.0;
        for (int i = 0; i < dist.length; i++) {
            if (weights[i] > 0.0 && dist[i] > bestD) {
                bestD = dist[i];
                best = i;
            }
        }
        return best;
    }

    /**
     * Replaces every part of a partition by its weighted mean: the result holds, in part order, each part with positive
     * weight as (mean, part weight), and its offset is {@code sum_p w_p ||p - mean(part of p)||^2}. Zero-weight points
     * are ignored. The offset is a sum of non-negative terms, so it is exact up to rounding even far from the origin.
     * Two passes over the points plus one sort of each part's stored entries; no O(dim) buffer.
     *
     * @throws IllegalArgumentException if points differ in dimension, a positive-weight point is assigned outside
     *     {@code [0, parts)} or the input overflows ({@link Guards#requireSummable})
     */
    public static Coreset meansOf(List<WeightedPoint> points, int[] assignment, int parts) {
        final int n = points.size();
        if (assignment.length != n) {
            throw new IllegalArgumentException("assignment has " + assignment.length + " entries for " + n + " points");
        }
        if (n == 0) {
            return Coreset.of(List.of());
        }
        final int dim = points.get(0).vector().dim();
        final SparseVector[] vectors = new SparseVector[n];
        final double[] weights = new double[n];
        double totalWeight = 0.0;
        double moment = 0.0;
        for (int i = 0; i < n; i++) {
            WeightedPoint p = points.get(i);
            if (p.vector().dim() != dim) {
                throw new IllegalArgumentException("point " + i + " has dim " + p.vector().dim() + ", expected " + dim);
            }
            if (p.weight() > 0.0 && (assignment[i] < 0 || assignment[i] >= parts)) {
                throw new IllegalArgumentException("point " + i + " assigned to " + assignment[i] + ", parts " + parts);
            }
            vectors[i] = p.vector();
            weights[i] = p.weight();
            totalWeight += p.weight();
            moment += p.weight() * p.vector().squaredNorm();
        }
        Guards.requireSummable(totalWeight, moment);
        ClusterSums sums = new ClusterSums(vectors, weights);
        double[] partWeight = sums.group(assignment, parts);
        List<WeightedPoint> out = new ArrayList<>(parts);
        double phi = 0.0;
        for (int j = 0; j < parts; j++) {
            if (partWeight[j] > 0.0) {
                ClusterSums.Mean mean = sums.mean(j, true);
                out.add(new WeightedPoint(SparseVector.of(dim, mean.indices, mean.values), partWeight[j]));
                phi += mean.cost;
            }
        }
        return new Coreset(out, phi);
    }
}
