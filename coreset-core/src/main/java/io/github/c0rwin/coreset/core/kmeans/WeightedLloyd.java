package io.github.c0rwin.coreset.core.kmeans;

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
 * Weighted Lloyd iterations on sparse points with dense centroids.
 *
 * <p>Each iteration is an update step (centers become weighted centroids of their clusters) followed by an assignment
 * step; the loop stops when an assignment step changes nothing or after {@code maxIterations} updates. A cluster with
 * no positive weight is re-seeded at the positive-weight point farthest from its current center, so centers never
 * become NaN. Assignment distances use {@link SparseVector#squaredDistance(double[], double)}.
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
        if (initialCenters.isEmpty()) {
            throw new IllegalArgumentException("initialCenters must not be empty");
        }
        if (maxIterations < 0) {
            throw new IllegalArgumentException("maxIterations must be >= 0: " + maxIterations);
        }
        try (Span ignored = metrics.span(MetricNames.LLOYD_CLUSTER)) {
            return run(points, initialCenters, maxIterations, metrics);
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
        final int n = points.size();
        final int m = initial.size();
        if (m == 0) {
            int[] none = new int[n];
            Arrays.fill(none, -1);
            metrics.record(MetricNames.LLOYD_ITERATIONS, 0);
            return new KMeansResult(List.of(), none, 0.0, 0);
        }
        final int dim = initial.get(0).dim();
        final SparseVector[] vectors = new SparseVector[n];
        final double[] weights = new double[n];
        for (int i = 0; i < n; i++) {
            WeightedPoint p = points.get(i);
            if (p.vector().dim() != dim) {
                throw new IllegalArgumentException("point " + i + " has dim " + p.vector().dim() + ", expected " + dim);
            }
            vectors[i] = p.vector();
            weights[i] = p.weight();
        }

        double[][] centers = new double[m][];
        double[] norms = new double[m];
        for (int j = 0; j < m; j++) {
            SparseVector c = initial.get(j);
            if (c.dim() != dim) {
                throw new IllegalArgumentException("center " + j + " has dim " + c.dim() + ", expected " + dim);
            }
            centers[j] = c.toDense();
            norms[j] = c.squaredNorm();
        }
        double[][] spare = new double[m][dim];
        double[] clusterWeight = new double[m];
        int[] assignment = new int[n];
        Arrays.fill(assignment, -1);
        double[] dist = new double[n];

        Lloyd state = new Lloyd(vectors, weights, centers, norms, assignment, dist);
        state.assign();
        metrics.record(MetricNames.LLOYD_ITERATION_COST, state.cost);
        int iterations = 0;
        long reseeds = 0;
        while (iterations < maxIterations) {
            reseeds += update(vectors, weights, centers, spare, norms, clusterWeight, assignment, dist);
            iterations++;
            boolean changed = state.assign();
            metrics.record(MetricNames.LLOYD_ITERATION_COST, state.cost);
            if (!changed) {
                break;
            }
        }
        metrics.count(MetricNames.LLOYD_DISTANCE_EVALUATIONS, (long) n * m * (iterations + 1));
        metrics.count(MetricNames.LLOYD_EMPTY_CLUSTER_RESEEDS, reseeds);
        metrics.record(MetricNames.LLOYD_ITERATIONS, iterations);

        List<SparseVector> out = new ArrayList<>(m);
        for (double[] c : centers) {
            out.add(SparseVector.dense(c));
        }
        return new KMeansResult(out, assignment, state.cost, iterations);
    }

    /** Recomputes centers in place (swapping with {@code spare}); returns the number of empty-cluster reseeds. */
    private static int update(SparseVector[] vectors, double[] weights, double[][] centers, double[][] spare,
                              double[] norms, double[] clusterWeight, int[] assignment, double[] dist) {
        final int m = centers.length;
        for (int j = 0; j < m; j++) {
            Arrays.fill(spare[j], 0.0);
        }
        Arrays.fill(clusterWeight, 0.0);
        for (int i = 0; i < vectors.length; i++) {
            double w = weights[i];
            if (w > 0.0) {
                int j = assignment[i];
                vectors[i].addTo(spare[j], w);
                clusterWeight[j] += w;
            }
        }
        int reseeds = 0;
        for (int j = 0; j < m; j++) {
            double[] acc = spare[j];
            double cw = clusterWeight[j];
            if (cw > 0.0) {
                double inv = 1.0 / cw;
                double norm = 0.0;
                for (int d = 0; d < acc.length; d++) {
                    double v = acc[d] * inv;
                    acc[d] = v;
                    norm += v * v;
                }
                spare[j] = centers[j];
                centers[j] = acc;
                norms[j] = norm;
                continue;
            }
            int far = farthestUnused(weights, dist);
            if (far < 0) {
                continue; // every positive-weight point already sits on a center: keep this one
            }
            vectors[far].addTo(acc, 1.0);
            spare[j] = centers[j];
            centers[j] = acc;
            norms[j] = vectors[far].squaredNorm();
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

    /** Assignment step state; arrays are shared with {@link #run} and updated in place. */
    private static final class Lloyd {
        final SparseVector[] vectors;
        final double[] weights;
        final double[][] centers;
        final double[] norms;
        final int[] assignment;
        final double[] dist;
        double cost;

        Lloyd(SparseVector[] vectors, double[] weights, double[][] centers, double[] norms, int[] assignment,
              double[] dist) {
            this.vectors = vectors;
            this.weights = weights;
            this.centers = centers;
            this.norms = norms;
            this.assignment = assignment;
            this.dist = dist;
        }

        /** Assigns every point to its nearest center (ties: lowest index); returns whether any assignment changed. */
        boolean assign() {
            final int m = centers.length;
            boolean changed = false;
            double total = 0.0;
            for (int i = 0; i < vectors.length; i++) {
                SparseVector p = vectors[i];
                int best = 0;
                double bestD = p.squaredDistance(centers[0], norms[0]);
                for (int j = 1; j < m; j++) {
                    double d = p.squaredDistance(centers[j], norms[j]);
                    if (d < bestD) {
                        bestD = d;
                        best = j;
                    }
                }
                if (assignment[i] != best) {
                    assignment[i] = best;
                    changed = true;
                }
                dist[i] = bestD;
                total += weights[i] * bestD;
            }
            cost = total;
            return changed;
        }
    }
}
