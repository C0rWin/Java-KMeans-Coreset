package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Weighted k-means++ (D^2) seeding: the first center is drawn with probability proportional to weight, each next one
 * proportional to {@code w * D(p)^2}, where {@code D(p)} is the distance to the closest chosen center.
 */
public final class KMeansPlusPlus {

    private KMeansPlusPlus() {
    }

    public static List<SparseVector> seed(List<WeightedPoint> points, int k, RandomGenerator rng) {
        return seed(points, k, rng, MetricsSink.NOOP);
    }

    /**
     * Returns {@code min(k, number of distinct points with positive weight)} distinct centers, chosen from the
     * input. Zero-weight points are never chosen. Performs {@code n * (c - 1)} distance evaluations when {@code c = k}
     * centers are returned and {@code n * c} when the input runs out of distinct points first, where {@code n} is the
     * number of positive-weight points. On sparse input each new center is scattered into one O(dim) buffer, so a
     * distance costs O(nnz(point)) ({@link SparseVector#scatterPays}, {@link SparseVector#squaredDistanceScattered}).
     *
     * @throws IllegalArgumentException if {@code k < 1} or the input overflows ({@link Guards#requireSummable})
     */
    public static List<SparseVector> seed(List<WeightedPoint> points, int k, RandomGenerator rng,
                                          MetricsSink metrics) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        try (Span ignored = metrics.span(MetricNames.KMEANSPP_SEED)) {
            int n = 0;
            double totalWeight = 0.0;
            double moment = 0.0;
            long entries = 0;
            for (int i = 0, size = points.size(); i < size; i++) {
                WeightedPoint p = points.get(i);
                moment += p.weight() * p.vector().squaredNorm();
                entries += p.vector().nnz();
                if (p.weight() > 0.0) {
                    n++;
                    totalWeight += p.weight();
                }
            }
            Guards.requireSummable(totalWeight, moment);
            List<SparseVector> centers = new ArrayList<>(Math.min(k, n));
            if (n == 0) {
                return centers;
            }
            SparseVector[] vectors = new SparseVector[n];
            double[] weights = new double[n];
            for (int i = 0, w = 0, size = points.size(); i < size; i++) {
                WeightedPoint p = points.get(i);
                if (p.weight() > 0.0) {
                    vectors[w] = p.vector();
                    weights[w] = p.weight();
                    w++;
                }
            }

            SparseVector last = vectors[sample(weights, totalWeight, rng)];
            centers.add(last);

            double[] minD2 = new double[n];
            Arrays.fill(minD2, Double.POSITIVE_INFINITY);
            double[] score = new double[n];
            // The latest center scattered into one O(dim) buffer: O(nnz(point)) per distance on sparse input.
            double[] scratch = k > 1 && SparseVector.scatterPays(last.dim(), points.size(), entries)
                    ? new double[last.dim()] : null;
            long evaluations = 0;
            while (centers.size() < k) {
                Guards.checkInterrupted("k-means++ seeding");
                if (scratch != null) {
                    last.scatterInto(scratch);
                }
                double total = 0.0;
                for (int i = 0; i < n; i++) {
                    double d = scratch != null ? vectors[i].squaredDistanceScattered(scratch, last)
                            : vectors[i].squaredDistance(last);
                    if (d < minD2[i]) {
                        minD2[i] = d;
                    }
                    score[i] = weights[i] * minD2[i];
                    total += score[i];
                }
                if (scratch != null) {
                    last.clearFrom(scratch);
                }
                evaluations += n;
                if (total == 0.0) {
                    break; // every positive-weight point coincides with a chosen center
                }
                last = vectors[sample(score, total, rng)];
                centers.add(last);
            }
            metrics.count(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS, evaluations);
            return centers;
        }
    }

    /** Draws an index with probability proportional to {@code mass[i]}; only indices with positive mass qualify. */
    private static int sample(double[] mass, double total, RandomGenerator rng) {
        if (!Double.isFinite(total)) {
            return argMax(mass); // overflowed masses cannot be sampled proportionally
        }
        double r = rng.nextDouble() * total;
        double cumulative = 0.0;
        int lastPositive = -1;
        for (int i = 0; i < mass.length; i++) {
            double m = mass[i];
            if (m > 0.0) {
                cumulative += m;
                lastPositive = i;
                if (r < cumulative) {
                    return i;
                }
            }
        }
        return lastPositive; // rounding left r at the very top of the range
    }

    private static int argMax(double[] mass) {
        int best = 0;
        for (int i = 1; i < mass.length; i++) {
            if (mass[i] > mass[best]) {
                best = i;
            }
        }
        return best;
    }
}
