package io.github.c0rwin.coreset.core.kmeans;

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

    /** {@link #seed(PointSet, int, RandomGenerator, MetricsSink)} on a copy of {@code points} in one block. */
    public static List<SparseVector> seed(List<WeightedPoint> points, int k, RandomGenerator rng,
                                          MetricsSink metrics) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        return seed(PointSet.of(PointBlock.of(points)), k, rng, metrics);
    }

    /**
     * Returns {@code min(k, number of distinct points with positive weight)} distinct centers, chosen from the
     * input (as views of its rows). Zero-weight points are never chosen. Performs {@code n * (c - 1)} distance
     * evaluations when {@code c = k} centers are returned and {@code n * c} when the input runs out of distinct points
     * first, where {@code n} is the number of positive-weight points. On sparse input each new center is scattered
     * into one O(dim) buffer, so a distance costs O(nnz(point)) ({@link SparseVector#scatterPays},
     * {@link SparseVector#squaredDistanceScattered}). The segments are streamed in order, so the result does not
     * depend on how the rows are segmented.
     *
     * @throws IllegalArgumentException if {@code k < 1} or the input overflows ({@link Guards#requireSummable})
     */
    public static List<SparseVector> seed(PointSet points, int k, RandomGenerator rng, MetricsSink metrics) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        try (Span ignored = metrics.span(MetricNames.KMEANSPP_SEED)) {
            final int size = points.size();
            final int segments = points.segmentCount();
            int n = 0;
            double totalWeight = 0.0;
            double moment = 0.0;
            for (int t = 0; t < segments; t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++) {
                    double w = b.weight(i);
                    moment += w * b.squaredNorm(i);
                    if (w > 0.0) {
                        n++;
                        totalWeight += w;
                    }
                }
            }
            Guards.requireSummable(totalWeight, moment);
            List<SparseVector> centers = new ArrayList<>(Math.min(k, n));
            if (n == 0) {
                return centers;
            }
            int[] rows = new int[n]; // positive-weight point -> global row, only to locate drawn centers
            double[] weights = new double[n];
            for (int t = 0, w = 0; t < segments; t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, r = b.size(); i < r; i++) {
                    if (b.weight(i) > 0.0) {
                        rows[w] = points.start(t) + i;
                        weights[w] = b.weight(i);
                        w++;
                    }
                }
            }

            int last = rows[sample(weights, totalWeight, rng)];
            centers.add(points.vector(last));

            double[] minD2 = new double[n];
            Arrays.fill(minD2, Double.POSITIVE_INFINITY);
            double[] score = new double[n];
            // The latest center scattered into one O(dim) buffer: O(nnz(point)) per distance on sparse input.
            double[] scratch = k > 1 && SparseVector.scatterPays(points.dim(), size, points.storedEntries())
                    ? new double[points.dim()] : null;
            long evaluations = 0;
            while (centers.size() < k) {
                Guards.checkInterrupted("k-means++ seeding");
                int cs = points.segmentOf(last);
                PointBlock cb = points.segment(cs);
                int ci = last - points.start(cs);
                if (scratch != null) {
                    cb.scatterRow(ci, scratch);
                }
                double total = 0.0;
                for (int t = 0, p = 0; t < segments; t++) {
                    PointBlock b = points.segment(t);
                    for (int i = 0, r = b.size(); i < r; i++) {
                        if (b.weight(i) > 0.0) {
                            double d = scratch != null ? b.squaredDistanceScatteredToRow(i, scratch, cb, ci)
                                    : b.squaredDistance(i, cb, ci);
                            if (d < minD2[p]) {
                                minD2[p] = d;
                            }
                            score[p] = weights[p] * minD2[p];
                            total += score[p];
                            p++;
                        }
                    }
                }
                if (scratch != null) {
                    cb.clearRow(ci, scratch);
                }
                evaluations += n;
                if (total == 0.0) {
                    break; // every positive-weight point coincides with a chosen center
                }
                last = rows[sample(score, total, rng)];
                centers.add(points.vector(last));
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
