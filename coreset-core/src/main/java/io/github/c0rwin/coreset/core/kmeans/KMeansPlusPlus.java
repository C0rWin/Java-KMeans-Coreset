package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.RowCandidates;
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
 *
 * <p>By default each step is greedy (Arthur and Vassilvitskii; scikit-learn's default): {@link #defaultLocalTrials}
 * candidates are drawn by D^2 and the one leaving the smallest potential {@code sum w * min(D^2, d(p, cand)^2)} is
 * kept. Plain D^2 seeding (one trial) misses small well-separated clusters far more often, which Lloyd cannot repair;
 * it stays available with {@code localTrials = 1}.
 */
public final class KMeansPlusPlus {

    private KMeansPlusPlus() {
    }

    public static List<SparseVector> seed(List<WeightedPoint> points, int k, RandomGenerator rng) {
        return seed(points, k, rng, MetricsSink.NOOP);
    }

    /** {@code 2 + floor(ln k)} candidates per greedy step, as in scikit-learn. */
    public static int defaultLocalTrials(int k) {
        return 2 + (int) Math.log(k);
    }

    /** {@link #seed(PointSet, int, RandomGenerator, MetricsSink)} on a copy of {@code points} in one block. */
    public static List<SparseVector> seed(List<WeightedPoint> points, int k, RandomGenerator rng,
                                          MetricsSink metrics) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        return seed(PointSet.of(PointBlock.of(points)), k, rng, metrics);
    }

    /** Greedy seeding with {@link #defaultLocalTrials} candidates per step. */
    public static List<SparseVector> seed(PointSet points, int k, RandomGenerator rng, MetricsSink metrics) {
        return seed(points, k, defaultLocalTrials(k), rng, metrics);
    }

    /**
     * Returns {@code min(k, number of distinct points with positive weight)} distinct centers, chosen from the
     * input (as views of its rows). Zero-weight points are never chosen. With {@code localTrials = 1} (plain D^2)
     * performs {@code n * (c - 1)} distance evaluations when {@code c = k} centers are returned and {@code n * c} when
     * the input runs out of distinct points first, where {@code n} is the number of positive-weight points; a greedy
     * step with {@code L} trials costs {@code (L + 1) n} (L candidates in one fused pass, then the kept one), plus
     * {@code n} for the first center. On sparse input a distance costs O(nnz(point)) lookups
     * ({@link SparseVector#scatterPays}, {@link RowCandidates}). The segments are streamed in order, so the result does
     * not depend on how the rows are segmented.
     *
     * @throws IllegalArgumentException if {@code k < 1}, {@code localTrials < 1} or the input overflows
     *     ({@link Guards#requireSummable})
     */
    public static List<SparseVector> seed(PointSet points, int k, int localTrials, RandomGenerator rng,
                                          MetricsSink metrics) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        if (localTrials < 1) {
            throw new IllegalArgumentException("localTrials must be >= 1: " + localTrials);
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
            if (localTrials > 1) {
                greedy(points, k, localTrials, last, rows, weights, centers, rng, metrics);
                return centers;
            }

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

    /**
     * Greedy steps after the first center {@code first}: each draws {@code trials} candidates by D^2, evaluates them
     * together in one pass over the positive-weight points and keeps the one with the smallest potential (ties: the
     * earliest draw); a second pass folds its distances into {@code D^2}. O(n + trials) memory besides the input.
     */
    private static void greedy(PointSet points, int k, int trials, int first, int[] rows, double[] weights,
                               List<SparseVector> centers, RandomGenerator rng, MetricsSink metrics) {
        final int n = rows.length;
        boolean map = SparseVector.scatterPays(points.dim(), points.size(), points.storedEntries());
        RowCandidates cand = new RowCandidates(points.dim(), trials, map);
        double[] minD2 = new double[n];
        Arrays.fill(minD2, Double.POSITIVE_INFINITY);
        double[] score = new double[n];
        double[] d = new double[trials];
        double[] potential = new double[trials];
        double[] draws = new double[trials];
        int[] picks = new int[trials];
        int[] order = new int[trials];
        addCandidate(points, cand, first);
        double total = absorb(points, cand, weights, minD2, score, d);
        long evaluations = n;
        while (centers.size() < k && total != 0.0) { // zero: every positive-weight point sits on a center
            Guards.checkInterrupted("k-means++ seeding");
            drawMany(score, total, rng, draws, order, picks);
            cand.clear();
            for (int t = 0; t < trials; t++) {
                addCandidate(points, cand, rows[picks[t]]);
            }
            Arrays.fill(potential, 0.0);
            for (int s = 0, q = 0; s < points.segmentCount(); s++) {
                PointBlock b = points.segment(s);
                for (int i = 0, r = b.size(); i < r; i++) {
                    if (b.weight(i) > 0.0) {
                        cand.squaredDistances(b, i, d);
                        double w = weights[q];
                        double m = minD2[q];
                        for (int t = 0; t < trials; t++) {
                            potential[t] += w * Math.min(m, d[t]);
                        }
                        q++;
                    }
                }
            }
            int best = 0;
            for (int t = 1; t < trials; t++) {
                if (potential[t] < potential[best]) {
                    best = t;
                }
            }
            int chosen = rows[picks[best]];
            centers.add(points.vector(chosen));
            cand.clear();
            addCandidate(points, cand, chosen);
            total = absorb(points, cand, weights, minD2, score, d);
            evaluations += (long) n * (trials + 1);
        }
        metrics.count(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS, evaluations);
    }

    private static void addCandidate(PointSet points, RowCandidates cand, int row) {
        int s = points.segmentOf(row);
        cand.add(points.segment(s), row - points.start(s));
    }

    /** Folds the distances to the single candidate into {@code minD2}; refreshes {@code score} and returns its sum. */
    private static double absorb(PointSet points, RowCandidates cand, double[] weights, double[] minD2,
                                 double[] score, double[] d) {
        double total = 0.0;
        for (int s = 0, q = 0; s < points.segmentCount(); s++) {
            PointBlock b = points.segment(s);
            for (int i = 0, r = b.size(); i < r; i++) {
                if (b.weight(i) > 0.0) {
                    cand.squaredDistances(b, i, d);
                    if (d[0] < minD2[q]) {
                        minD2[q] = d[0];
                    }
                    score[q] = weights[q] * minD2[q];
                    total += score[q];
                    q++;
                }
            }
        }
        return total;
    }

    /**
     * {@code picks[t]} = {@link #sample} of {@code draws[t] = rng.nextDouble() * total}, the draws taken in order
     * {@code t = 0, 1, ...}, resolved together in one sweep over {@code mass}.
     */
    private static void drawMany(double[] mass, double total, RandomGenerator rng, double[] draws, int[] order,
                                 int[] picks) {
        final int trials = draws.length;
        if (!Double.isFinite(total)) {
            Arrays.fill(picks, argMax(mass));
            return;
        }
        // trials is tiny: insertion sort of the draw order
        for (int t = 0; t < trials; t++) {
            double u = rng.nextDouble() * total;
            draws[t] = u;
            int j = t;
            while (j > 0 && draws[order[j - 1]] > u) {
                order[j] = order[j - 1];
                j--;
            }
            order[j] = t;
        }
        double cumulative = 0.0;
        int lastPositive = -1;
        int next = 0;
        for (int i = 0; i < mass.length && next < trials; i++) {
            double m = mass[i];
            if (m > 0.0) {
                cumulative += m;
                lastPositive = i;
                while (next < trials && draws[order[next]] < cumulative) {
                    picks[order[next++]] = i;
                }
            }
        }
        while (next < trials) {
            picks[order[next++]] = lastPositive; // rounding left the draw at the very top of the range
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
