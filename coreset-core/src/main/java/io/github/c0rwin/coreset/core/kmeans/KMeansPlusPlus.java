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
     * step with {@code L} trials costs at most {@code (L + 1) n} (L candidates in one fused pass, then the kept one),
     * plus {@code n} for the first center. Greedy steps skip a point's distance to a candidate when the triangle
     * inequality, with a margin above rounding, proves it cannot beat the point's current {@code D}; the result is
     * bit-identical to evaluating them all. On sparse input a distance costs O(nnz(point)) lookups
     * ({@link SparseVector#scatterPays}, {@link RowCandidates}). The segments are streamed in order, so the result does
     * not depend on how the rows are segmented.
     *
     * @throws IllegalArgumentException if {@code k < 1}, {@code localTrials < 1} or the input overflows
     *     ({@link Guards#requireSummable})
     */
    public static List<SparseVector> seed(PointSet points, int k, int localTrials, RandomGenerator rng,
                                          MetricsSink metrics) {
        return seed(points, k, localTrials, rng, metrics, true);
    }

    /** {@code prune == false} evaluates every greedy distance (tests compare it with the pruned run). */
    static List<SparseVector> seed(PointSet points, int k, int localTrials, RandomGenerator rng, MetricsSink metrics,
                                   boolean prune) {
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
            int maxNnz = 0;
            for (int t = 0; t < segments; t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++) {
                    double w = b.weight(i);
                    moment += w * b.squaredNorm(i);
                    if (w > 0.0) {
                        n++;
                        totalWeight += w;
                        maxNnz = Math.max(maxNnz, b.nnz(i));
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
                greedy(points, k, localTrials, last, rows, weights, centers, rng, metrics,
                        prune ? Bounds.relativeError(2L * maxNnz) : Double.NaN);
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
     * earliest draw); a second pass folds its distances into {@code D^2}. With a usable {@code relativeError} (see
     * {@link Bounds}) both passes skip the distances the triangle inequality rules out. O(n + k * trials) memory
     * besides the input.
     */
    private static void greedy(PointSet points, int k, int trials, int first, int[] rows, double[] weights,
                               List<SparseVector> centers, RandomGenerator rng, MetricsSink metrics,
                               double relativeError) {
        Greedy g = new Greedy(points, trials, rows, weights, relativeError);
        addCandidate(points, g.cand, first);
        double total = g.absorb(-1);
        g.addCenter(first);
        double[] potential = new double[trials];
        double[] draws = new double[trials];
        int[] picks = new int[trials];
        int[] order = new int[trials];
        while (centers.size() < k && total != 0.0) { // zero: every positive-weight point sits on a center
            Guards.checkInterrupted("k-means++ seeding");
            drawMany(g.score, total, rng, draws, order, picks);
            g.cand.clear();
            for (int t = 0; t < trials; t++) {
                addCandidate(points, g.cand, rows[picks[t]]);
            }
            g.thresholds(picks);
            g.potentials(potential);
            int best = 0;
            for (int t = 1; t < trials; t++) {
                if (potential[t] < potential[best]) {
                    best = t;
                }
            }
            int chosen = rows[picks[best]];
            centers.add(points.vector(chosen));
            g.cand.clear();
            addCandidate(points, g.cand, chosen);
            total = g.absorb(best);
            g.addCenter(chosen);
        }
        metrics.count(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS, g.evaluations);
        metrics.count(MetricNames.KMEANSPP_PRUNED_DISTANCES, g.pruned);
    }

    private static void addCandidate(PointSet points, RowCandidates cand, int row) {
        int s = points.segmentOf(row);
        cand.add(points.segment(s), row - points.start(s));
    }

    /**
     * Greedy seeding state. A point p with {@code D(p)^2 = minD2} at center {@code a = nearest[p]} cannot get closer to
     * candidate c when {@code d(a, c) >= 2 D(p)}: then {@code d(p, c) >= d(a, c) - D(p) >= D(p)}. The test runs on
     * squared distances, {@code minD2 <= thr[a][c] = d(a, c)^2 / (4 (1 + 8 err))}; the factor covers every computed
     * distance being off by the relative error {@code err}, so a skipped distance always computes to at least
     * {@code minD2} and leaves {@code min(minD2, d)} and the strict comparisons unchanged.
     */
    private static final class Greedy {
        final PointSet points;
        final int trials;
        final int[] rows;
        final double[] weights;
        final RowCandidates cand;
        final double[] minD2;
        final double[] score;
        final double[] d;
        final boolean prune;
        final double shrink;
        final int[] nearest; // index of the center giving minD2, when pruning
        PointBlock[] centerBlock = new PointBlock[8];
        int[] centerRow = new int[8];
        double[] thr; // [center * trials + candidate]
        int centerCount;
        long evaluations;
        long pruned;
        boolean bounding; // the bounds are worth their cost on this input lately
        boolean stepBounded; // the current step uses them (thr is filled)
        int idleSteps;

        Greedy(PointSet points, int trials, int[] rows, double[] weights, double relativeError) {
            this.points = points;
            this.trials = trials;
            this.rows = rows;
            this.weights = weights;
            boolean map = SparseVector.scatterPays(points.dim(), points.size(), points.storedEntries());
            this.cand = new RowCandidates(points.dim(), trials, map);
            this.minD2 = new double[rows.length];
            Arrays.fill(minD2, Double.POSITIVE_INFINITY);
            this.score = new double[rows.length];
            this.d = new double[trials];
            this.prune = Bounds.usable(relativeError) && trials < Integer.SIZE; // false for NaN; masks are ints
            this.shrink = prune ? 1.0 / (4.0 * (1.0 + 8.0 * relativeError)) : 0.0;
            this.nearest = prune ? new int[rows.length] : null;
            this.thr = prune ? new double[8 * trials] : null;
            this.bounding = prune;
        }

        void addCenter(int row) {
            if (centerCount == centerRow.length) {
                centerBlock = Arrays.copyOf(centerBlock, 2 * centerCount);
                centerRow = Arrays.copyOf(centerRow, 2 * centerCount);
                if (prune) {
                    thr = Arrays.copyOf(thr, 2 * centerCount * trials);
                }
            }
            int s = points.segmentOf(row);
            centerBlock[centerCount] = points.segment(s);
            centerRow[centerCount] = row - points.start(s);
            centerCount++;
        }

        /**
         * Decides whether this step uses the bounds and if so fills {@code thr} for every center and the candidates
         * {@code rows[picks[t]]}: O(centers * trials) merges. A step that skipped under 1/8 of its distances turns the
         * bounds off (their bookkeeping would cost more than it saves) until a probe every fourth step.
         */
        void thresholds(int[] picks) {
            if (prune && !bounding && ++idleSteps >= 4) {
                bounding = true;
            }
            stepBounded = bounding;
            if (!stepBounded) {
                return;
            }
            for (int t = 0; t < trials; t++) {
                int row = rows[picks[t]];
                int s = points.segmentOf(row);
                PointBlock cb = points.segment(s);
                int ci = row - points.start(s);
                for (int c = 0; c < centerCount; c++) {
                    double cc = centerBlock[c].squaredDistance(centerRow[c], cb, ci);
                    thr[c * trials + t] = cc < Double.POSITIVE_INFINITY ? cc * shrink : -1.0;
                }
            }
        }

        /** Whether {@code m} is outside the underflow range where relative error bounds fail. */
        private static boolean prunable(double m) {
            return m == 0.0 || m >= Bounds.TINY_SQUARED;
        }

        /** {@code potential[t] = sum_p w_p min(minD2_p, d(p, candidate t)^2)}, in point order. */
        void potentials(double[] potential) {
            Arrays.fill(potential, 0.0);
            long all = (long) rows.length * trials;
            if (!stepBounded) {
                fullPotentials(potential);
                evaluations += all;
                return;
            }
            long computed = boundedPotentials(potential);
            evaluations += computed;
            pruned += all - computed;
            if (8 * (all - computed) < all) {
                bounding = false;
                idleSteps = 0;
            }
        }

        /** Kept apart from {@link #boundedPotentials} so each hot loop stays small enough to compile well. */
        private void fullPotentials(double[] potential) {
            final int trials = this.trials;
            final double[] minD2 = this.minD2;
            final double[] weights = this.weights;
            final double[] d = this.d;
            final RowCandidates cand = this.cand;
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
        }

        /** The potentials with the bounds of the current step; returns the distances computed. */
        private long boundedPotentials(double[] potential) {
            final int trials = this.trials;
            final double[] minD2 = this.minD2;
            final double[] weights = this.weights;
            final double[] d = this.d;
            final int[] nearest = this.nearest;
            final double[] thr = this.thr;
            final RowCandidates cand = this.cand;
            final int all = (1 << trials) - 1;
            long computed = 0;
            for (int s = 0, q = 0; s < points.segmentCount(); s++) {
                PointBlock b = points.segment(s);
                for (int i = 0, r = b.size(); i < r; i++) {
                    if (b.weight(i) > 0.0) {
                        double w = weights[q];
                        double m = minD2[q];
                        int mask = all;
                        if (prunable(m)) {
                            mask = 0;
                            for (int t = 0, base = nearest[q] * trials; t < trials; t++) {
                                mask |= m > thr[base + t] ? 1 << t : 0;
                            }
                        }
                        if (mask == 0) {
                            for (int t = 0; t < trials; t++) {
                                potential[t] += w * m;
                            }
                        } else {
                            computed += cand.squaredDistances(b, i, d, mask);
                            for (int t = 0; t < trials; t++) {
                                potential[t] += w * ((mask & (1 << t)) != 0 ? Math.min(m, d[t]) : m);
                            }
                        }
                        q++;
                    }
                }
            }
            return computed;
        }

        /**
         * Folds the distances to the single candidate in {@code cand} (candidate {@code best} of the last
         * {@link #thresholds}, or -1 for no bound) into {@code minD2}, the new center getting index
         * {@code centerCount}; refreshes {@code score} and returns its sum.
         */
        double absorb(int best) {
            final double[] minD2 = this.minD2;
            final double[] weights = this.weights;
            final double[] score = this.score;
            final double[] d = this.d;
            final int[] nearest = this.nearest;
            final double[] thr = this.thr;
            final RowCandidates cand = this.cand;
            final boolean bounded = best >= 0 && stepBounded;
            final int trials = this.trials;
            final int index = centerCount;
            long computed = 0;
            double total = 0.0;
            for (int s = 0, q = 0; s < points.segmentCount(); s++) {
                PointBlock b = points.segment(s);
                for (int i = 0, r = b.size(); i < r; i++) {
                    if (b.weight(i) > 0.0) {
                        double m = minD2[q];
                        if (!bounded || !prunable(m) || m > thr[nearest[q] * trials + best]) {
                            cand.squaredDistances(b, i, d);
                            computed++;
                            if (d[0] < m) {
                                minD2[q] = d[0];
                                if (nearest != null) {
                                    nearest[q] = index;
                                }
                            }
                        }
                        score[q] = weights[q] * minD2[q];
                        total += score[q];
                        q++;
                    }
                }
            }
            evaluations += computed;
            pruned += rows.length - computed;
            return total;
        }
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
