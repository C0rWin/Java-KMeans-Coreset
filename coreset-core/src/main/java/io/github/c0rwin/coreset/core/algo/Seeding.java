package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Computes the rough clustering B that {@link SensitivityCoreset} derives sensitivities from. Both strategies keep the
 * nearest-center index and distance of every point while seeding, so no separate assignment pass is needed.
 *
 * <p>Instances are immutable and thread-safe; all randomness comes from the supplied generator.
 */
public abstract class Seeding {

    private Seeding() {
    }

    /**
     * Weighted k-means++ (D^2) seeding, the same distribution as {@code kmeans.KMeansPlusPlus}, followed by up to
     * {@code lloydIterations} Lloyd steps. Seeding costs {@code n * k} distance evaluations; Lloyd adds
     * {@code n * k} per iteration.
     *
     * @throws IllegalArgumentException if {@code lloydIterations < 0}
     */
    public static Seeding kMeansPlusPlus(int lloydIterations) {
        if (lloydIterations < 0) {
            throw new IllegalArgumentException("lloydIterations must be >= 0: " + lloydIterations);
        }
        return new KMeansPlusPlusSeeding(lloydIterations);
    }

    /** {@link #bicriteria(double)} removing the closest half of the remaining weight per round. */
    public static Seeding bicriteria() {
        return bicriteria(0.5);
    }

    /**
     * Bicriteria seeding by repeated sample-and-remove: each round draws k centers by D^2 sampling among the remaining
     * points, then removes the remaining points closest to the centers chosen so far until {@code removeFraction} of
     * the remaining weight is gone. Rounds repeat until no positive-weight point remains, giving about
     * {@code k * log(n) / log(1 / (1 - removeFraction))} centers and {@code n} distance evaluations per center.
     *
     * @throws IllegalArgumentException unless {@code 0 < removeFraction < 1}
     */
    public static Seeding bicriteria(double removeFraction) {
        if (!(removeFraction > 0.0 && removeFraction < 1.0)) {
            throw new IllegalArgumentException("removeFraction must be in (0, 1): " + removeFraction);
        }
        return new BicriteriaSeeding(removeFraction);
    }

    /** {@link #cluster(PointSet, int, RandomGenerator, MetricsSink)} on a copy of {@code points} in one block. */
    public final Clustering cluster(List<WeightedPoint> points, int k, RandomGenerator rng, MetricsSink metrics) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        return cluster(PointSet.of(PointBlock.of(points)), k, rng, metrics);
    }

    /**
     * Clusters {@code points}; with no positive-weight point the result has no centers, assignments -1 and cost 0.
     *
     * @throws IllegalArgumentException if {@code k < 1} or the input overflows ({@link Guards#requireSummable})
     */
    public final Clustering cluster(PointSet points, int k, RandomGenerator rng, MetricsSink metrics) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        try (Span ignored = metrics.span(MetricNames.SEEDING_CLUSTER)) {
            D2 state = new D2(points);
            if (state.positive == 0) {
                int[] none = new int[points.size()];
                Arrays.fill(none, -1);
                return new Clustering(List.of(), none, new double[points.size()], 0.0);
            }
            Clustering result = run(state, k, rng, metrics);
            metrics.count(MetricNames.SEEDING_DISTANCE_EVALUATIONS, state.evaluations);
            metrics.record(MetricNames.SEEDING_CENTERS, result.centers().size());
            return result;
        }
    }

    abstract Clustering run(D2 state, int k, RandomGenerator rng, MetricsSink metrics);

    private static final class KMeansPlusPlusSeeding extends Seeding {
        private final int lloydIterations;

        KMeansPlusPlusSeeding(int lloydIterations) {
            this.lloydIterations = lloydIterations;
        }

        @Override
        Clustering run(D2 s, int k, RandomGenerator rng, MetricsSink metrics) {
            double total = s.add(D2.draw(s.weights, s.totalWeight, rng), null);
            while (s.centers.size() < k && total != 0.0) { // zero: every positive-weight point sits on a center
                total = s.add(D2.draw(s.mass, total, rng), null);
            }
            if (lloydIterations == 0) {
                return s.toClustering();
            }
            double[] sq = new double[s.n];
            KMeansResult r = WeightedLloyd.refine(s.points, s.centers, lloydIterations, metrics, sq);
            return new Clustering(r.centers(), r.assignment(), sq, r.cost());
        }

        @Override
        public String toString() {
            return "kmeans++(lloydIterations=" + lloydIterations + ")";
        }
    }

    private static final class BicriteriaSeeding extends Seeding {
        private final double removeFraction;

        BicriteriaSeeding(double removeFraction) {
            this.removeFraction = removeFraction;
        }

        @Override
        Clustering run(D2 s, int k, RandomGenerator rng, MetricsSink metrics) {
            final int n = s.n;
            final double[] w = s.weights;
            final double[] d2 = s.minD2;
            boolean[] active = new boolean[n];
            int activeCount = 0;
            for (int i = 0; i < n; i++) {
                if (w[i] > 0.0) {
                    active[i] = true;
                    activeCount++;
                }
            }
            double[] keys = new double[activeCount];
            double[] keyWeights = new double[activeCount];
            int rounds = 0;
            while (activeCount > 0) {
                rounds++;
                for (int r = 0; r < k; r++) {
                    double total = s.centers.isEmpty() ? s.fillWeightMass(active) : s.fillD2Mass(active);
                    if (total == 0.0) {
                        break; // all remaining points sit on centers and are removed below
                    }
                    s.add(D2.draw(s.mass, total, rng), active);
                }
                int a = 0;
                double activeWeight = 0.0;
                for (int i = 0; i < n; i++) {
                    if (active[i]) {
                        keys[a] = d2[i];
                        keyWeights[a++] = w[i];
                        activeWeight += w[i];
                    }
                }
                double threshold = weightedThreshold(keys, keyWeights, a, removeFraction * activeWeight);
                for (int i = 0; i < n; i++) {
                    if (active[i] && d2[i] <= threshold) {
                        active[i] = false;
                        activeCount--;
                    }
                }
            }
            metrics.record(MetricNames.SEEDING_BICRITERIA_ROUNDS, rounds);
            return s.toClustering();
        }

        @Override
        public String toString() {
            return "bicriteria(removeFraction=" + removeFraction + ")";
        }
    }

    /**
     * Smallest key t among {@code keys[0, a)} whose closed ball holds weight {@code >= target}, i.e. the weighted
     * {@code target} quantile, found by weighted quickselect with a three-way partition (reorders both arrays):
     * expected O(a). After {@code 2 log2(a) + 8} partitions the rest is sorted, so the worst case is O(a log a).
     */
    static double weightedThreshold(double[] keys, double[] weights, int a, double target) {
        int lo = 0;
        int hi = a;
        double largest = keys[0];
        for (int i = 1; i < a; i++) {
            largest = Math.max(largest, keys[i]);
        }
        int budget = 2 * (32 - Integer.numberOfLeadingZeros(a)) + 8;
        while (hi - lo > 1) {
            if (budget-- == 0) {
                return sortedThreshold(keys, weights, lo, hi, target);
            }
            double pivot = medianOfThree(keys[lo], keys[(lo + hi) >>> 1], keys[hi - 1]);
            int lt = lo;
            int gt = hi;
            double less = 0.0;
            double equal = 0.0;
            for (int i = lo; i < gt; ) {
                double key = keys[i];
                if (key < pivot) {
                    less += weights[i];
                    swap(keys, weights, lt++, i++);
                } else if (key > pivot) {
                    swap(keys, weights, i, --gt);
                } else {
                    equal += weights[i++];
                }
            }
            if (less >= target) {
                hi = lt;
            } else if (less + equal >= target) {
                return pivot;
            } else {
                target -= less + equal;
                lo = gt;
            }
        }
        return hi > lo ? keys[lo] : largest; // empty only when rounding left target above the total
    }

    /** The same threshold on {@code [lo, hi)} by sorting a copy of the keys and binary-searching: O(len log len). */
    static double sortedThreshold(double[] keys, double[] weights, int lo, int hi, double target) {
        double[] sorted = Arrays.copyOfRange(keys, lo, hi);
        Arrays.sort(sorted);
        int l = 0;
        int h = sorted.length - 1;
        while (l < h) {
            int mid = (l + h) >>> 1;
            double within = 0.0;
            for (int i = lo; i < hi; i++) {
                within += keys[i] <= sorted[mid] ? weights[i] : 0.0;
            }
            if (within >= target) {
                h = mid;
            } else {
                l = mid + 1;
            }
        }
        return sorted[l];
    }

    private static double medianOfThree(double x, double y, double z) {
        return Math.max(Math.min(x, y), Math.min(Math.max(x, y), z));
    }

    private static void swap(double[] keys, double[] weights, int i, int j) {
        double k = keys[i];
        keys[i] = keys[j];
        keys[j] = k;
        double w = weights[i];
        weights[i] = weights[j];
        weights[j] = w;
    }

    /** D^2 seeding state: every point's distance to, and index of, its nearest chosen center. */
    static final class D2 {
        final int n;
        final PointSet points;
        final double[] weights;
        final double[] minD2;
        final int[] nearest;
        final double[] mass;
        final List<SparseVector> centers = new ArrayList<>();
        final int positive;
        final double totalWeight;
        long evaluations;
        private final boolean useScratch;
        private double[] scratch; // O(dim), zero outside the center being added

        D2(List<WeightedPoint> points) {
            this(PointSet.of(PointBlock.of(points)));
        }

        D2(PointSet points) {
            n = points.size();
            this.points = points;
            weights = new double[n];
            int pos = 0;
            double total = 0.0;
            double moment = 0.0;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double w = b.weight(i);
                    weights[g] = w;
                    moment += w * b.squaredNorm(i);
                    if (w > 0.0) {
                        pos++;
                        total += w;
                    }
                }
            }
            Guards.requireSummable(total, moment);
            positive = pos;
            totalWeight = total;
            useScratch = n > 0 && SparseVector.scatterPays(points.dim(), n, points.storedEntries());
            minD2 = new double[n];
            Arrays.fill(minD2, Double.POSITIVE_INFINITY);
            nearest = new int[n];
            Arrays.fill(nearest, -1);
            mass = new double[n];
        }

        /**
         * Adds row {@code index} as a center in one pass over the points (preceded by a cancellation check): on sparse
         * input the center is scattered into a dense scratch once, so each distance costs O(nnz(point)) lookups
         * ({@link SparseVector#scatterPays}); otherwise supports are merged. The same pass refreshes {@code mass} as
         * {@link #fillD2Mass} would and returns its total.
         */
        double add(int index, boolean[] active) {
            Guards.checkInterrupted("seeding");
            int id = centers.size();
            int cs = points.segmentOf(index);
            PointBlock cb = points.segment(cs);
            int ci = index - points.start(cs);
            centers.add(cb.vector(ci));
            if (useScratch) {
                if (scratch == null) {
                    scratch = new double[points.dim()];
                }
                cb.scatterRow(ci, scratch);
            }
            double total = 0.0;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double d = useScratch ? b.squaredDistanceScatteredToRow(i, scratch, cb, ci)
                            : b.squaredDistance(i, cb, ci);
                    if (d < minD2[g]) {
                        minD2[g] = d;
                        nearest[g] = id;
                    }
                    double v = active == null || active[g] ? weights[g] * minD2[g] : 0.0;
                    mass[g] = v;
                    total += v;
                }
            }
            if (useScratch) {
                cb.clearRow(ci, scratch);
            }
            evaluations += n;
            return total;
        }

        /** {@code mass = w * minD2} over eligible points ({@code active == null}: all); returns the total. */
        double fillD2Mass(boolean[] active) {
            double total = 0.0;
            for (int i = 0; i < n; i++) {
                double v = active == null || active[i] ? weights[i] * minD2[i] : 0.0;
                mass[i] = v;
                total += v;
            }
            return total;
        }

        double fillWeightMass(boolean[] active) {
            double total = 0.0;
            for (int i = 0; i < n; i++) {
                double v = active[i] ? weights[i] : 0.0;
                mass[i] = v;
                total += v;
            }
            return total;
        }

        Clustering toClustering() {
            double cost = 0.0;
            for (int i = 0; i < n; i++) {
                cost += weights[i] * minD2[i];
            }
            return new Clustering(centers, nearest, minD2, cost);
        }

        /** Index drawn with probability proportional to {@code mass[i]}; only positive masses qualify. */
        static int draw(double[] mass, double total, RandomGenerator rng) {
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
}
