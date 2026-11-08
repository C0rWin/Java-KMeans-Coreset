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
            metrics.count(MetricNames.SEEDING_POINTS_SCANNED, state.scanned);
            metrics.count(MetricNames.SEEDING_DRAW_POINTS_SCANNED, state.drawScanned);
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
            double total = s.add(s.draw(null, true, s.totalWeight, rng), null);
            while (s.centers.size() < k && total != 0.0) { // zero: every positive-weight point sits on a center
                total = s.add(s.draw(null, false, total, rng), null);
            }
            if (lloydIterations == 0) {
                return s.toClustering();
            }
            double[] sq = new double[s.n];
            KMeansResult r = WeightedLloyd.refine(s.points, s.centers, lloydIterations, metrics, sq);
            int[] assignment = r.assignment();
            int clusters = r.centers().size();
            double[] clusterWeights = new double[clusters];
            double[] clusterCosts = new double[clusters];
            for (int t = 0, g = 0; t < s.points.segmentCount(); t++) {
                PointBlock b = s.points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double w = b.weight(i);
                    if (w > 0.0) {
                        clusterWeights[assignment[g]] += w;
                        clusterCosts[assignment[g]] += w * sq[g];
                    }
                }
            }
            s.scanned += s.n;
            return new Clustering(r.centers(), assignment, sq, r.cost(), clusterWeights, clusterCosts);
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

        /**
         * Per round: k center passes, the last of which also gathers the active distances and weights for the
         * threshold search (which then works on those arrays only), and one pass that removes the closest points and
         * sums the survivors' D^2 mass for the next round's first draw.
         */
        @Override
        Clustering run(D2 s, int k, RandomGenerator rng, MetricsSink metrics) {
            final PointSet points = s.points;
            final double[] d2 = s.minD2;
            boolean[] active = new boolean[s.n];
            int activeCount = 0;
            double total = 0.0;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double w = b.weight(i);
                    if (w > 0.0) {
                        active[g] = true;
                        activeCount++;
                        total += w;
                    }
                }
            }
            s.scanned += s.n;
            double[] keys = new double[activeCount];
            double[] keyWeights = new double[activeCount];
            int rounds = 0;
            while (activeCount > 0) {
                rounds++;
                for (int r = 0; r < k && total != 0.0; r++) {
                    boolean last = r == k - 1;
                    total = s.add(s.draw(active, s.centers.isEmpty(), total, rng), active,
                            last ? keys : null, last ? keyWeights : null);
                }
                if (total == 0.0) {
                    break; // every active point sits on a center: no further center can be drawn
                }
                double threshold = weightedThreshold(keys, keyWeights, s.collected,
                        removeFraction * s.collectedWeight);
                total = 0.0;
                for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                    PointBlock b = points.segment(t);
                    for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                        if (active[g]) {
                            if (d2[g] <= threshold) {
                                active[g] = false;
                                activeCount--;
                            } else {
                                total += b.weight(i) * d2[g];
                            }
                        }
                    }
                }
                s.scanned += s.n;
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
        return weightedThreshold(keys, weights, a, target, 2 * (32 - Integer.numberOfLeadingZeros(a)) + 8);
    }

    /** {@link #weightedThreshold(double[], double[], int, double)} sorting the rest after {@code budget} partitions. */
    static double weightedThreshold(double[] keys, double[] weights, int a, double target, int budget) {
        int lo = 0;
        int hi = a;
        double largest = keys[0];
        for (int i = 1; i < a; i++) {
            largest = Math.max(largest, keys[i]);
        }
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

    /**
     * The same threshold on {@code [lo, hi)}: heapsorts both arrays in place by key, then one prefix-sum pass stops at
     * the first key whose run of ties brings the weight to {@code target}. O(len log len), no allocation.
     */
    static double sortedThreshold(double[] keys, double[] weights, int lo, int hi, double target) {
        final int len = hi - lo;
        for (int i = len / 2 - 1; i >= 0; i--) {
            siftDown(keys, weights, lo, i, len);
        }
        for (int end = len - 1; end > 0; end--) {
            swap(keys, weights, lo, lo + end);
            siftDown(keys, weights, lo, 0, end);
        }
        double within = 0.0;
        for (int i = lo; i < hi; i++) {
            within += weights[i];
            if ((i + 1 == hi || keys[i + 1] != keys[i]) && within >= target) {
                return keys[i];
            }
        }
        return keys[hi - 1]; // rounding left target above the total
    }

    /** Max-heap sift on {@code keys[lo, lo + size)}, carrying the weights along. */
    private static void siftDown(double[] keys, double[] weights, int lo, int root, int size) {
        while (true) {
            int child = 2 * root + 1;
            if (child >= size) {
                return;
            }
            if (child + 1 < size && keys[lo + child + 1] > keys[lo + child]) {
                child++;
            }
            if (!(keys[lo + child] > keys[lo + root])) {
                return;
            }
            swap(keys, weights, lo + root, lo + child);
            root = child;
        }
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

    /**
     * D^2 seeding state: every point's distance to, and index of, its nearest chosen center (12 bytes per point).
     * Weights are read from the segments, and sampling masses {@code w * minD2} are recomputed where needed instead
     * of being stored, with the same products in the same order.
     */
    static final class D2 {
        final int n;
        final PointSet points;
        final double[] minD2;
        final int[] nearest;
        final List<SparseVector> centers = new ArrayList<>();
        final int positive;
        final double totalWeight;
        long evaluations;
        long scanned; // points visited by full passes
        long drawScanned; // points visited by draws
        int collected; // entries gathered by the last collecting add
        double collectedWeight; // their weight, added in row order
        private final boolean useScratch;
        private double[] scratch; // O(dim), zero outside the center being added

        D2(List<WeightedPoint> points) {
            this(PointSet.of(PointBlock.of(points)));
        }

        D2(PointSet points) {
            n = points.size();
            this.points = points;
            int pos = 0;
            double total = 0.0;
            double moment = 0.0;
            for (int t = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++) {
                    double w = b.weight(i);
                    moment += w * b.squaredNorm(i);
                    if (w > 0.0) {
                        pos++;
                        total += w;
                    }
                }
            }
            scanned += n;
            Guards.requireSummable(total, moment);
            positive = pos;
            totalWeight = total;
            useScratch = n > 0 && SparseVector.scatterPays(points.dim(), n, points.storedEntries());
            minD2 = new double[n];
            Arrays.fill(minD2, Double.POSITIVE_INFINITY);
            nearest = new int[n];
            Arrays.fill(nearest, -1);
        }

        double add(int index, boolean[] active) {
            return add(index, active, null, null);
        }

        /**
         * Adds row {@code index} as a center in one pass over the points (preceded by a cancellation check): on sparse
         * input the center is scattered into a dense scratch once, so each distance costs O(nnz(point)) lookups
         * ({@link SparseVector#scatterPays}); otherwise supports are merged. Returns {@code sum w * minD2} over the
         * eligible points ({@code active == null}: all) in row order, computed in the same pass. With {@code keys}
         * set, the same pass also stores every eligible point's new distance and weight in row order (see
         * {@link #collected}, {@link #collectedWeight}).
         */
        double add(int index, boolean[] active, double[] keys, double[] keyWeights) {
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
            int a = 0;
            double weight = 0.0;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double d = useScratch ? b.squaredDistanceScatteredToRow(i, scratch, cb, ci)
                            : b.squaredDistance(i, cb, ci);
                    if (d < minD2[g]) {
                        minD2[g] = d;
                        nearest[g] = id;
                    }
                    if (active == null || active[g]) {
                        double w = b.weight(i);
                        total += w * minD2[g];
                        if (keys != null) {
                            keys[a] = minD2[g];
                            keyWeights[a++] = w;
                            weight += w;
                        }
                    }
                }
            }
            if (useScratch) {
                cb.clearRow(ci, scratch);
            }
            if (keys != null) {
                collected = a;
                collectedWeight = weight;
            }
            evaluations += n;
            scanned += n;
            return total;
        }

        /**
         * Row drawn with probability proportional to its mass among eligible points ({@code active == null}: all):
         * {@code w} when {@code byWeight}, else {@code w * minD2}. Only positive masses qualify; {@code total} is the
         * sum of the masses. The scan stops at the drawn row.
         */
        int draw(boolean[] active, boolean byWeight, double total, RandomGenerator rng) {
            if (!Double.isFinite(total)) {
                return argMax(active, byWeight); // overflowed masses cannot be sampled proportionally
            }
            double r = rng.nextDouble() * total;
            double cumulative = 0.0;
            int lastPositive = -1;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double v = mass(b.weight(i), g, active, byWeight);
                    if (v > 0.0) {
                        cumulative += v;
                        lastPositive = g;
                        if (r < cumulative) {
                            drawScanned += g + 1;
                            return g;
                        }
                    }
                }
            }
            drawScanned += n;
            return lastPositive; // rounding left r at the very top of the range
        }

        private double mass(double w, int g, boolean[] active, boolean byWeight) {
            return active == null || active[g] ? (byWeight ? w : w * minD2[g]) : 0.0;
        }

        private int argMax(boolean[] active, boolean byWeight) {
            int best = 0;
            double bestMass = Double.NEGATIVE_INFINITY;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double v = mass(b.weight(i), g, active, byWeight);
                    if (v > bestMass) {
                        bestMass = v;
                        best = g;
                    }
                }
            }
            drawScanned += n;
            return best;
        }

        /** The clustering with its cost and per-center totals, gathered in one pass. */
        Clustering toClustering() {
            int k = centers.size();
            double[] clusterWeights = new double[k];
            double[] clusterCosts = new double[k];
            double cost = 0.0;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock b = points.segment(t);
                for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                    double w = b.weight(i);
                    double c = w * minD2[g];
                    cost += c;
                    if (w > 0.0) {
                        clusterWeights[nearest[g]] += w;
                        clusterCosts[nearest[g]] += c;
                    }
                }
            }
            scanned += n;
            return new Clustering(centers, nearest, minD2, cost, clusterWeights, clusterCosts);
        }
    }
}
