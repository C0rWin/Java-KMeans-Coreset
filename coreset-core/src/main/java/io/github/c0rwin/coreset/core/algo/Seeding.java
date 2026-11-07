package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Guards;
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

    /**
     * Clusters {@code points}; with no positive-weight point the result has no centers, assignments -1 and cost 0.
     *
     * @throws IllegalArgumentException if {@code k < 1} or the input overflows ({@link Guards#requireSummable})
     */
    public final Clustering cluster(List<WeightedPoint> points, int k, RandomGenerator rng, MetricsSink metrics) {
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
            Clustering result = run(state, points, k, rng, metrics);
            metrics.count(MetricNames.SEEDING_DISTANCE_EVALUATIONS, state.evaluations);
            metrics.record(MetricNames.SEEDING_CENTERS, result.centers().size());
            return result;
        }
    }

    abstract Clustering run(D2 state, List<WeightedPoint> points, int k, RandomGenerator rng, MetricsSink metrics);

    private static final class KMeansPlusPlusSeeding extends Seeding {
        private final int lloydIterations;

        KMeansPlusPlusSeeding(int lloydIterations) {
            this.lloydIterations = lloydIterations;
        }

        @Override
        Clustering run(D2 s, List<WeightedPoint> points, int k, RandomGenerator rng, MetricsSink metrics) {
            s.add(D2.draw(s.weights, s.totalWeight, rng));
            while (s.centers.size() < k) {
                double total = s.fillD2Mass(null);
                if (total == 0.0) {
                    break; // every positive-weight point coincides with a center
                }
                s.add(D2.draw(s.mass, total, rng));
            }
            if (lloydIterations == 0) {
                return s.toClustering();
            }
            double[] sq = new double[s.n];
            KMeansResult r = WeightedLloyd.refine(points, s.centers, lloydIterations, metrics, sq);
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
        Clustering run(D2 s, List<WeightedPoint> points, int k, RandomGenerator rng, MetricsSink metrics) {
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
            int rounds = 0;
            while (activeCount > 0) {
                rounds++;
                for (int r = 0; r < k; r++) {
                    double total = s.centers.isEmpty() ? s.fillWeightMass(active) : s.fillD2Mass(active);
                    if (total == 0.0) {
                        break; // all remaining points sit on centers and are removed below
                    }
                    s.add(D2.draw(s.mass, total, rng));
                }
                int a = 0;
                double activeWeight = 0.0;
                for (int i = 0; i < n; i++) {
                    if (active[i]) {
                        keys[a++] = d2[i];
                        activeWeight += w[i];
                    }
                }
                Arrays.sort(keys, 0, a);
                double target = removeFraction * activeWeight;
                // Smallest distance threshold whose closed ball holds at least the target weight.
                int lo = 0;
                int hi = a - 1;
                while (lo < hi) {
                    int mid = (lo + hi) >>> 1;
                    if (weightWithin(keys[mid], active, d2, w) >= target) {
                        hi = mid;
                    } else {
                        lo = mid + 1;
                    }
                }
                double threshold = keys[lo];
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

        private static double weightWithin(double threshold, boolean[] active, double[] d2, double[] w) {
            double sum = 0.0;
            for (int i = 0; i < d2.length; i++) {
                if (active[i] && d2[i] <= threshold) {
                    sum += w[i];
                }
            }
            return sum;
        }

        @Override
        public String toString() {
            return "bicriteria(removeFraction=" + removeFraction + ")";
        }
    }

    /** D^2 seeding state: every point's distance to, and index of, its nearest chosen center. */
    static final class D2 {
        final int n;
        final SparseVector[] vectors;
        final double[] weights;
        final double[] minD2;
        final int[] nearest;
        final double[] mass;
        final List<SparseVector> centers = new ArrayList<>();
        final int positive;
        final double totalWeight;
        long evaluations;

        D2(List<WeightedPoint> points) {
            n = points.size();
            vectors = new SparseVector[n];
            weights = new double[n];
            int pos = 0;
            double total = 0.0;
            double moment = 0.0;
            for (int i = 0; i < n; i++) {
                WeightedPoint p = points.get(i);
                vectors[i] = p.vector();
                weights[i] = p.weight();
                moment += p.weight() * p.vector().squaredNorm();
                if (p.weight() > 0.0) {
                    pos++;
                    total += p.weight();
                }
            }
            Guards.requireSummable(total, moment);
            positive = pos;
            totalWeight = total;
            minD2 = new double[n];
            Arrays.fill(minD2, Double.POSITIVE_INFINITY);
            nearest = new int[n];
            Arrays.fill(nearest, -1);
            mass = new double[n];
        }

        void add(int index) {
            SparseVector c = vectors[index];
            int id = centers.size();
            centers.add(c);
            for (int i = 0; i < n; i++) {
                double d = vectors[i].squaredDistance(c);
                if (d < minD2[i]) {
                    minD2[i] = d;
                    nearest[i] = id;
                }
            }
            evaluations += n;
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
