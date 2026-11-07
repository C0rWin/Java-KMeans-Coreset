package io.github.c0rwin.coreset.core.algo;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.MetricsSnapshot;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class SeedingTest {

    private static final List<Seeding> ALL = List.of(
            Seeding.kMeansPlusPlus(0), Seeding.kMeansPlusPlus(5), Seeding.bicriteria(), Seeding.bicriteria(0.25));

    /** The cached assignment must be the true nearest center, since sensitivities are computed from it. */
    @Test
    void assignmentDistancesAndCostMatchBruteForce() {
        Coreset data = TestData.mixture(600, 6, 3, true, 1);
        for (Seeding s : ALL) {
            Clustering c = s.cluster(data.points(), 6, new SplittableRandom(2), MetricsSink.NOOP);
            int[] a = c.assignment();
            double[] sq = c.squaredDistances();
            double cost = 0.0;
            for (int i = 0; i < data.size(); i++) {
                WeightedPoint p = data.points().get(i);
                double best = Double.POSITIVE_INFINITY;
                for (SparseVector center : c.centers()) {
                    best = Math.min(best, p.vector().squaredDistance(center));
                }
                assertEquals(best, p.vector().squaredDistance(c.centers().get(a[i])), 1e-9 * (1 + best), s + " #" + i);
                assertEquals(best, sq[i], 1e-9 * (1 + best), s + " #" + i);
                cost += p.weight() * sq[i];
            }
            assertEquals(cost, c.cost(), 1e-9 * cost, s.toString());
        }
    }

    @Test
    void bicriteriaRemovesWeightGeometrically() {
        Coreset data = TestData.mixture(4096, 4, 2, false, 3);
        Clustering c = Seeding.bicriteria().cluster(data.points(), 4, new SplittableRandom(4), MetricsSink.NOOP);
        // Each round removes at least half the remaining weight, so at most log2(4096) + 1 rounds of 4 centers.
        assertTrue(c.centers().size() >= 4 && c.centers().size() <= 4 * 13, "centers " + c.centers().size());
        double lloyd = WeightedLloyd.cluster(data.points(), 4, 50, new SplittableRandom(5)).cost();
        assertTrue(c.cost() <= lloyd, "bicriteria " + c.cost() + " vs lloyd(k) " + lloyd);
    }

    /**
     * Review L4 guard: bicriteria must keep choosing, per round, the smallest distance threshold whose closed ball
     * holds the target weight. Compares against a direct scan of that definition, on data with many tied distances.
     */
    @Test
    void bicriteriaMatchesTheDirectThresholdDefinition() {
        List<List<WeightedPoint>> inputs = new ArrayList<>();
        inputs.add(TestData.mixture(2000, 5, 2, false, 9).points());
        inputs.add(TestData.mixture(2000, 5, 3, true, 10).points());
        List<WeightedPoint> zeros = new ArrayList<>(TestData.mixture(1500, 4, 2, true, 16).points());
        for (int i = 0; i < zeros.size(); i += 3) {
            zeros.set(i, new WeightedPoint(zeros.get(i).vector(), 0.0));
        }
        inputs.add(zeros);
        List<WeightedPoint> ties = new ArrayList<>();
        for (int i = 0; i < 900; i++) {
            ties.add(point(1.0 + i % 3, i % 30, (i / 30) % 4));
        }
        inputs.add(ties);
        for (List<WeightedPoint> pts : inputs) {
            for (double fraction : new double[] {0.5, 0.25, 0.9}) {
                Clustering fast = Seeding.bicriteria(fraction).cluster(pts, 4, new SplittableRandom(11), MetricsSink.NOOP);
                Clustering direct = directBicriteria(pts, 4, fraction, new SplittableRandom(11));
                assertEquals(direct.centers(), fast.centers(), "fraction " + fraction);
                assertArrayEquals(direct.assignment(), fast.assignment());
            }
        }
    }

    /**
     * Bicriteria with every total recomputed from scratch before each draw and the threshold found by scanning all
     * active points for every candidate.
     */
    private static Clustering directBicriteria(List<WeightedPoint> pts, int k, double fraction, SplittableRandom rng) {
        Seeding.D2 s = new Seeding.D2(pts);
        double[] w = new double[s.n];
        boolean[] active = new boolean[s.n];
        int activeCount = 0;
        for (int i = 0; i < s.n; i++) {
            w[i] = pts.get(i).weight();
            active[i] = w[i] > 0.0;
            activeCount += active[i] ? 1 : 0;
        }
        while (activeCount > 0) {
            for (int r = 0; r < k; r++) {
                boolean first = s.centers.isEmpty();
                double total = 0.0;
                for (int i = 0; i < s.n; i++) {
                    total += active[i] ? (first ? w[i] : w[i] * s.minD2[i]) : 0.0;
                }
                if (total == 0.0) {
                    break;
                }
                s.add(s.draw(active, first, total, rng), active);
            }
            double activeWeight = 0.0;
            for (int i = 0; i < s.n; i++) {
                activeWeight += active[i] ? w[i] : 0.0;
            }
            double threshold = Double.POSITIVE_INFINITY;
            for (int c = 0; c < s.n; c++) {
                if (!active[c] || s.minD2[c] >= threshold) {
                    continue;
                }
                double within = 0.0;
                for (int i = 0; i < s.n; i++) {
                    within += active[i] && s.minD2[i] <= s.minD2[c] ? w[i] : 0.0;
                }
                if (within >= fraction * activeWeight) {
                    threshold = s.minD2[c];
                }
            }
            for (int i = 0; i < s.n; i++) {
                if (active[i] && s.minD2[i] <= threshold) {
                    active[i] = false;
                    activeCount--;
                }
            }
        }
        return s.toClustering();
    }

    @Test
    void weightedThresholdIsTheSmallestQualifyingKey() {
        SplittableRandom rnd = new SplittableRandom(12);
        for (int t = 0; t < 300; t++) {
            int a = 1 + rnd.nextInt(200);
            double[] keys = new double[a];
            double[] weights = new double[a];
            double total = 0.0;
            for (int i = 0; i < a; i++) {
                keys[i] = rnd.nextInt(t % 2 == 0 ? 5 : 1000); // many ties on even trials
                weights[i] = t % 3 == 0 ? 1.0 : 0.5 + rnd.nextInt(4);
                total += weights[i];
            }
            double target = (0.05 + 0.9 * rnd.nextDouble()) * total;
            double expected = Double.POSITIVE_INFINITY;
            for (int c = 0; c < a; c++) {
                double within = 0.0;
                for (int i = 0; i < a; i++) {
                    within += keys[i] <= keys[c] ? weights[i] : 0.0;
                }
                if (within >= target) {
                    expected = Math.min(expected, keys[c]);
                }
            }
            assertEquals(expected, Seeding.sortedThreshold(keys.clone(), weights.clone(), 0, a, target), "trial " + t);
            for (int budget = 0; budget < 4; budget++) { // sorting fallback after 0..3 partitions
                assertEquals(expected, Seeding.weightedThreshold(keys.clone(), weights.clone(), a, target, budget),
                        "trial " + t + " budget " + budget);
            }
            assertEquals(expected, Seeding.weightedThreshold(keys, weights, a, target), "trial " + t);
        }
    }

    /**
     * Guard: bicriteria reads the points once, once per center (the last center of a round also gathers the
     * threshold keys), once per round to remove points (none after a round that ends with every active point on a
     * center), and once for the final clustering. kmeans++ needs no per-round passes.
     */
    @Test
    void seedingPassesAreCounted() {
        List<WeightedPoint> pts = TestData.mixture(3000, 5, 3, true, 13).points();
        long n = pts.size();
        for (Seeding s : List.of(Seeding.bicriteria(), Seeding.bicriteria(0.25))) {
            RecordingMetrics m = new RecordingMetrics();
            Clustering c = s.cluster(pts, 5, new SplittableRandom(14), m);
            MetricsSnapshot snap = m.snapshot();
            long rounds = (long) snap.distributions().get(MetricNames.SEEDING_BICRITERIA_ROUNDS).max();
            long centers = c.centers().size();
            long scanned = snap.counters().get(MetricNames.SEEDING_POINTS_SCANNED);
            assertTrue(scanned == n * (2 + centers + rounds) || scanned == n * (1 + centers + rounds),
                    s + ": " + scanned / n + " passes for " + centers + " centers, " + rounds + " rounds");
            assertTrue(snap.counters().get(MetricNames.SEEDING_DRAW_POINTS_SCANNED) <= n * centers);
        }
        RecordingMetrics m = new RecordingMetrics();
        Clustering c = Seeding.kMeansPlusPlus(0).cluster(pts, 5, new SplittableRandom(15), m);
        assertEquals(n * (2 + c.centers().size()), m.snapshot().counters().get(MetricNames.SEEDING_POINTS_SCANNED));
    }

    @Test
    void centersAreDistinctPositiveWeightInputPoints() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            pts.add(point(1.0, i % 3, 0)); // three distinct locations
        }
        pts.add(point(0.0, 100, 100));
        for (Seeding s : List.of(Seeding.kMeansPlusPlus(0), Seeding.bicriteria())) {
            Clustering c = s.cluster(pts, 10, new SplittableRandom(6), MetricsSink.NOOP);
            assertEquals(3, c.centers().size(), s.toString());
            assertEquals(3, new HashSet<>(c.centers()).size());
            assertFalse(c.centers().contains(SparseVector.dense(new double[] {100, 100})));
            assertEquals(0.0, c.cost());
        }
    }

    @Test
    void noPositiveWeightGivesEmptyClustering() {
        List<WeightedPoint> pts = List.of(point(0.0, 1, 2), point(0.0, 3, 4));
        Clustering c = Seeding.bicriteria().cluster(pts, 2, new SplittableRandom(7), MetricsSink.NOOP);
        assertTrue(c.centers().isEmpty());
        assertEquals(-1, c.assignment()[1]);
    }

    @Test
    void rejectsInvalidParameters() {
        assertThrows(IllegalArgumentException.class, () -> Seeding.kMeansPlusPlus(-1));
        assertThrows(IllegalArgumentException.class, () -> Seeding.bicriteria(0.0));
        assertThrows(IllegalArgumentException.class, () -> Seeding.bicriteria(1.0));
        assertThrows(IllegalArgumentException.class,
                () -> Seeding.bicriteria().cluster(List.of(point(1, 0)), 0, new SplittableRandom(8), MetricsSink.NOOP));
    }
}
