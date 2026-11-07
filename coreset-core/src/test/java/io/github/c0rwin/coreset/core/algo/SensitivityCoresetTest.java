package io.github.c0rwin.coreset.core.algo;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.MetricsSnapshot;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import io.github.c0rwin.coreset.core.validate.TheoryBounds;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class SensitivityCoresetTest {

    /** The construction before the in-place scratch: fresh bounds, an int[n] of counts and a scan over them. */
    private static SampledCoreset countsReference(Coreset input, int k, int m, Seeding seeding, SensitivityBound bound,
                                                  SplittableRandom rng) {
        PointSet points = input.pointSet();
        Clustering b = seeding.cluster(points, k, rng, MetricsSink.NOOP);
        double[] s = bound.sensitivities(points, b);
        double total = 0.0;
        for (double v : s) {
            total += v;
        }
        int[] counts = AliasSampler.drawCounts(s, m, rng);
        List<Integer> rows = new ArrayList<>();
        for (int g = 0; g < counts.length; g++) {
            if (counts[g] > 0) {
                rows.add(g);
            }
        }
        int[] r = new int[rows.size()];
        double[] w = new double[r.length];
        int[] c = new int[r.length];
        double[] q = new double[r.length];
        for (int j = 0; j < r.length; j++) {
            r[j] = rows.get(j);
            c[j] = counts[r[j]];
            q[j] = s[r[j]] / total;
            w[j] = points.weight(r[j]) * c[j] / (m * q[j]);
        }
        return new SampledCoreset(new Coreset(points.select(r, w, r.length), input.offset()), c, q, m, total);
    }

    /** Bounds written over the seeding's distances and sorted draws give exactly the previous output and provenance. */
    @Test
    void outputIsBitIdenticalToTheCountsConstruction() {
        Coreset weighted = new Coreset(TestData.mixture(2500, 5, 3, true, 41).points(), 1.5);
        PointSet ps = weighted.pointSet();
        Coreset segmented = new Coreset(PointSet.concat(List.of(ps.slice(0, 900),
                PointSet.of(PointBlock.of(ps.slice(900, 1700).asList())), ps.slice(1700, 2500))), 1.5);
        List<Seeding> seedings = List.of(Seeding.kMeansPlusPlus(0), Seeding.kMeansPlusPlus(2), Seeding.bicriteria());
        List<SensitivityBound> bounds = List.of(SensitivityBound.standard(2.0), SensitivityBound.legacy());
        for (Coreset input : List.of(weighted, segmented, TestData.mixture(2000, 4, 2, false, 42))) {
            for (Seeding seeding : seedings) {
                for (SensitivityBound bound : bounds) {
                    for (int m : new int[] {1, 300, 1999}) {
                        long seed = 43 + m;
                        SampledCoreset expected = countsReference(input, 5, m, seeding, bound,
                                new SplittableRandom(seed));
                        SampledCoreset actual = new SensitivityCoreset(5, m, seeding, bound)
                                .compressWithProvenance(input, new SplittableRandom(seed), MetricsSink.NOOP);
                        assertEquals(expected, actual, seeding + " " + bound + " m=" + m);
                    }
                }
            }
        }
    }

    /**
     * Guard: seeding passes, then exactly one pass for the bounds and only the output rows; per-point scratch is the
     * seeding's 12 bytes plus the 8-byte probability table.
     */
    @Test
    void boundsTakeOnePassAndTwentyBytesPerPoint() {
        int n = 200_000;
        int k = 4;
        List<WeightedPoint> pts = new ArrayList<>(n);
        SplittableRandom rnd = new SplittableRandom(44);
        for (int i = 0; i < n; i++) {
            pts.add(new WeightedPoint(SparseVector.dense(new double[] {rnd.nextGaussian() + 10 * (i % k),
                    rnd.nextGaussian()}), 1.0 + i % 3));
        }
        Coreset input = Coreset.of(pts);
        SensitivityCoreset algo = new SensitivityCoreset(k, 500, Seeding.kMeansPlusPlus(0));
        algo.compress(input, new SplittableRandom(1)); // warm-up: class loading must not count
        RecordingMetrics metrics = new RecordingMetrics(true);
        Coreset c = algo.compress(input, new SplittableRandom(45), metrics);
        MetricsSnapshot snap = metrics.snapshot();
        assertEquals((long) n + c.size(), snap.counters().get(MetricNames.SENSITIVITY_POINTS_SCANNED));
        // Seeding: input read, one pass per center, the final clustering pass.
        assertEquals((long) n * (k + 2), snap.counters().get(MetricNames.SEEDING_POINTS_SCANNED));
        assertTrue(snap.counters().get(MetricNames.SEEDING_DRAW_POINTS_SCANNED) <= (long) n * k);
        double alloc = snap.distributions().get(MetricNames.SENSITIVITY_COMPRESS + MetricNames.ALLOC_BYTES_SUFFIX)
                .max();
        // Before: 60 bytes per point (weights, mass, bounds, four alias arrays, counts). Slack: output and metrics.
        assertTrue(alloc < 22.0 * n, "allocated " + alloc + " bytes, " + alloc / n + " per point");
    }

    /**
     * Averages cost(C, Q) and total weight over R seeded runs; both are unbiased, so the means must lie within 5
     * empirical standard errors of cost(P, Q) and W.
     */
    @Test
    void costAndWeightAreUnbiased() {
        Coreset p = new Coreset(TestData.mixture(400, 5, 3, true, 1).points(), 3.0);
        List<SparseVector> q = TestData.randomCenters(4, 3, 8.0, new SplittableRandom(2));
        double truth = Cost.cost(p, q);
        for (Seeding seeding : List.of(Seeding.kMeansPlusPlus(0), Seeding.kMeansPlusPlus(3), Seeding.bicriteria())) {
            SensitivityCoreset algo = new SensitivityCoreset(5, 40, seeding);
            int runs = 1500;
            double[] cost = new double[runs];
            double[] weight = new double[runs];
            for (int r = 0; r < runs; r++) {
                Coreset c = algo.compress(p, new SplittableRandom(5000 + r));
                assertEquals(p.offset(), c.offset());
                cost[r] = Cost.cost(c, q);
                weight[r] = c.totalWeight();
            }
            assertWithinStandardErrors(truth, cost, 5, seeding + " cost");
            assertWithinStandardErrors(p.totalWeight(), weight, 5, seeding + " weight");
        }
    }

    static void assertWithinStandardErrors(double truth, double[] samples, double z, String what) {
        double mean = 0.0;
        for (double s : samples) {
            mean += s;
        }
        mean /= samples.length;
        double ss = 0.0;
        for (double s : samples) {
            ss += (s - mean) * (s - mean);
        }
        double se = Math.sqrt(ss / (samples.length - 1) / samples.length);
        assertTrue(Math.abs(mean - truth) <= z * se,
                what + ": mean " + mean + " vs " + truth + " (se " + se + ", rel " + (mean / truth - 1) + ")");
    }

    @Test
    void provenanceMatchesWeightsAndTotalSensitivity() {
        Coreset p = TestData.mixture(300, 4, 2, true, 3);
        SensitivityCoreset algo = new SensitivityCoreset(4, 50, Seeding.kMeansPlusPlus(0));
        SampledCoreset s = algo.compressWithProvenance(p, new SplittableRandom(4), MetricsSink.NOOP);
        assertEquals(s.coreset(), algo.compress(p, new SplittableRandom(4)));
        // D^2 centers are input points, so all 4 clusters are non-empty and S = 6 alpha + 4 * 4 (Lemma 2.2), with the
        // default alpha = 2 chosen empirically (see SensitivityBound.DEFAULT_ALPHA).
        assertEquals("standard(alpha=2.0)", algo.bound().toString());
        assertEquals(TheoryBounds.kMeansTotalSensitivity(4, 2.0), s.totalSensitivity(), 1e-9);
        assertEquals(50, s.draws());
        Map<SparseVector, Double> inputWeight = new HashMap<>();
        for (WeightedPoint x : p.points()) {
            inputWeight.put(x.vector(), x.weight());
        }
        int[] counts = s.drawCounts();
        double[] q = s.probabilities();
        int total = 0;
        for (int j = 0; j < counts.length; j++) {
            WeightedPoint x = s.coreset().points().get(j);
            double expected = inputWeight.get(x.vector()) * counts[j] / (50 * q[j]);
            assertEquals(expected, x.weight(), 1e-12 * expected);
            total += counts[j];
        }
        assertEquals(50, total);
    }

    /**
     * Review F2: k = 1, 100 points at 0 and one at 1. When k-means++ seeds at the outlier, B = {1} and cost(B) = 100,
     * yet Q = {0} gives the outlier sensitivity 1 / cost(P, {0}) = 1. Its s_p, read back as q_p * S from a sample
     * that drew it, must not be below that.
     */
    @Test
    void outlierSeedingStillUpperBoundsTheOutliersSensitivity() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            pts.add(point(1.0, 0.0));
        }
        pts.add(point(1.0, 1.0));
        Coreset p = Coreset.of(pts);
        SparseVector outlier = SparseVector.dense(new double[] {1.0});
        double trueSensitivityLowerBound = 1.0 / Cost.cost(p, List.of(SparseVector.dense(new double[] {0.0})));
        // The guarantee needs a theory-valid alpha, so pin it rather than rely on the (empirical) default.
        SensitivityCoreset algo = new SensitivityCoreset(1, 100, Seeding.kMeansPlusPlus(0),
                SensitivityBound.standard(SensitivityBound.kMeansPlusPlusAlpha(1)));
        for (long seed = 0; seed < 100_000; seed++) {
            Clustering b = Seeding.kMeansPlusPlus(0).cluster(pts, 1, new SplittableRandom(seed), MetricsSink.NOOP);
            if (!b.centers().get(0).equals(outlier)) {
                continue;
            }
            // compress draws the same seeding first, then samples from the same generator.
            SampledCoreset s = algo.compressWithProvenance(p, new SplittableRandom(seed), MetricsSink.NOOP);
            int j = s.coreset().points().stream().map(WeightedPoint::vector).toList().indexOf(outlier);
            if (j < 0) {
                continue;
            }
            double sOutlier = s.probabilities()[j] * s.totalSensitivity();
            assertTrue(sOutlier >= trueSensitivityLowerBound,
                    "seed " + seed + ": s_outlier = " + sOutlier + " < " + trueSensitivityLowerBound);
            return;
        }
        throw new AssertionError("no seed drew the outlier both as center and as sample");
    }

    /** The k = 1 counterexample evaluated on the bounds directly, with B = {outlier} and cost(B) = 100. */
    @Test
    void standardBoundCoversTheOutlierAndLegacyDoesNot() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            pts.add(point(1.0, 0.0));
        }
        pts.add(point(1.0, 1.0));
        int[] assignment = new int[101];
        double[] sq = new double[101];
        java.util.Arrays.fill(sq, 0, 100, 1.0);
        Clustering b = new Clustering(List.of(SparseVector.dense(new double[] {1.0})), assignment, sq, 100.0);

        double[] standard = SensitivityBound.standard(SensitivityBound.kMeansPlusPlusAlpha(1)).sensitivities(pts, b);
        double[] legacy = SensitivityBound.legacy().sensitivities(pts, b);
        assertTrue(standard[100] >= 1.0, "standard s_outlier = " + standard[100]);
        assertTrue(legacy[100] < 1.0, "legacy s_outlier = " + legacy[100]);
        double total = 0.0;
        for (double v : standard) {
            total += v;
        }
        assertEquals(TheoryBounds.kMeansTotalSensitivity(1, 32.0), total, 1e-9);
        assertThrows(IllegalArgumentException.class, () -> SensitivityBound.standard(0.5));
        assertThrows(IllegalArgumentException.class, () -> SensitivityBound.kMeansPlusPlusAlpha(0));
    }

    /**
     * s_p must dominate w_p d(p, Q)^2 / cost(P, Q) for every Q when B is within alpha of optimal. Lloyd-refined
     * seedings on well separated data are, so check s_p against many random and data-derived center sets.
     */
    @Test
    void standardBoundDominatesObservedSensitivities() {
        Coreset p = TestData.mixture(400, 3, 2, true, 41);
        List<WeightedPoint> pts = p.points();
        SplittableRandom rnd = new SplittableRandom(42);
        Clustering b = Seeding.kMeansPlusPlus(10).cluster(pts, 3, rnd, MetricsSink.NOOP);
        double[] s = SensitivityBound.standard(SensitivityBound.kMeansPlusPlusAlpha(3)).sensitivities(pts, b);
        for (int t = 0; t < 300; t++) {
            List<SparseVector> q = t % 2 == 0
                    ? TestData.randomCenters(1 + rnd.nextInt(3), 2, 30.0, rnd)
                    : List.of(pts.get(rnd.nextInt(pts.size())).vector());
            double cost = Cost.cost(pts, q);
            for (int i = 0; i < pts.size(); i++) {
                WeightedPoint x = pts.get(i);
                double share = x.weight() * x.vector().squaredDistance(q.get(Cost.nearest(x.vector(), q))) / cost;
                assertTrue(s[i] >= share, "point " + i + ", Q #" + t + ": " + s[i] + " < " + share);
            }
        }
    }

    @Test
    void legacyBoundIsSelectableForComparison() {
        Coreset p = TestData.mixture(300, 4, 2, true, 3);
        SensitivityCoreset legacy = new SensitivityCoreset(4, 50, Seeding.kMeansPlusPlus(0), SensitivityBound.legacy());
        SampledCoreset s = legacy.compressWithProvenance(p, new SplittableRandom(4), MetricsSink.NOOP);
        assertEquals(34.0, s.totalSensitivity(), 1e-9); // 8 * 4 + 2
    }

    /** cost(B) = 0 (every point on a center): only the mass term is used, so S = 4 * clusters and nothing is NaN. */
    @Test
    void zeroSeedingCostUsesMassTermOnly() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 120; i++) {
            pts.add(point(1.0 + i % 4, i % 3, 0));
        }
        SensitivityCoreset algo = new SensitivityCoreset(3, 20, Seeding.kMeansPlusPlus(0));
        SampledCoreset s = algo.compressWithProvenance(Coreset.of(pts), new SplittableRandom(5), MetricsSink.NOOP);
        assertEquals(12.0, s.totalSensitivity(), 1e-12);
        for (WeightedPoint x : s.coreset().points()) {
            assertTrue(Double.isFinite(x.weight()) && x.weight() > 0.0, x.toString());
        }
        // Every point of a cluster gets q proportional to its weight, so a cluster drawn c times carries
        // W(cluster) * 3 * c / 20 in total; summed over clusters that is W * (sum c) / 20 = W when masses are equal.
        assertEquals(Coreset.of(pts).totalWeight(), s.coreset().totalWeight(), 1e-9);
    }

    @Test
    void moreCentersThanDistinctPoints() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 60; i++) {
            pts.add(point(1.0, i % 5, i % 5));
        }
        SampledCoreset s = new SensitivityCoreset(10, 8, Seeding.bicriteria())
                .compressWithProvenance(Coreset.of(pts), new SplittableRandom(6), MetricsSink.NOOP);
        assertEquals(20.0, s.totalSensitivity(), 1e-12); // cost(B) = 0: 4 * 5 distinct-point clusters
        assertTrue(s.coreset().size() <= 8);
    }

    @Test
    void zeroWeightPointsAreNeverSampled() {
        List<WeightedPoint> pts = new ArrayList<>(TestData.mixture(200, 3, 2, false, 7).points());
        SparseVector marker = SparseVector.dense(new double[] {1000, 1000});
        pts.add(new WeightedPoint(marker, 0.0));
        for (Seeding seeding : List.of(Seeding.kMeansPlusPlus(2), Seeding.bicriteria())) {
            Coreset c = new SensitivityCoreset(3, 30, seeding).compress(Coreset.of(pts), new SplittableRandom(8));
            for (WeightedPoint x : c.points()) {
                assertTrue(!x.vector().equals(marker));
            }
        }
    }

    @Test
    void smallInputIsReturnedUnchangedAndExact() {
        Coreset p = new Coreset(List.of(point(2.0, 1, 1), point(3.0, 2, 2)), 1.0);
        SampledCoreset s = new SensitivityCoreset(2, 2, Seeding.kMeansPlusPlus(0))
                .compressWithProvenance(p, new SplittableRandom(9), MetricsSink.NOOP);
        assertSame(p, s.coreset());
        assertTrue(s.exact());
    }

    @Test
    void sameSeedSameCoreset() {
        Coreset p = TestData.mixture(500, 3, 2, true, 10);
        for (Seeding seeding : List.of(Seeding.kMeansPlusPlus(2), Seeding.bicriteria())) {
            SensitivityCoreset algo = new SensitivityCoreset(3, 40, seeding);
            SampledCoreset a = algo.compressWithProvenance(p, new SplittableRandom(11), MetricsSink.NOOP);
            SampledCoreset b = algo.compressWithProvenance(p, new SplittableRandom(11), MetricsSink.NOOP);
            assertEquals(a, b);
            assertArrayEquals(a.drawCounts(), b.drawCounts());
        }
    }

    @Test
    void rejectsInvalidParameters() {
        assertThrows(IllegalArgumentException.class, () -> new SensitivityCoreset(0, 5, Seeding.bicriteria()));
        assertThrows(IllegalArgumentException.class, () -> new SensitivityCoreset(2, 0, Seeding.bicriteria()));
        assertThrows(NullPointerException.class, () -> new SensitivityCoreset(2, 5, null));
    }
}
