package io.github.c0rwin.coreset.core.kmeans;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.MetricsSnapshot;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class WeightedLloydTest {

    private static WeightedPoint wp(double w, double... c) {
        return new WeightedPoint(SparseVector.dense(c), w);
    }

    private static SparseVector v(double... c) {
        return SparseVector.dense(c);
    }

    private static List<WeightedPoint> blobs(double[][] means, int perBlob, double sigma, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < perBlob; i++) {
            for (double[] m : means) {
                double[] c = new double[m.length];
                for (int d = 0; d < m.length; d++) {
                    c[d] = m[d] + sigma * gaussian(rnd);
                }
                pts.add(wp(1.0, c));
            }
        }
        return pts;
    }

    private static double gaussian(SplittableRandom rnd) {
        double u = 1.0 - rnd.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rnd.nextDouble());
    }

    @Test
    void recoversWellSeparatedGaussianBlobs() {
        double[][] means = {{0, 0, 0}, {20, 0, 0}, {0, 20, 0}, {0, 0, 20}};
        List<WeightedPoint> pts = blobs(means, 200, 1.0, 11);

        KMeansResult r = WeightedLloyd.cluster(pts, 4, 100, new SplittableRandom(12));

        assertEquals(4, r.centers().size());
        for (double[] m : means) {
            SparseVector truth = v(m);
            double best = Double.POSITIVE_INFINITY;
            for (SparseVector c : r.centers()) {
                best = Math.min(best, c.squaredDistance(truth));
            }
            assertTrue(best < 0.3 * 0.3, "no center near " + truth + ": " + best);
        }
        // Points generated round-robin over blobs: same blob => same cluster.
        int[] a = r.assignment();
        for (int i = means.length; i < a.length; i++) {
            assertEquals(a[i - means.length], a[i]);
        }
        assertEquals(Cost.cost(pts, r.centers()), r.cost(), 1e-9 * r.cost());
        assertTrue(r.iterations() >= 1);
    }

    /** Two unit-variance clusters 6 apart in the plane, translated by {@code base} in both coordinates. */
    private static List<WeightedPoint> twoClusters(double base, int n) {
        SplittableRandom rnd = new SplittableRandom(1);
        List<WeightedPoint> pts = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double off = i % 2 == 0 ? 0.0 : 6.0;
            pts.add(wp(1.0, base + off + gaussian(rnd), base + gaussian(rnd)));
        }
        return pts;
    }

    /** Review F1: k-means is translation invariant, so data far from the origin must cluster like the original. */
    @Test
    void farFromTheOriginClustersLikeTheUntranslatedData() {
        List<WeightedPoint> near = twoClusters(0.0, 4000);
        KMeansResult reference = WeightedLloyd.cluster(near, 2, 20, new SplittableRandom(7));
        for (double base : new double[] {1e8, 1e9}) {
            List<WeightedPoint> far = twoClusters(base, 4000);
            KMeansResult r = WeightedLloyd.cluster(far, 2, 20, new SplittableRandom(7));

            assertArrayEquals(reference.assignment(), r.assignment(), "assignments at base " + base);
            assertEquals(reference.cost(), r.cost(), 1e-6 * reference.cost(), "cost at base " + base);
            assertEquals(Cost.cost(far, r.centers()), r.cost(), 1e-6 * reference.cost(), "exact cost at " + base);
            int[] a = r.assignment();
            for (int i = 0; i < far.size(); i++) {
                assertEquals(Cost.nearest(far.get(i).vector(), r.centers()), a[i], "point " + i + " at " + base);
            }
        }
    }

    /** Review F4: sparse and dense centers must give the same result, bit for bit, wherever both apply. */
    @Test
    void sparseAndDenseCentersGiveIdenticalResults() {
        List<List<WeightedPoint>> inputs = new ArrayList<>();
        inputs.add(blobs(new double[][] {{0, 0, 0}, {20, 0, 0}, {0, 20, 0}, {0, 0, 20}}, 50, 3.0, 31));
        inputs.add(twoClusters(1e9, 400));
        SplittableRandom rnd = new SplittableRandom(32);
        List<WeightedPoint> sparse = new ArrayList<>();
        for (int i = 0; i < 600; i++) {
            int[] idx = {i % 7, 7 + rnd.nextInt(40), 7 + rnd.nextInt(40)};
            double[] val = {1.0 + rnd.nextDouble(), rnd.nextGaussian(), rnd.nextGaussian()};
            sparse.add(new WeightedPoint(SparseVector.of(50, idx, val), i % 11 == 0 ? 0.0 : 0.5 + rnd.nextDouble()));
        }
        inputs.add(sparse);
        for (int t = 0; t < inputs.size(); t++) {
            List<WeightedPoint> pts = inputs.get(t);
            for (int k : new int[] {1, 4, 25}) {
                List<SparseVector> seeds = KMeansPlusPlus.seed(pts, k, new SplittableRandom(33 + k));
                double[] denseDist = new double[pts.size()];
                double[] sparseDist = new double[pts.size()];
                KMeansResult dense =
                        WeightedLloyd.run(block(pts), seeds, 30, MetricsSink.NOOP, Centroids.Layout.DENSE, denseDist);
                KMeansResult sparseCenters =
                        WeightedLloyd.run(block(pts), seeds, 30, MetricsSink.NOOP, Centroids.Layout.SPARSE, sparseDist);
                assertEquals(dense, sparseCenters, "input " + t + ", k=" + k);
                assertArrayEquals(denseDist, sparseDist);
            }
        }
        // A far-away initial center owns nothing and is reseeded, in both layouts alike.
        List<WeightedPoint> pts = List.of(wp(1, 0, 0), wp(1, 1, 0), wp(1, 10, 0), wp(1, 11, 0));
        List<SparseVector> seeds = List.of(v(5, 0), v(1e6, 1e6));
        assertEquals(WeightedLloyd.run(block(pts), seeds, 20, MetricsSink.NOOP, Centroids.Layout.DENSE, null),
                WeightedLloyd.run(block(pts), seeds, 20, MetricsSink.NOOP, Centroids.Layout.SPARSE, null));
    }

    private static PointSet block(List<WeightedPoint> pts) {
        return PointSet.of(PointBlock.of(pts));
    }

    @Test
    void refineReportsTheFinalSquaredDistances() {
        List<WeightedPoint> pts = blobs(new double[][] {{0, 0}, {10, 0}}, 30, 1.0, 34);
        double[] dist = new double[pts.size()];
        KMeansResult r = WeightedLloyd.refine(pts, List.of(v(1, 1), v(9, -1)), 10, MetricsSink.NOOP, dist);
        int[] a = r.assignment();
        for (int i = 0; i < pts.size(); i++) {
            assertEquals(pts.get(i).vector().squaredDistance(r.centers().get(a[i])), dist[i], 1e-12);
        }
        assertThrows(IllegalArgumentException.class,
                () -> WeightedLloyd.refine(pts, List.of(v(0, 0)), 1, MetricsSink.NOOP, new double[1]));
    }

    @Test
    void layoutFollowsCellsVersusStoredEntries() {
        assertEquals(Centroids.Layout.DENSE, Centroids.layoutFor(1000, 1000, 10));
        assertEquals(Centroids.Layout.DENSE, Centroids.layoutFor(20, 1_000_000, 10_000_000));
        assertEquals(Centroids.Layout.SPARSE, Centroids.layoutFor(20, 1_000_000, 1_000_000));
        assertEquals(Centroids.Layout.SPARSE, Centroids.layoutFor(1000, 10_000_000, 25_000));
    }

    @Test
    void pointsScannedAndLayoutAreReported() {
        List<WeightedPoint> pts = blobs(new double[][] {{0, 0}, {50, 50}}, 20, 1.0, 35);
        int n = pts.size();
        List<SparseVector> seeds = List.of(v(1, 1), v(49, 49), v(25, 25));
        RecordingMetrics dense = new RecordingMetrics();
        KMeansResult d = WeightedLloyd.run(block(pts), seeds, 10, dense, Centroids.Layout.DENSE, null);
        long it = d.iterations();
        assertEquals(n * (1 + (it + 1) + it), dense.snapshot().counters().get(MetricNames.LLOYD_POINTS_SCANNED));
        assertEquals(null, dense.snapshot().counters().get(MetricNames.LLOYD_SPARSE_CENTERS));

        RecordingMetrics sparse = new RecordingMetrics();
        WeightedLloyd.run(block(pts), seeds, 10, sparse, Centroids.Layout.SPARSE, null);
        assertEquals(n * (1 + (it + 1) * 3 + it * 3), sparse.snapshot().counters().get(MetricNames.LLOYD_POINTS_SCANNED));
        assertEquals(1L, sparse.snapshot().counters().get(MetricNames.LLOYD_SPARSE_CENTERS));
    }

    /** meansOf against a dense reference: means, part weights, and phi from exact distances. */
    @Test
    void meansOfMatchesDenseReference() {
        List<WeightedPoint> pts = twoClusters(1e9, 300);
        int[] assignment = new int[pts.size()];
        for (int i = 0; i < assignment.length; i++) {
            assignment[i] = i % 3 == 0 ? 2 : i % 2; // part 3 stays empty and is dropped
        }
        var c = WeightedLloyd.meansOf(pts, assignment, 4);
        assertEquals(3, c.size());
        double phi = 0.0;
        for (int j = 0; j < 3; j++) {
            double[] sum = new double[2];
            double w = 0.0;
            for (int i = 0; i < pts.size(); i++) {
                if (assignment[i] == j) {
                    pts.get(i).vector().addTo(sum, pts.get(i).weight());
                    w += pts.get(i).weight();
                }
            }
            double[] mean = {sum[0] / w, sum[1] / w};
            assertEquals(SparseVector.dense(mean), c.points().get(j).vector());
            assertEquals(w, c.points().get(j).weight());
            for (int i = 0; i < pts.size(); i++) {
                if (assignment[i] == j) {
                    phi += pts.get(i).weight() * pts.get(i).vector().squaredDistance(c.points().get(j).vector());
                }
            }
        }
        assertEquals(phi, c.offset(), 1e-9 * phi);
        assertThrows(IllegalArgumentException.class, () -> WeightedLloyd.meansOf(pts, new int[pts.size()], 0));
    }

    @Test
    void weightTwoPointEqualsTwoCopies() {
        SplittableRandom rnd = new SplittableRandom(13);
        List<WeightedPoint> weighted = new ArrayList<>();
        List<WeightedPoint> copies = new ArrayList<>();
        for (int i = 0; i < 30; i++) {
            double[] c = {rnd.nextDouble(), rnd.nextDouble()};
            int w = 1 + rnd.nextInt(3);
            weighted.add(wp(w, c));
            for (int j = 0; j < w; j++) {
                copies.add(wp(1, c));
            }
        }
        KMeansResult a = WeightedLloyd.cluster(weighted, 1, 10, new SplittableRandom(14));
        KMeansResult b = WeightedLloyd.cluster(copies, 1, 10, new SplittableRandom(15));
        assertArrayEquals(a.centers().get(0).toDense(), b.centers().get(0).toDense(), 1e-12);
        assertEquals(b.cost(), a.cost(), 1e-9);

        // Explicit weighted mean: (0,0) x1 and (3,0) x2 -> (2,0).
        KMeansResult m = WeightedLloyd.cluster(List.of(wp(1, 0, 0), wp(2, 3, 0)), 1, 5, new SplittableRandom(16));
        assertEquals(v(2, 0), m.centers().get(0));
        assertEquals(1 * 4 + 2 * 1, m.cost(), 1e-12);
    }

    @Test
    void emptyClusterIsReseededWithoutNaN() {
        // A far-away initial center owns no points and must be moved to the farthest point.
        List<WeightedPoint> pts = List.of(wp(1, 0, 0), wp(1, 1, 0), wp(1, 10, 0), wp(1, 11, 0));
        RecordingMetrics m = new RecordingMetrics();
        KMeansResult r = WeightedLloyd.refine(pts, List.of(v(5, 0), v(1e6, 1e6)), 20, m);

        assertEquals(2, r.centers().size());
        for (SparseVector c : r.centers()) {
            for (int k = 0; k < c.nnz(); k++) {
                assertFalse(Double.isNaN(c.valueAt(k)));
            }
        }
        assertTrue(m.snapshot().counters().get(MetricNames.LLOYD_EMPTY_CLUSTER_RESEEDS) >= 1);
        assertEquals(2 * 0.25 * 2, r.cost(), 1e-9);
    }

    @Test
    void clusterHoldingOnlyZeroWeightPointsCountsAsEmpty() {
        List<WeightedPoint> pts = List.of(wp(1, 0), wp(1, 1), wp(0, 100), wp(1, 50));
        KMeansResult r = WeightedLloyd.refine(pts, List.of(v(0.5), v(100)), 10, MetricsSink.NOOP);
        for (SparseVector c : r.centers()) {
            assertTrue(Double.isFinite(c.squaredNorm()));
        }
        assertEquals(0.5, r.cost(), 1e-9); // centers end at 0.5 and 50
    }

    @Test
    void degenerateInputsTerminate() {
        List<WeightedPoint> same = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            same.add(wp(1, 2, 2));
        }
        KMeansResult r = WeightedLloyd.cluster(same, 3, 50, new SplittableRandom(17));
        assertEquals(List.of(v(2, 2)), r.centers());
        assertEquals(0.0, r.cost());

        KMeansResult zero = WeightedLloyd.cluster(List.of(wp(0, 1), wp(0, 2)), 2, 10, new SplittableRandom(18));
        assertTrue(zero.centers().isEmpty());
        assertArrayEquals(new int[] {-1, -1}, zero.assignment());
        assertEquals(0.0, zero.cost());

        KMeansResult empty = WeightedLloyd.cluster(List.of(), 2, 10, new SplittableRandom(19));
        assertTrue(empty.centers().isEmpty());
        assertEquals(0, empty.assignment().length);
    }

    @Test
    void distanceEvaluationsMatchFormula() {
        List<WeightedPoint> pts = blobs(new double[][] {{0, 0}, {50, 50}, {-50, 50}}, 20, 1.0, 20);
        int n = pts.size();
        int k = 3;
        RecordingMetrics m = new RecordingMetrics();
        KMeansResult r = WeightedLloyd.cluster(pts, k, 30, new SplittableRandom(21), m);
        MetricsSnapshot s = m.snapshot();

        int trials = KMeansPlusPlus.defaultLocalTrials(k); // greedy seeding: see KMeansPlusPlusTest
        assertEquals((long) n + (long) n * (k - 1) * (trials + 1),
                s.counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS)
                        + s.counters().get(MetricNames.KMEANSPP_PRUNED_DISTANCES));
        assertEquals((long) n * k * (r.iterations() + 1), s.counters().get(MetricNames.LLOYD_DISTANCE_EVALUATIONS));
        assertEquals(r.iterations() + 1, s.distributions().get(MetricNames.LLOYD_ITERATION_COST).count());
        assertEquals(r.iterations(), s.distributions().get(MetricNames.LLOYD_ITERATIONS).max());
        assertEquals(r.cost(), s.distributions().get(MetricNames.LLOYD_ITERATION_COST).min(), 1e-9);
        assertEquals(1, s.distributions().get(MetricNames.LLOYD_CLUSTER + MetricNames.WALL_NS_SUFFIX).count());
    }

    @Test
    void maxIterationsBoundsWork() {
        List<WeightedPoint> pts = blobs(new double[][] {{0, 0}, {3, 0}}, 50, 2.0, 22);
        KMeansResult r = WeightedLloyd.cluster(pts, 2, 0, new SplittableRandom(23));
        assertEquals(0, r.iterations());
        assertEquals(Cost.cost(pts, r.centers()), r.cost(), 1e-9 * (1 + r.cost()));
        assertThrows(IllegalArgumentException.class, () -> WeightedLloyd.cluster(pts, 2, -1, new SplittableRandom(1)));
    }

    @Test
    void assignmentAccessorReturnsCopy() {
        KMeansResult r = WeightedLloyd.cluster(List.of(wp(1, 0), wp(1, 5)), 2, 5, new SplittableRandom(24));
        int[] a = r.assignment();
        a[0] = 99;
        assertTrue(r.assignment()[0] != 99);
    }

    @Test
    void bestOfNeverLosesToAnyRestartWithTheSameDerivedSeeds() {
        double[][] means = new double[12][];
        for (int i = 0; i < means.length; i++) {
            means[i] = new double[] {10.0 * (i % 4), 10.0 * (i / 4)};
        }
        // Overlapping blobs: greedy seeding solves well-separated ones in every restart.
        List<WeightedPoint> pts = blobs(means, 30, 6.0, 31);
        int restarts = 8;
        RecordingMetrics metrics = new RecordingMetrics();
        KMeansResult best = WeightedLloyd.bestOf(pts, 12, 30, restarts, new SplittableRandom(32), metrics);

        SplittableRandom derive = new SplittableRandom(32);
        double min = Double.POSITIVE_INFINITY;
        double max = 0.0;
        for (int i = 0; i < restarts; i++) {
            KMeansResult single = WeightedLloyd.cluster(pts, 12, 30, new SplittableRandom(derive.nextLong()));
            assertTrue(best.cost() <= single.cost(), "restart " + i + ": " + single.cost() + " < " + best.cost());
            min = Math.min(min, single.cost());
            max = Math.max(max, single.cost());
        }
        assertEquals(min, best.cost());
        assertTrue(max > min, "fixture should have restarts landing in different local optima");
        assertEquals(Cost.cost(pts, best.centers()), best.cost(), 1e-9 * best.cost());
        MetricsSnapshot snap = metrics.snapshot();
        assertEquals(restarts, snap.distributions().get(MetricNames.LLOYD_RESTART_COST).count());
        assertEquals(1, snap.distributions().get(MetricNames.LLOYD_BEST_OF + MetricNames.WALL_NS_SUFFIX).count());
        assertThrows(IllegalArgumentException.class,
                () -> WeightedLloyd.bestOf(pts, 2, 5, 0, new SplittableRandom(1), MetricsSink.NOOP));
    }
}
