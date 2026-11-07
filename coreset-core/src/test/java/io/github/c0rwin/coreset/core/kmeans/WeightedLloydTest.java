package io.github.c0rwin.coreset.core.kmeans;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Cost;
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

        assertEquals((long) n * (k - 1), s.counters().get(MetricNames.KMEANSPP_DISTANCE_EVALUATIONS));
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
}
