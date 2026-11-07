package io.github.c0rwin.coreset.core.algo;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class MeanCoresetTest {

    private static final double TOL = 1e-10;

    @Test
    void preservesWeightAndFirstAndSecondMomentExactly() {
        Coreset p = new Coreset(TestData.mixture(800, 5, 4, true, 1).points(), 2.5);
        Coreset c = new MeanCoreset(25, 10).compress(p, new SplittableRandom(2));
        assertTrue(c.size() <= 25 && c.size() > 5);
        assertEquals(p.totalWeight(), c.totalWeight(), TOL * p.totalWeight());
        double[] firstP = new double[4];
        double[] firstC = new double[4];
        for (WeightedPoint x : p.points()) {
            x.vector().addTo(firstP, x.weight());
        }
        for (WeightedPoint x : c.points()) {
            x.vector().addTo(firstC, x.weight());
        }
        for (int d = 0; d < 4; d++) {
            assertEquals(firstP[d], firstC[d], TOL * p.totalWeight() * 20);
        }
        // Second moment is cost w.r.t. Q = {0}: offset + sum w ||p||^2 (Lemma 1 at x = 0).
        List<SparseVector> origin = List.of(SparseVector.dense(new double[4]));
        double second = Cost.cost(p, origin);
        assertEquals(second, Cost.cost(c, origin), TOL * second);
    }

    /** Lemma 1: cost(C, Q) = sum_i min_q cost(P_i, q) >= cost(P, Q) for every Q, not only good ones. */
    @Test
    void neverUnderestimatesAnyCenterSet() {
        Coreset p = TestData.mixture(600, 4, 3, true, 3);
        Coreset c = new MeanCoreset(20, 10).compress(p, new SplittableRandom(4));
        SplittableRandom rnd = new SplittableRandom(5);
        for (int t = 0; t < 500; t++) {
            int k = 1 + rnd.nextInt(8);
            double scale = t % 2 == 0 ? 12.0 : 0.5 + 50.0 * rnd.nextDouble();
            List<SparseVector> q = TestData.randomCenters(k, 3, scale, rnd);
            double full = Cost.cost(p, q);
            assertTrue(Cost.cost(c, q) >= full * (1 - 1e-12), "Q #" + t + " underestimated");
        }
        // Centers of the coreset itself make every mean cost 0: what remains is exactly phi >= cost(P, Q) > 0.
        List<SparseVector> own = new ArrayList<>();
        for (WeightedPoint x : c.points()) {
            own.add(x.vector());
        }
        assertTrue(c.offset() > 0.0);
        assertEquals(c.offset(), Cost.cost(c, own), TOL * c.offset());
        assertTrue(c.offset() >= Cost.cost(p, own) * (1 - 1e-12));
    }

    @Test
    void offsetAccumulatesInputOffset() {
        Coreset p = TestData.mixture(300, 3, 2, false, 6);
        Coreset shifted = new Coreset(p.points(), 11.0);
        MeanCoreset algo = new MeanCoreset(9, 5);
        Coreset a = algo.compress(p, new SplittableRandom(7));
        Coreset b = algo.compress(shifted, new SplittableRandom(7));
        assertEquals(a.points(), b.points());
        assertEquals(a.offset() + 11.0, b.offset(), TOL * b.offset());
    }

    @Test
    void duplicatesCollapseExactly() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            pts.add(point(1.0 + i % 3, i % 4, -(i % 4)));
        }
        Coreset c = new MeanCoreset(10, 5).compress(Coreset.of(pts), new SplittableRandom(8));
        assertEquals(4, c.size());
        assertEquals(0.0, c.offset(), 1e-12);
        List<SparseVector> q = TestData.randomCenters(2, 2, 5.0, new SplittableRandom(9));
        assertEquals(Cost.cost(Coreset.of(pts), q), Cost.cost(c, q), 1e-9);
    }

    @Test
    void smallInputIsReturnedUnchanged() {
        Coreset p = new Coreset(List.of(point(2.0, 1, 1), point(3.0, 2, 2)), 1.0);
        assertSame(p, new MeanCoreset(2, 5).compress(p, new SplittableRandom(10)));
    }

    @Test
    void zeroWeightPointsDoNotAffectTheCoreset() {
        List<WeightedPoint> positive = TestData.mixture(100, 2, 2, false, 11).points();
        List<WeightedPoint> pts = new ArrayList<>(positive);
        pts.add(point(0.0, 500, 500));
        MeanCoreset algo = new MeanCoreset(4, 5);
        Coreset c = algo.compress(Coreset.of(pts), new SplittableRandom(12));
        assertEquals(algo.compress(Coreset.of(positive), new SplittableRandom(12)), c);
        assertEquals(100.0, c.totalWeight(), 1e-9);
    }

    @Test
    void adaptiveChoosesPowerOfKWithinMaxMAndStaysOneSided() {
        Coreset p = TestData.mixture(1000, 3, 2, false, 13);
        RecordingMetrics metrics = new RecordingMetrics();
        Coreset c = MeanCoreset.adaptive(3, 0.5, 50, 10).compress(p, new SplittableRandom(14), metrics);
        double m = metrics.snapshot().distributions().get(MetricNames.MEAN_ADAPTIVE_M).max();
        assertTrue(m == 1 || m == 3 || m == 9 || m == 27 || m == 50, "m=" + m);
        assertTrue(c.size() <= m);
        SplittableRandom rnd = new SplittableRandom(15);
        for (int t = 0; t < 200; t++) {
            List<SparseVector> q = TestData.randomCenters(1 + rnd.nextInt(5), 2, 15.0, rnd);
            assertTrue(Cost.cost(c, q) >= Cost.cost(p, q) * (1 - 1e-12));
        }
        Coreset one = MeanCoreset.adaptive(1, 0.1, 50, 10).compress(p, new SplittableRandom(16));
        assertEquals(1, one.size());
        Coreset capped = MeanCoreset.adaptive(3, 0.01, 5, 10).compress(p, new SplittableRandom(17));
        assertTrue(capped.size() <= 5);
    }

    @Test
    void rejectsInvalidParameters() {
        assertThrows(IllegalArgumentException.class, () -> new MeanCoreset(0, 5));
        assertThrows(IllegalArgumentException.class, () -> new MeanCoreset(3, -1));
        assertThrows(IllegalArgumentException.class, () -> MeanCoreset.adaptive(0, 0.1, 5, 5));
        assertThrows(IllegalArgumentException.class, () -> MeanCoreset.adaptive(2, 0.0, 5, 5));
        assertThrows(IllegalArgumentException.class, () -> MeanCoreset.adaptive(2, 1.0, 5, 5));
        assertThrows(IllegalArgumentException.class, () -> MeanCoreset.adaptive(2, 0.1, 0, 5));
    }
}
