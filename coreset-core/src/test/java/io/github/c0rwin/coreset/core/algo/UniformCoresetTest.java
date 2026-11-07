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
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class UniformCoresetTest {

    /**
     * Mean of cost(C, Q) over R seeded runs vs cost(P, Q). One run averages m i.i.d. terms T = W f(p) with p ~ w/W,
     * so the run variance is Var(T) / m with Var(T) computed exactly; we allow 5 standard errors.
     */
    @Test
    void costEstimateIsUnbiasedOnWeightedInput() {
        Coreset p = new Coreset(TestData.mixture(400, 4, 3, true, 1).points(), 7.5);
        List<SparseVector> q = TestData.randomCenters(3, 3, 8.0, new SplittableRandom(2));
        int m = 40;
        double w = p.totalWeight();
        double truth = Cost.cost(p, q);
        double sampled = truth - p.offset();
        double second = 0.0;
        for (WeightedPoint x : p.points()) {
            double f = Cost.cost(List.of(new WeightedPoint(x.vector(), 1.0)), q);
            second += (x.weight() / w) * (w * f) * (w * f);
        }
        double varRun = (second - sampled * sampled) / m;

        UniformCoreset algo = new UniformCoreset(m);
        int runs = 4000;
        double sum = 0.0;
        for (int r = 0; r < runs; r++) {
            Coreset c = algo.compress(p, new SplittableRandom(1000 + r));
            assertEquals(w, c.totalWeight(), 1e-9 * w, "weight is conserved exactly");
            assertEquals(p.offset(), c.offset());
            sum += Cost.cost(c, q);
        }
        double tol = 5 * Math.sqrt(varRun / runs);
        assertEquals(truth, sum / runs, tol);
    }

    @Test
    void smallInputIsReturnedUnchanged() {
        Coreset p = new Coreset(List.of(point(2.0, 1, 1), point(3.0, 2, 2)), 1.0);
        assertSame(p, new UniformCoreset(2).compress(p, new SplittableRandom(3)));
    }

    @Test
    void zeroWeightPointsAreNeverSampled() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            pts.add(point(i % 2 == 0 ? 0.0 : 1.0, i, 0));
        }
        Coreset c = new UniformCoreset(50).compress(Coreset.of(pts), new SplittableRandom(4));
        for (WeightedPoint x : c.points()) {
            assertTrue(((int) x.vector().get(0)) % 2 == 1, "zero-weight point sampled: " + x);
        }
        assertEquals(100.0, c.totalWeight(), 1e-9);
    }

    @Test
    void allZeroWeightsGiveEmptyCoresetWithOffset() {
        List<WeightedPoint> pts = List.of(point(0, 1), point(0, 2), point(0, 3));
        Coreset c = new UniformCoreset(1).compress(new Coreset(pts, 4.0), new SplittableRandom(5));
        assertTrue(c.isEmpty());
        assertEquals(4.0, c.offset());
    }

    @Test
    void sameSeedSameCoreset() {
        Coreset p = TestData.mixture(500, 3, 2, true, 6);
        UniformCoreset algo = new UniformCoreset(60);
        assertEquals(algo.compress(p, new SplittableRandom(7)), algo.compress(p, new SplittableRandom(7)));
    }

    @Test
    void rejectsNonPositiveM() {
        assertThrows(IllegalArgumentException.class, () -> new UniformCoreset(0));
    }
}
