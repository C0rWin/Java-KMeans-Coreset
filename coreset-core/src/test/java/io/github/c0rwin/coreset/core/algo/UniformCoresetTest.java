package io.github.c0rwin.coreset.core.algo;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
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
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class UniformCoresetTest {

    /** The construction before the equal-weight path and sorted draws: an alias table and an int[n] of counts. */
    private static Coreset countsReference(Coreset input, int m, SplittableRandom rng) {
        PointSet points = input.pointSet();
        int n = points.size();
        double[] weights = new double[n];
        double total = 0.0;
        for (int g = 0; g < n; g++) {
            weights[g] = points.weight(g);
            total += weights[g];
        }
        int[] counts = AliasSampler.drawCounts(weights, m, rng);
        int[] rows = new int[m];
        double[] out = new double[m];
        int distinct = 0;
        for (int i = 0; i < n; i++) {
            if (counts[i] > 0) {
                rows[distinct] = i;
                out[distinct++] = total * counts[i] / m;
            }
        }
        return new Coreset(points.select(rows, out, distinct), input.offset());
    }

    /** Both paths (equal weights replayed without a table, unequal weights in place) are bit-identical to before. */
    @Test
    void outputIsBitIdenticalToTheCountsConstruction() {
        List<WeightedPoint> base = TestData.mixture(3000, 4, 3, false, 21).points();
        List<Coreset> inputs = new ArrayList<>();
        for (double w : new double[] {1.0, 0.3, 7e10}) {
            List<WeightedPoint> equal = new ArrayList<>();
            for (WeightedPoint p : base) {
                equal.add(new WeightedPoint(p.vector(), w));
            }
            inputs.add(new Coreset(equal, 2.5));
        }
        inputs.add(TestData.mixture(3000, 4, 3, true, 22));
        List<WeightedPoint> late = new ArrayList<>(base);
        late.set(2999, new WeightedPoint(base.get(2999).vector(), 4.0)); // first unequal weight at the very end
        late.set(5, new WeightedPoint(base.get(5).vector(), 0.0));
        inputs.add(Coreset.of(late));
        Coreset whole = inputs.get(0);
        PointSet segmented = PointSet.concat(List.of(whole.pointSet().slice(0, 700),
                PointSet.of(PointBlock.of(whole.pointSet().slice(700, 2000).asList())), whole.pointSet().slice(2000, 3000)));
        inputs.add(new Coreset(segmented, 1.0));
        for (Coreset input : inputs) {
            for (int m : new int[] {1, 200, 2999}) {
                for (long seed = 0; seed < 5; seed++) {
                    Coreset expected = countsReference(input, m, new SplittableRandom(seed));
                    Coreset actual = new UniformCoreset(m).compress(input, new SplittableRandom(seed));
                    assertEquals(expected, actual, "m=" + m + " seed=" + seed);
                }
            }
        }
    }

    /** Guard: one pass over the input plus the output rows; no n-sized scratch when the weights are equal. */
    @Test
    void scansTheInputOnceAndAllocatesPerPointOnlyForUnequalWeights() {
        int n = 200_000;
        int m = 500;
        List<WeightedPoint> equal = new ArrayList<>(n);
        List<WeightedPoint> unequal = new ArrayList<>(n);
        SplittableRandom rnd = new SplittableRandom(23);
        for (int i = 0; i < n; i++) {
            SparseVector v = SparseVector.dense(new double[] {rnd.nextDouble(), rnd.nextDouble()});
            equal.add(new WeightedPoint(v, 2.0));
            unequal.add(new WeightedPoint(v, 1.0 + i % 3));
        }
        UniformCoreset algo = new UniformCoreset(m);
        for (List<WeightedPoint> pts : List.of(equal, unequal)) {
            Coreset input = Coreset.of(pts);
            algo.compress(input, new SplittableRandom(1)); // warm-up: class loading must not count
            RecordingMetrics metrics = new RecordingMetrics(true);
            Coreset c = algo.compress(input, new SplittableRandom(24), metrics);
            assertEquals((long) n + c.size(), metrics.snapshot().counters().get(MetricNames.UNIFORM_POINTS_SCANNED));
            double alloc = metrics.snapshot().distributions()
                    .get(MetricNames.UNIFORM_COMPRESS + MetricNames.ALLOC_BYTES_SUFFIX).max();
            // One int[n] alone is 4n bytes; the old path used 32n. The slack covers the output and metric buffers.
            long perPoint = pts == equal ? 0 : 12; // prob + alias tables
            assertTrue(alloc < (perPoint + 2) * n, "allocated " + alloc + " bytes");
        }
    }

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
