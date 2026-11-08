package io.github.c0rwin.coreset.examples.lightweight;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class LightweightCoresetTest {

    /** One compression of 20,000 blobs to m = 1000 (measured: 1.083). */
    static final double MAX_OFFLINE_DISTORTION = 1.15;
    /** Any merge-reduce or distributed run in this module (measured: 1.04 gRPC to 1.17 for the 64-block tree). */
    static final double MAX_STREAM_DISTORTION = 1.25;
    /** Each draw's weight estimate lies in [0, 2W], so one compression's relative standard error is at most
     *  1 / sqrt(m), 5% at m = 400 (measured: 0.4% to 4.2%). */
    static final double MAX_WEIGHT_REL_ERR = 0.10;

    private static final List<WeightedPoint> POINTS = Blobs.points(20_000, 5);
    private static final Coreset FULL = Coreset.of(POINTS);

    private static CoresetAlgorithm lightweight(int m) {
        return CoresetAlgorithms.load().create("lightweight", Map.of("m", Integer.toString(m)));
    }

    @Test
    void theServiceFileAloneRegistersTheAlgorithm() {
        CoresetAlgorithms registry = CoresetAlgorithms.load();
        assertEquals(List.of("lightweight", "mean", "sensitivity", "uniform"), List.copyOf(registry.ids()));
        CoresetAlgorithm a = registry.create("lightweight", Map.of("m", "64"));
        assertInstanceOf(LightweightCoreset.class, a);
        assertEquals("lightweight(m=64)", a.toString());
    }

    @Test
    void invalidParametersAreRejectedByTheProvider() {
        CoresetAlgorithms registry = CoresetAlgorithms.load();
        assertThrows(IllegalArgumentException.class, () -> registry.create("lightweight", Map.of()));
        assertThrows(IllegalArgumentException.class, () -> registry.create("lightweight", Map.of("m", "4", "k", "2")));
        assertThrows(IllegalArgumentException.class, () -> registry.create("lightweight", Map.of("m", "four")));
        assertThrows(IllegalArgumentException.class, () -> registry.create("lightweight", Map.of("m", "0")));
        assertThrows(IllegalArgumentException.class, () -> registry.create("lightweight", Map.of("m", "1000001")));
    }

    @Test
    void totalWeightAndCostAreUnbiased() {
        Coreset small = Coreset.of(POINTS.subList(0, 2_000));
        double w = small.totalWeight();
        List<SparseVector> centers = List.of(POINTS.get(0).vector(), POINTS.get(1).vector(), POINTS.get(2).vector());
        double cost = Cost.cost(small, centers);
        int runs = 400;
        double sumW = 0;
        double sumW2 = 0;
        double sumC = 0;
        double sumC2 = 0;
        CoresetAlgorithm a = lightweight(100);
        for (int r = 0; r < runs; r++) {
            Coreset c = a.compress(small, new SplittableRandom(1_000 + r));
            double cw = c.totalWeight();
            double cc = Cost.cost(c, centers);
            sumW += cw;
            sumW2 += cw * cw;
            sumC += cc;
            sumC2 += cc * cc;
        }
        double meanW = sumW / runs;
        double seW = Math.sqrt((sumW2 / runs - meanW * meanW) / runs);
        double meanC = sumC / runs;
        double seC = Math.sqrt((sumC2 / runs - meanC * meanC) / runs);
        System.out.printf("lightweight unbiasedness over %d runs: weight %.2f vs %.2f (se %.2f), cost %.1f vs %.1f"
                + " (se %.1f)%n", runs, meanW, w, seW, meanC, cost, seC);
        // Four standard errors: a false alarm about once in 15,000 seeds; fixed seeds make it deterministic anyway.
        assertTrue(Math.abs(meanW - w) < 4 * seW, "weight " + meanW + " vs " + w + " (se " + seW + ")");
        assertTrue(Math.abs(meanC - cost) < 4 * seC, "cost " + meanC + " vs " + cost + " (se " + seC + ")");
        assertTrue(seW < 0.02 * w && seC < 0.02 * cost, "estimates too noisy to test");
    }

    @Test
    void oneCompressionHasBoundedDistortion() {
        Coreset c = lightweight(1_000).compress(FULL, new SplittableRandom(3));
        double distortion = Blobs.distortion(FULL, c);
        System.out.printf("lightweight offline: %d points, distortion %.4f%n", c.size(), distortion);
        assertTrue(c.size() <= 1_000);
        assertTrue(distortion < MAX_OFFLINE_DISTORTION, "distortion " + distortion);
    }

    @Test
    void theInProcessMergeReduceTreeRunsItUnchanged() {
        CoresetAlgorithm a = lightweight(1_000);
        MergeReduceTree tree = new MergeReduceTree(a, new SplittableRandom(11), MetricsSink.NOOP);
        int blocks = 64;
        for (int b = 0; b < blocks; b++) {
            int from = POINTS.size() * b / blocks;
            int to = POINTS.size() * (b + 1) / blocks;
            tree.add(Coreset.of(POINTS.subList(from, to))); // blocks of 312 points: below m, kept exactly
        }
        Coreset c = tree.compressedSnapshot();
        double distortion = Blobs.distortion(FULL, c);
        double weightErr = Math.abs(c.totalWeight() - FULL.totalWeight()) / FULL.totalWeight();
        System.out.printf("lightweight stream: %d levels, %d points, distortion %.4f, weight rel err %.2e%n",
                tree.levels(), c.size(), distortion, weightErr);
        assertTrue(c.size() <= 1_000);
        assertTrue(distortion < MAX_STREAM_DISTORTION, "distortion " + distortion);
        assertTrue(weightErr < MAX_WEIGHT_REL_ERR, "weight rel err " + weightErr);
    }

    @Test
    void edgeCasesKeepTheEstimatorExact() {
        CoresetAlgorithm a = new LightweightCoreset(4);
        Coreset tiny = new Coreset(POINTS.subList(0, 4), 2.5);
        assertSame(tiny, a.compress(tiny, new SplittableRandom(1)));

        List<WeightedPoint> same = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            same.add(new WeightedPoint(SparseVector.dense(new double[] {1, 2}), 1 + i % 3));
        }
        Coreset equal = new Coreset(same, 7.0);
        Coreset c = a.compress(equal, new SplittableRandom(2));
        assertEquals(7.0, c.offset());
        assertTrue(c.size() <= 4);
        // Every point at the mean: q is proportional to the weight, so each draw carries exactly W / m.
        assertEquals(equal.totalWeight(), c.totalWeight(), 1e-9 * equal.totalWeight());

        List<WeightedPoint> zero = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            zero.add(new WeightedPoint(SparseVector.dense(new double[] {i}), 0));
        }
        Coreset none = a.compress(new Coreset(zero, 1.5), new SplittableRandom(3));
        assertTrue(none.isEmpty());
        assertEquals(1.5, none.offset());
        assertThrows(IllegalArgumentException.class, () -> new LightweightCoreset(0));
    }

    @Test
    void theFactoryBuildsTheSameAlgorithm() {
        LightweightCoresetFactory f = new LightweightCoresetFactory(1_000);
        assertEquals(lightweight(1_000).compress(FULL, new SplittableRandom(4)),
                f.create().compress(FULL, new SplittableRandom(4)));
        assertEquals("lightweight-factory(m=1000)", f.identity());
        assertThrows(IllegalArgumentException.class, () -> new LightweightCoresetFactory(0));
    }
}
