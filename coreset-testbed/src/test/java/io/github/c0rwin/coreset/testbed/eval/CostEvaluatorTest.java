package io.github.c0rwin.coreset.testbed.eval;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.testbed.data.SyntheticData;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ForkJoinPool;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Test;

class CostEvaluatorTest {

    private static final ForkJoinPool POOL = new ForkJoinPool(4);

    @AfterAll
    static void shutdown() {
        POOL.shutdownNow();
    }

    private static List<List<SparseVector>> seeds(List<WeightedPoint> pts, int k, int count, long seed) {
        SplittableRandom rng = new SplittableRandom(seed);
        List<List<SparseVector>> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(KMeansPlusPlus.seed(pts, k, rng));
        }
        return out;
    }

    @Test
    void kMeansCostsMatchCoreCostAndDistortionOnSparseAndDenseData() {
        List<WeightedPoint> dense = SyntheticData.gaussianMixture(9000, 6, 4, 2.0, 1.0, 50.0, 1);
        List<WeightedPoint> sparse = SyntheticData.sparseBagOfWords(9000, 3000, 5, 200, 20, 0.8, 0.5, 2);
        for (List<WeightedPoint> pts : List.of(dense, sparse)) {
            Coreset full = Coreset.of(pts);
            Coreset sample = new Coreset(pts.subList(0, 700).stream()
                    .map(p -> new WeightedPoint(p.vector(), p.weight() * 9000 / 700.0)).toList(), 3.5);
            List<List<SparseVector>> qs = seeds(pts, 7, 3, 5);
            List<Centers> cs = qs.stream().map(Centers::of).toList();
            CostEvaluator ev = new CostEvaluator(Problem.kMeans(), POOL);
            double[] onFull = ev.costs(full, cs);
            double[] onSample = ev.costs(sample, cs);
            List<Candidate> cands = new ArrayList<>();
            for (int j = 0; j < qs.size(); j++) {
                assertEquals(Cost.cost(full, qs.get(j)), onFull[j], 1e-9 * onFull[j]);
                assertEquals(Cost.cost(sample, qs.get(j)), onSample[j], 1e-9 * onSample[j]);
                cands.add(new Candidate("q" + j, qs.get(j)));
            }
            Distortion.Report core = Distortion.evaluate(full, sample, cands);
            for (int j = 0; j < qs.size(); j++) {
                assertEquals(core.entries().get(j).ratio(), onSample[j] / onFull[j], 1e-9);
            }
        }
    }

    @Test
    void parallelSegmentedPassIsBitIdenticalToSequentialAndSumsToTheWhole() {
        List<WeightedPoint> pts = SyntheticData.gaussianMixture(3 * CostEvaluator.CHUNK + 17, 5, 3, 1.0, 0.0, 20.0, 3);
        List<Centers> cs = seeds(pts, 4, 2, 7).stream().map(Centers::of).toList();
        int n = pts.size();
        int[] ends = {5, 5, CostEvaluator.CHUNK + 100, n};
        CostEvaluator par = new CostEvaluator(Problem.kMeans(), POOL);
        CostEvaluator seq = new CostEvaluator(Problem.kMeans(), null);
        double[][] a = par.segmentCosts(pts, ends, cs);
        double[][] b = seq.segmentCosts(pts, ends, cs);
        for (int s = 0; s < ends.length; s++) {
            assertArrayEquals(b[s], a[s], 0.0);
        }
        assertEquals(0.0, a[1][0]);
        double[] whole = par.costs(Coreset.of(pts), cs);
        for (int j = 0; j < cs.size(); j++) {
            assertEquals(whole[j], a[0][j] + a[1][j] + a[2][j] + a[3][j], 1e-9 * whole[j]);
        }
        assertEquals(2L * n, par.pointsScanned());
        assertEquals(2L * n * 8, par.distanceEvaluations());
        assertThrows(IllegalArgumentException.class, () -> par.segmentCosts(pts, new int[] {n - 1}, cs));
        assertThrows(IllegalArgumentException.class, () -> par.segmentCosts(pts, new int[] {9, 3, n}, cs));
    }

    @Test
    void kMedianUsesUnsquaredDistancesAndRejectsSquaredOffsets() {
        List<WeightedPoint> pts = List.of(new WeightedPoint(SparseVector.dense(new double[] {3, 4}), 2.0),
                new WeightedPoint(SparseVector.dense(new double[] {0, 1}), 1.0));
        Centers origin = Centers.of(List.of(SparseVector.dense(new double[] {0, 0})));
        CostEvaluator ev = new CostEvaluator(Problem.kMedian(), null);
        assertEquals(2 * 5.0 + 1.0, ev.cost(Coreset.of(pts), origin), 1e-12);
        assertEquals(2 * 25.0 + 1.0, new CostEvaluator(Problem.kMeans(), null).cost(Coreset.of(pts), origin), 1e-12);
        assertThrows(IllegalArgumentException.class, () -> ev.cost(new Coreset(pts, 1.0), origin));
    }
}
