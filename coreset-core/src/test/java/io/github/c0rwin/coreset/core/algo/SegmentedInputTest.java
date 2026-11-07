package io.github.c0rwin.coreset.core.algo;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** Every algorithm gives bit-identical results on a union of segments and on the same rows in one block. */
class SegmentedInputTest {

    @Test
    void denseInputIsSegmentationInvariant() {
        List<WeightedPoint> pts = TestData.mixture(TestData.mixtureCenters(5, 6, 10.0, new SplittableRandom(1)), 900,
                true, new SplittableRandom(2));
        check(pts);
    }

    /** High dimension: scattered seeding distances and sparse Lloyd centers. */
    @Test
    void sparseInputIsSegmentationInvariant() {
        SplittableRandom rnd = new SplittableRandom(3);
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 700; i++) {
            int[] idx = {i % 9, 9 + rnd.nextInt(200_000), 9 + rnd.nextInt(200_000)};
            double[] val = {1.0 + rnd.nextDouble(), rnd.nextGaussian(), rnd.nextGaussian()};
            pts.add(new WeightedPoint(SparseVector.of(200_009, idx, val), i % 13 == 0 ? 0.0 : 0.5 + rnd.nextDouble()));
        }
        check(pts);
    }

    private static void check(List<WeightedPoint> pts) {
        Coreset whole = Coreset.of(pts);
        int a = pts.size() / 5;
        int b = pts.size() / 2;
        // A copied segment and two zero-copy slices of the whole block.
        Coreset split = Coreset.union(List.of(new Coreset(pts.subList(0, a), 0.25),
                Coreset.of(whole.pointSet().slice(a, b)), new Coreset(whole.pointSet().slice(b, pts.size()), 0.5)));
        Coreset shifted = new Coreset(whole.pointSet(), 0.75);
        PointSet p = split.pointSet();
        assertEquals(3, p.segmentCount());
        assertEquals(whole.pointSet(), p);
        assertEquals(shifted, split);

        assertEquals(KMeansPlusPlus.seed(whole.pointSet(), 12, new SplittableRandom(4), MetricsSink.NOOP),
                KMeansPlusPlus.seed(p, 12, new SplittableRandom(4), MetricsSink.NOOP));
        for (Seeding s : List.of(Seeding.kMeansPlusPlus(0), Seeding.kMeansPlusPlus(3), Seeding.bicriteria())) {
            Clustering x = s.cluster(whole.pointSet(), 6, new SplittableRandom(5), MetricsSink.NOOP);
            Clustering y = s.cluster(p, 6, new SplittableRandom(5), MetricsSink.NOOP);
            assertEquals(x.centers(), y.centers(), s.toString());
            assertArrayEquals(x.assignment, y.assignment, s.toString());
            assertArrayEquals(x.squaredDistances, y.squaredDistances, s.toString());
            assertEquals(x.cost(), y.cost(), 0.0, s.toString());
        }
        for (int k : new int[] {1, 7, 30}) {
            KMeansResult x = WeightedLloyd.cluster(whole.pointSet(), k, 15, new SplittableRandom(6), MetricsSink.NOOP);
            KMeansResult y = WeightedLloyd.cluster(p, k, 15, new SplittableRandom(6), MetricsSink.NOOP);
            assertEquals(x, y, "k=" + k);
            assertEquals(WeightedLloyd.meansOf(whole.pointSet(), x.assignment(), k),
                    WeightedLloyd.meansOf(p, y.assignment(), k));
            assertEquals(Cost.cost(shifted, x.centers()), Cost.cost(split, x.centers()), 0.0);
        }
        List<CoresetAlgorithm> algorithms = List.of(new SensitivityCoreset(5, 100, Seeding.kMeansPlusPlus(2)),
                new SensitivityCoreset(5, 100, Seeding.bicriteria()), new UniformCoreset(100),
                new MeanCoreset(40, 5), MeanCoreset.adaptive(3, 0.3, 64, 5));
        for (CoresetAlgorithm alg : algorithms) {
            assertEquals(alg.compress(shifted, new SplittableRandom(7), MetricsSink.NOOP),
                    alg.compress(split, new SplittableRandom(7), MetricsSink.NOOP), alg.toString());
        }
    }
}
