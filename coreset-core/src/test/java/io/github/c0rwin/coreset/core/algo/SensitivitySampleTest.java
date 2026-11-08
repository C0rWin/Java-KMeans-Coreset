package io.github.c0rwin.coreset.core.algo;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class SensitivitySampleTest {

    private static final SensitivityCoreset ALGO = new SensitivityCoreset(3, 64, Seeding.kMeansPlusPlus(0));

    @Test
    void drawsMEvenFromAnInputOfAtMostMPoints() {
        Coreset small = TestData.mixture(40, 3, 2, true, 1);
        assertTrue(ALGO.compressWithProvenance(small, new SplittableRandom(1), MetricsSink.NOOP).exact());
        SampledCoreset s = ALGO.sample(small, new SplittableRandom(1), new SplittableRandom(2), MetricsSink.NOOP);
        assertEquals(64, s.draws());
        assertEquals(64, Arrays.stream(s.drawCounts()).sum());
    }

    @Test
    void sharedSeedingGivesOneDistributionAndSamplingStreamsDiffer() {
        Coreset input = TestData.mixture(500, 3, 2, false, 2);
        SampledCoreset a = ALGO.sample(input, new SplittableRandom(9), new SplittableRandom(1), MetricsSink.NOOP);
        SampledCoreset b = ALGO.sample(input, new SplittableRandom(9), new SplittableRandom(2), MetricsSink.NOOP);
        assertEquals(a.totalSensitivity(), b.totalSensitivity(), 0.0); // same bound: same seeding
        assertNotEquals(a.coreset(), b.coreset());
        // a point drawn by both carries the same per-draw weight w / (m q)
        for (int i = 0; i < a.coreset().size(); i++) {
            for (int j = 0; j < b.coreset().size(); j++) {
                if (a.coreset().pointSet().vector(i).equals(b.coreset().pointSet().vector(j))) {
                    assertEquals(a.coreset().pointSet().weight(i) / a.drawCounts()[i],
                            b.coreset().pointSet().weight(j) / b.drawCounts()[j], 1e-9);
                }
            }
        }
    }

    @Test
    void compressWithProvenanceIsUnchangedByTheSplit() {
        Coreset input = TestData.mixture(500, 3, 2, true, 3);
        SampledCoreset viaCompress = ALGO.compressWithProvenance(input, new SplittableRandom(5), MetricsSink.NOOP);
        assertEquals(ALGO.compress(input, new SplittableRandom(5)), viaCompress.coreset());
        SplittableRandom same = new SplittableRandom(5);
        SampledCoreset viaSample = ALGO.sample(input, same, same, MetricsSink.NOOP);
        assertEquals(viaCompress, viaSample);
    }

    @Test
    void anInputWithoutPositiveWeightGivesAnExactEmptyCoreset() {
        Coreset zero = new Coreset(PointBlock.of(TestData.mixture(10, 2, 2, false, 4).points().stream()
                .map(p -> new io.github.c0rwin.coreset.core.WeightedPoint(p.vector(), 0.0)).toList()), 1.5);
        SampledCoreset s = ALGO.sample(zero, new SplittableRandom(1), new SplittableRandom(1), MetricsSink.NOOP);
        assertTrue(s.exact());
        assertEquals(0, s.coreset().size());
        assertEquals(1.5, s.coreset().offset());
        assertArrayEquals(new int[0], s.drawCounts());
    }
}
