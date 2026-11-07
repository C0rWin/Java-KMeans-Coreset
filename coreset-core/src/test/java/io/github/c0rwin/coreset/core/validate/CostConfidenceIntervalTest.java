package io.github.c0rwin.coreset.core.validate;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.algo.SampledCoreset;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.algo.TestData;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class CostConfidenceIntervalTest {

    /**
     * For a Q fixed before sampling, the 95% CLT interval must cover cost(P, Q) in a fraction of R runs consistent with
     * 0.95: within 4 binomial standard deviations, sqrt(0.95 * 0.05 / R).
     */
    @Test
    void normalIntervalCoverageIsNominal() {
        Coreset p = new Coreset(TestData.mixture(2000, 5, 3, true, 1).points(), 4.0);
        List<SparseVector> q = TestData.randomCenters(5, 3, 8.0, new SplittableRandom(2));
        double truth = Cost.cost(p, q);
        SensitivityCoreset algo = new SensitivityCoreset(5, 200, Seeding.kMeansPlusPlus(0));
        int runs = 1000;
        int normalCovered = 0;
        int hoeffdingCovered = 0;
        for (int r = 0; r < runs; r++) {
            SampledCoreset s = algo.compressWithProvenance(p, new SplittableRandom(10_000 + r), MetricsSink.NOOP);
            CostConfidenceInterval.Interval ci = CostConfidenceInterval.normal(s, q, 0.95);
            assertEquals(Cost.cost(s.coreset(), q), ci.estimate(), 1e-9 * truth);
            normalCovered += ci.contains(truth) ? 1 : 0;
            hoeffdingCovered += CostConfidenceInterval.hoeffding(s, q, 0.05).contains(truth) ? 1 : 0;
        }
        double coverage = (double) normalCovered / runs;
        double sd = Math.sqrt(0.95 * 0.05 / runs);
        assertTrue(Math.abs(coverage - 0.95) <= 4 * sd, "coverage " + coverage);
        assertEquals(runs, hoeffdingCovered, "Hoeffding is conservative");
    }

    @Test
    void exactCoresetGivesZeroWidthInterval() {
        Coreset p = new Coreset(List.of(point(2.0, 1, 1), point(3.0, 2, 2)), 1.0);
        SampledCoreset s = new SensitivityCoreset(1, 5, Seeding.bicriteria())
                .compressWithProvenance(p, new SplittableRandom(3), MetricsSink.NOOP);
        List<SparseVector> q = List.of(SparseVector.dense(new double[] {0, 0}));
        double c = Cost.cost(p, q);
        assertEquals(new CostConfidenceInterval.Interval(c, c, c), CostConfidenceInterval.normal(s, q, 0.9));
        assertEquals(new CostConfidenceInterval.Interval(c, c, c), CostConfidenceInterval.hoeffding(s, q, 0.1));
    }

    @Test
    void normalQuantileMatchesKnownValues() {
        assertEquals(1.959963984540054, CostConfidenceInterval.normalQuantile(0.975), 1e-8);
        assertEquals(-2.326347874040841, CostConfidenceInterval.normalQuantile(0.01), 1e-8);
        assertEquals(3.090232306167813, CostConfidenceInterval.normalQuantile(0.999), 1e-8);
        assertEquals(0.0, CostConfidenceInterval.normalQuantile(0.5), 1e-15);
    }

    @Test
    void rejectsInvalidLevels() {
        Coreset p = TestData.mixture(50, 2, 2, false, 4);
        SampledCoreset s = new SensitivityCoreset(2, 10, Seeding.bicriteria())
                .compressWithProvenance(p, new SplittableRandom(5), MetricsSink.NOOP);
        List<SparseVector> q = List.of(SparseVector.dense(new double[] {0, 0}));
        assertThrows(IllegalArgumentException.class, () -> CostConfidenceInterval.normal(s, q, 1.0));
        assertThrows(IllegalArgumentException.class, () -> CostConfidenceInterval.hoeffding(s, q, 0.0));
    }
}
