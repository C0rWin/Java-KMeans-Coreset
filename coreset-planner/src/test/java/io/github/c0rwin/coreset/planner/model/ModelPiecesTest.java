package io.github.c0rwin.coreset.planner.model;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ModelPiecesTest {

    @Test
    void bundledCalibrationCarriesVersionProvenanceAndTheFaultLabConstants() {
        Calibration c = Calibration.bundled();
        assertEquals("fault-lab-2026-11-07-v1", c.version());
        assertTrue(c.provenance().get("source").contains("82fe39d"));
        assertEquals(0.022, c.get("mergeReduceCoefficient"));
        assertEquals(1.03, c.get("kappaLawHigh"));
    }

    @Test
    void normalQuantileMatchesTables() {
        assertEquals(1.959964, Probability.normalQuantile(0.975), 1e-6);
        assertEquals(2.575829, Probability.normalQuantile(0.995), 1e-6);
        assertEquals(-1.644854, Probability.normalQuantile(0.05), 1e-6);
    }

    @Test
    void binomialQuantileIsTheExactUpperTail() {
        // Bin(10, 0.5): P[X > 8] = 11/1024 > 0.01, P[X > 9] = 1/1024
        assertEquals(9, Probability.binomialUpperQuantile(10, 0.5, 0.01));
        assertEquals(0, Probability.binomialUpperQuantile(16, 0.0, 0.01));
    }

    @Test
    void memorylessBlockFailureIsTheBinomialTail() {
        double rho = 0.1;
        double expected = 0.0;
        // P[Bin(6, 0.1) > 2]
        for (int l = 3; l <= 6; l++) {
            expected += binom(6, l) * Math.pow(rho, l) * Math.pow(1 - rho, 6 - l);
        }
        assertEquals(expected, Probability.blockFailure(4, 2, rho, 1 - rho), 1e-12);
    }

    @Test
    void burstsRaiseBlockFailureAtTheSameLossRate() {
        double memoryless = Probability.blockFailure(16, 4, 0.1, 0.9);
        double bursty = Probability.blockFailure(16, 4, 0.1 / 0.9 / 8.0, 1.0 / 8.0);
        assertTrue(bursty > 3 * memoryless, bursty + " vs " + memoryless);
    }

    @Test
    void chernoffQuantileOfDeliveriesMatchesCorLossEps() {
        long units = 10_000;
        double rho = 0.2;
        double eta = 0.01;
        double mu = (1 - rho) * units;
        long expected = (long) Math.ceil(mu - Math.sqrt(2 * mu * Math.log(1 / eta)));
        assertEquals(expected, Probability.deliveredLowerBound(units, rho, 0.0, eta));
    }

    @Test
    void splitExcessInterpolatesBetweenPerDrawAndMerged() {
        double lambda = 20.0;
        double merged = lambda * lambda; // E[K^2] - E[K] for Poisson
        double split = Probability.splitExcess(lambda, 4);
        assertTrue(split > 0 && split < merged, split + " vs " + merged);
        assertEquals(merged, Probability.splitExcess(lambda, 1_000_000), 1e-6 * merged);
    }

    private static double binom(int n, int k) {
        double r = 1;
        for (int i = 1; i <= k; i++) {
            r = r * (n - k + i) / i;
        }
        return r;
    }
}
