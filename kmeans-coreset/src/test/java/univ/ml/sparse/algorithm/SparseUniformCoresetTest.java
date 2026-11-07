package univ.ml.sparse.algorithm;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

import univ.ml.sparse.SparseWeightableVector;

class SparseUniformCoresetTest {

    @Test
    void inputNoLargerThanSampleSizeIsReturnedWithItsWeights() {
        final List<SparseWeightableVector> input = Arrays.asList(
                point(0, 1), point(1, 2), point(2, 3));

        final List<SparseWeightableVector> coreset = new SparseUniformCoreset(5).takeSample(input);

        assertEquals(3, coreset.size());
        for (int i = 0; i < input.size(); i++) {
            assertEquals(input.get(i).getEntry(0), coreset.get(i).getEntry(0), "point " + i + " coordinate");
            assertEquals(input.get(i).getWeight(), coreset.get(i).getWeight(), "point " + i + " weight");
        }
    }

    @Test
    void costEstimateIsUnbiasedOnWeightedInput() {
        // One heavy far point and two light ones; total weight W = 100, cost to the origin = 98 * 10^2 + 1 * 1^2.
        final List<SparseWeightableVector> input = Arrays.asList(
                point(10, 98), point(0, 1), point(1, 1));
        final double trueCost = 9801;
        final int sampleSize = 2;
        final int trials = 20_000;

        double costSum = 0;
        double minTotalWeight = Double.POSITIVE_INFINITY;
        double maxTotalWeight = Double.NEGATIVE_INFINITY;
        for (int trial = 0; trial < trials; trial++) {
            final List<SparseWeightableVector> coreset = new SparseUniformCoreset(sampleSize).takeSample(input);
            double totalWeight = 0;
            for (SparseWeightableVector p : coreset) {
                totalWeight += p.getWeight();
                costSum += p.getWeight() * p.getEntry(0) * p.getEntry(0);
            }
            minTotalWeight = Math.min(minTotalWeight, totalWeight);
            maxTotalWeight = Math.max(maxTotalWeight, totalWeight);
        }

        // Each estimate lies in [0, 2 * (W/2) * 10^2] = [0, 1e4]; Hoeffding over 2e4 independent trials gives
        // P(|mean - 9801| >= 300) <= 2 * exp(-2 * 2e4 * 300^2 / 1e8) < 1e-15.
        assertEquals(trueCost, costSum / trials, 300, "mean coreset cost to the origin");
        assertEquals(100, minTotalWeight, 1e-9, "smallest coreset total weight");
        assertEquals(100, maxTotalWeight, 1e-9, "largest coreset total weight");
    }

    private static SparseWeightableVector point(final double x, final double weight) {
        return new SparseWeightableVector(new double[]{x}, weight);
    }
}
