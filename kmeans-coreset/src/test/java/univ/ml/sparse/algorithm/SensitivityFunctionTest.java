package univ.ml.sparse.algorithm;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import univ.ml.sparse.SparseWeightableVector;

/**
 * s_p = 8 * w_p / W(cluster of p) + 2 * w_p * d(p, B)^2 / cost(B) sums to exactly 8k + 2 over k non-empty clusters,
 * which is the total the sampling probabilities and importance weights are normalized by.
 */
public class SensitivityFunctionTest {

    // Two weighted 1-d clusters {x, w}: around 0 with W = 3, around 10 with W = 4; cost(B) = 1*1 + 1*4 = 5.
    private static final double[] CENTERS = {0, 10};
    private static final double[][][] CLUSTERS = {
            {{0, 2}, {1, 1}},
            {{10, 3}, {12, 1}},
    };

    @Test
    public void sensitivitiesOfWeightedClustersSumTo8kPlus2() {
        assertEquals(8 * 2 + 2, totalSensitivity(new SensitivityFunction()), 1e-12);
    }

    @Test
    public void sensitivitiesV2OfWeightedClustersSumTo8kPlus2() {
        assertEquals(8 * 2 + 2, totalSensitivity(new SensitivityFunctionV2()), 1e-12);
    }

    private static double totalSensitivity(final SensitivityFunction function) {
        double totalVariance = 0;
        for (int c = 0; c < CLUSTERS.length; c++) {
            for (double[] p : CLUSTERS[c]) {
                totalVariance += weightedSqDist(p, CENTERS[c]);
            }
        }

        double total = 0;
        for (int c = 0; c < CLUSTERS.length; c++) {
            double clusterWeight = 0;
            for (double[] p : CLUSTERS[c]) {
                clusterWeight += p[1];
            }
            for (double[] p : CLUSTERS[c]) {
                final SparseWeightableVector point = new SparseWeightableVector(new double[]{p[0]}, p[1]);
                // Callers pass dist already multiplied by the point weight, as SparseNonUniformCoreset does.
                total += function.sensitivity(point, weightedSqDist(p, CENTERS[c]), clusterWeight, totalVariance);
            }
        }
        return total;
    }

    private static double weightedSqDist(final double[] p, final double center) {
        return p[1] * (p[0] - center) * (p[0] - center);
    }
}
