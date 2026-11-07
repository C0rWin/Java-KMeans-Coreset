package univ.ml.sparse.algorithm;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.junit.Test;

import univ.ml.sparse.RandomSampleAlgorithm;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

/**
 * Checks importance weights against u_p = w_p * c_p / (m * q_p), q_p = s_p / sum(s), c_p = draws of p, m = draws.
 * Seeding and sampling are fixed test doubles, so every expected weight is exact.
 */
public class SparseNonUniformCoresetTest {

    private static final double EPS = 1e-12;

    // Three points on a center at 0 and one outlier at 10, all of unit weight: W = 4, cost = 100,
    // s = 8 * 1/4 (+ 2 * 100/100 for the outlier) = {2, 2, 2, 4}, sum(s) = 10.
    private static final double[] X = {0, 0, 0, 10};
    private static final double[] UNIT = {1, 1, 1, 1};
    private static final double[] Q_UNIT = {.2, .2, .2, .4};

    // Same positions with weights {2, 1, 1, 4}: W = 8, cost = 400, s = {2, 1, 1, 4 + 2}, sum(s) = 10.
    private static final double[] WEIGHTED = {2, 1, 1, 4};
    private static final double[] Q_WEIGHTED = {.2, .1, .1, .6};

    private static final SparseSeedingAlgorithm ONE_CLUSTER_AT_0 = seeding(new double[]{0}, new int[][]{{0, 1, 2, 3}});

    @Test
    public void outlierDrawnTwiceCountsTwice() {
        final CountingSampler sampler = new CountingSampler(3, 0, 3);
        final List<SparseWeightableVector> coreset =
                new SparseNonUniformCoreset(ONE_CLUSTER_AT_0, 3, new SensitivityFunction(), sampler)
                        .takeSample(points(X, UNIT));

        final Map<Integer, Double> u = weightsByIndex(coreset, sampler.dataset);
        assertEquals(2, u.size());
        assertEquals(1 * 1 / (3 * Q_UNIT[0]), u.get(0), EPS);
        assertEquals(1 * 2 / (3 * Q_UNIT[3]), u.get(3), EPS);
    }

    @Test
    public void emptyClusterDoesNotChangeTheNormalization() {
        // Seeding returned the same center twice: the second cluster is empty and contributes nothing to sum(s).
        final SparseSeedingAlgorithm duplicateCenters = seeding(new double[]{0, 0}, new int[][]{{0, 1, 2, 3}, {}});
        final CountingSampler sampler = new CountingSampler(0, 1, 3);
        final List<SparseWeightableVector> coreset =
                new SparseNonUniformCoreset(duplicateCenters, 3, new SensitivityFunction(), sampler)
                        .takeSample(points(X, UNIT));

        final Map<Integer, Double> u = weightsByIndex(coreset, sampler.dataset);
        assertEquals(1 / (3 * Q_UNIT[0]), u.get(0), EPS);
        assertEquals(1 / (3 * Q_UNIT[1]), u.get(1), EPS);
        assertEquals(1 / (3 * Q_UNIT[3]), u.get(3), EPS);
    }

    @Test
    public void pointsOnTheirCentersGiveFiniteWeights() {
        // Total cost 0: only the 8 * w_p / W(cluster) term remains, s = {4, 4, 4, 4}, q = 1/4.
        final SparseSeedingAlgorithm exact = seeding(new double[]{0, 5}, new int[][]{{0, 1}, {2, 3}});
        final CountingSampler sampler = new CountingSampler(0, 2, 3);
        final List<SparseWeightableVector> coreset =
                new SparseNonUniformCoreset(exact, 3, new SensitivityFunction(), sampler)
                        .takeSample(points(new double[]{0, 0, 5, 5}, UNIT));

        for (SparseWeightableVector point : coreset) {
            assertFalse(point.toString(), Double.isNaN(point.getWeight()));
            assertEquals(1 / (3 * .25), point.getWeight(), EPS);
        }
        assertEquals(3, coreset.size());
    }

    @Test
    public void inputWeightsEnterSensitivitiesAndWeights() {
        final CountingSampler sampler = new CountingSampler(0, 3);
        final List<SparseWeightableVector> coreset =
                new SparseNonUniformCoreset(ONE_CLUSTER_AT_0, 2, new SensitivityFunction(), sampler)
                        .takeSample(points(X, WEIGHTED));

        final Map<Integer, Double> u = weightsByIndex(coreset, sampler.dataset);
        assertEquals(2 / (2 * Q_WEIGHTED[0]), u.get(0), EPS);
        assertEquals(4 / (2 * Q_WEIGHTED[3]), u.get(3), EPS);
    }

    @Test
    public void expectedCoresetMatchesInputOverAllDrawSequences() {
        // E[sum u_p] = W and E[sum u_p * d(p, Q)^2] = cost(P, Q); enumerate every ordered pair of draws.
        final double center = 3;
        double expectedWeight = 0;
        double expectedCost = 0;
        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X.length; j++) {
                final CountingSampler sampler = new CountingSampler(i, j);
                final List<SparseWeightableVector> coreset =
                        new SparseNonUniformCoreset(ONE_CLUSTER_AT_0, 2, new SensitivityFunction(), sampler)
                                .takeSample(points(X, WEIGHTED));
                final double probability = Q_WEIGHTED[i] * Q_WEIGHTED[j];
                for (SparseWeightableVector point : coreset) {
                    final double d = point.getEntry(0) - center;
                    expectedWeight += probability * point.getWeight();
                    expectedCost += probability * point.getWeight() * d * d;
                }
            }
        }

        assertEquals(2 + 1 + 1 + 4, expectedWeight, 1e-9);
        assertEquals(2 * 9 + 1 * 9 + 1 * 9 + 4 * 49, expectedCost, 1e-9);
    }

    @Test
    public void samplerRepeatingDrawnPointsGetsTheSameWeights() {
        // BinarySearchRandomSample returns the same object once per draw instead of scaling it by its count.
        final RepeatingSampler sampler = new RepeatingSampler(0, 3, 3);
        final List<SparseWeightableVector> coreset =
                new SparseNonUniformCoreset(ONE_CLUSTER_AT_0, 3, new SensitivityFunction(), sampler)
                        .takeSample(points(X, UNIT));

        final Map<Integer, Double> u = weightsByIndex(coreset, sampler.dataset);
        assertEquals(1 * 1 / (3 * Q_UNIT[0]), u.get(0), EPS);
        assertEquals(1 * 2 / (3 * Q_UNIT[3]), u.get(3), EPS);
    }

    @Test
    public void inputNoLargerThanSampleSizeIsReturnedWithItsWeights() {
        final List<SparseWeightableVector> coreset =
                new SparseNonUniformCoreset(ONE_CLUSTER_AT_0, 4).takeSample(points(X, WEIGHTED));

        assertEquals(4, coreset.size());
        for (int i = 0; i < X.length; i++) {
            assertEquals(WEIGHTED[i], coreset.get(i).getWeight(), 0d);
        }
    }

    private static List<SparseWeightableVector> points(final double[] xs, final double[] weights) {
        final List<SparseWeightableVector> points = new ArrayList<>();
        for (int i = 0; i < xs.length; i++) {
            points.add(new SparseWeightableVector(new double[]{xs[i]}, weights[i]));
        }
        return points;
    }

    /** Sums coreset weights per input index, so a point listed twice is counted once with both weights. */
    private static Map<Integer, Double> weightsByIndex(final List<SparseWeightableVector> coreset,
                                                      final List<SparseWeightableVector> dataset) {
        final Map<SparseWeightableVector, Integer> indexOf = new IdentityHashMap<>();
        for (int i = 0; i < dataset.size(); i++) {
            indexOf.put(dataset.get(i), i);
        }
        final Map<Integer, Double> weights = new TreeMap<>();
        for (SparseWeightableVector point : coreset) {
            weights.merge(indexOf.get(point), point.getWeight(), Double::sum);
        }
        return weights;
    }

    /** Fixed partition of the (copied) input into clusters by index. */
    private static SparseSeedingAlgorithm seeding(final double[] centers, final int[][] members) {
        return vectors -> {
            final List<SparseCentroidCluster> clusters = new ArrayList<>();
            for (int c = 0; c < centers.length; c++) {
                final SparseCentroidCluster cluster =
                        new SparseCentroidCluster(new SparseWeightableVector(new double[]{centers[c]}, 1d));
                for (int i : members[c]) {
                    cluster.addPoint(vectors.get(i));
                }
                clusters.add(cluster);
            }
            return clusters;
        };
    }

    /** Draws a fixed index sequence and, like SparseRandomSample, returns each drawn point once scaled by its count. */
    private static final class CountingSampler implements RandomSampleAlgorithm {
        private final int[] draws;
        private List<SparseWeightableVector> dataset;

        CountingSampler(final int... draws) {
            this.draws = draws;
        }

        @Override
        public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {
            assertEquals(draws.length, t);
            this.dataset = dataset;
            final Map<Integer, Integer> counts = new TreeMap<>();
            for (int i : draws) {
                counts.merge(i, 1, Integer::sum);
            }
            final List<SparseWeightableVector> result = new ArrayList<>();
            for (Map.Entry<Integer, Integer> each : counts.entrySet()) {
                final SparseWeightableVector vector = dataset.get(each.getKey());
                vector.setProbability(vector.getProbability() * each.getValue());
                vector.setWeight(vector.getWeight() * each.getValue());
                result.add(vector);
            }
            return result;
        }
    }

    /** Draws a fixed index sequence and, like BinarySearchRandomSample, lists a point once per draw, unmodified. */
    private static final class RepeatingSampler implements RandomSampleAlgorithm {
        private final int[] draws;
        private List<SparseWeightableVector> dataset;

        RepeatingSampler(final int... draws) {
            this.draws = draws;
        }

        @Override
        public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {
            assertEquals(draws.length, t);
            this.dataset = dataset;
            final List<SparseWeightableVector> result = new ArrayList<>();
            for (int i : draws) {
                result.add(dataset.get(i));
            }
            return result;
        }
    }
}
