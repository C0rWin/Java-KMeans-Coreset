package univ.ml.sparse.algorithm;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.util.FastMath;

import com.google.common.collect.Lists;

import univ.ml.sparse.RandomSampleAlgorithm;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseClusterable;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWeightableVector;

public class SparseNonUniformCoreset implements SparseCoresetAlgorithm {

    private static final long serialVersionUID = -390692758659255533L;

    private int sampleSize;

    private SensitivityFunction sensitivityFunction = new SensitivityFunction();

    private SparseSeedingAlgorithm seedingAlgorithm;

    private RandomSampleAlgorithm samplingAlg = new SparseRandomSample();

    /**
     * Constructor for non uniform sparse coreset algorithm for k-means
     *
     * @param seedingAlgorithm seeding algorithm which is capable to provide (alpha, beta) approximation for k-means.
     * @param sampleSize       the coreset size to sample
     */
    public SparseNonUniformCoreset(final SparseSeedingAlgorithm seedingAlgorithm, final int sampleSize) {
        this.sampleSize = sampleSize;
        // Default is the k-means++ algorithm which provides (1, log(n)) approximation for k-means problem.
        this.seedingAlgorithm = seedingAlgorithm;
    }

    /**
     * @param seedingAlgorithm    seeding algorithm which is capable to provide (alpha, beta) approximation for k-means.
     * @param sampleSize          coreset size
     * @param sensitivityFunction preferred sensitivity function to compute
     */
    public SparseNonUniformCoreset(final SparseSeedingAlgorithm seedingAlgorithm, int sampleSize, SensitivityFunction sensitivityFunction) {
        this(seedingAlgorithm, sampleSize);
        this.sensitivityFunction = sensitivityFunction;
    }

    /**
     * @param seedingAlgorithm    seeding algorithm which is capable to provide (alpha, beta) approximation for k-means.
     * @param sampleSize          coreset size
     * @param sensitivityFunction preferred sensitivity function to compute
     * @param samplingAlg         points sampling algorithm
     */
    public SparseNonUniformCoreset(final SparseSeedingAlgorithm seedingAlgorithm, int sampleSize,
                                   final SensitivityFunction sensitivityFunction,
                                   final RandomSampleAlgorithm samplingAlg) {
        this(seedingAlgorithm, sampleSize, sensitivityFunction);
        this.samplingAlg = samplingAlg;
    }

    @Override
    public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
        final List<SparseWeightableVector> dataset = Lists.newArrayList();

        // Copy initial set of points to not affect original points weights and probabilities
        final Map<SparseWeightableVector, Integer> indexOf = new IdentityHashMap<>();
        for (SparseWeightableVector vector : pointset) {
            // copy() yields a plain OpenMapRealVector, so the weight has to be carried over explicitly
            final SparseWeightableVector copy = new SparseWeightableVector(vector.copy(), vector.getWeight());
            indexOf.put(copy, dataset.size());
            dataset.add(copy);
        }

        if (dataset.size() <= sampleSize) {
            double totalWeight = 0d;
            for (SparseWeightableVector vector : dataset) {
                totalWeight += vector.getWeight();
            }

            final List<SparseWeightableVector> result = Lists.newArrayList(dataset);
            for (SparseWeightableVector each : result) {
                each.setProbability(totalWeight / dataset.size());
            }
            return result;
        }

        final List<SparseCentroidCluster> clusters = seedingAlgorithm.seed(dataset);
        double totalVariance = 0d;

        for (final SparseCentroidCluster cluster : clusters) {
            final SparseClusterable center = cluster.getCenter();
            // Compute sum_p(w_p*dist(p,c)^2)
            for (SparseWeightableVector point : cluster.getPoints()) {
                totalVariance += getPoint2ClusterSqDist(center, point);
            }
        }

        // Zero cost means every point lies on its center, so the distance term must vanish rather than be 0/0.
        final double variance = totalVariance > 0 ? totalVariance : 1d;

        final double[] sensitivities = new double[dataset.size()];
        double totalSensitivity = 0d;

        for (final SparseCentroidCluster cluster : clusters) {
            // Sum of weights of points belongs to cluster
            final double clusterWeight = getTotalWeight(cluster);
            final SparseClusterable center = cluster.getCenter();

            for (SparseWeightableVector point : cluster.getPoints()) {
                // Distance between cluster and the point multiplied by point weight
                final double dist = getPoint2ClusterSqDist(center, point);

                // Compute sensitivity for each point.
                final double s_p = sensitivityFunction.sensitivity(point, dist, clusterWeight, variance);
                sensitivities[indexOf.get(point)] = s_p;
                totalSensitivity += s_p;
            }
        }

        // Normalize by the actual sum: 8k+2 overcounts whenever the seeding leaves some clusters empty.
        for (int i = 0; i < dataset.size(); i++) {
            dataset.get(i).setProbability(sensitivities[i] / totalSensitivity);
        }

        final List<SparseWeightableVector> sample = samplingAlg.getSampleOfSize(dataset, sampleSize);

        // Samplers either list a point once per draw, or once with probability and weight scaled by its draw count.
        final int[] draws = new int[dataset.size()];
        final List<SparseWeightableVector> coreset = Lists.newArrayList();
        for (SparseWeightableVector point : sample) {
            final int i = indexOf.get(point);
            if (draws[i] == 0) {
                coreset.add(point);
            }
            draws[i] += (int) Math.round(point.getProbability() * totalSensitivity / sensitivities[i]);
        }

        // u_p = w_p * c_p / (m * q_p) makes the expected coreset cost equal the input cost for any centers.
        for (SparseWeightableVector point : coreset) {
            final int i = indexOf.get(point);
            final double q_p = sensitivities[i] / totalSensitivity;
            point.setWeight(pointset.get(i).getWeight() * draws[i] / (sampleSize * q_p));
            point.setProbability(q_p);
        }

        return coreset;
    }

    private double getTotalWeight(SparseCentroidCluster cluster) {
        double totalWeight = 0d;
        for (final SparseWeightableVector vector : cluster.getPoints()) {
            totalWeight += vector.getWeight();
        }
        return totalWeight;
    }

    private double getPoint2ClusterSqDist(SparseClusterable center, SparseWeightableVector point) {
        return point.getWeight() * FastMath.pow(center.getVector().getDistance(point.getVector()), 2);
    }
}
