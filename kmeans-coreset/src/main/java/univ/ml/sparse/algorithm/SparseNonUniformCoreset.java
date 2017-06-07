package univ.ml.sparse.algorithm;

import java.util.List;

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
        for (SparseWeightableVector vector : pointset) {
            dataset.add(new SparseWeightableVector(vector.copy()));
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

        for (final SparseCentroidCluster cluster : clusters) {
            // Sum of weights of points belongs to cluster
            final double clusterWeight = getTotalWeight(cluster);
            final SparseClusterable center = cluster.getCenter();

            for (SparseWeightableVector point : cluster.getPoints()) {
                // Distance between cluster and the point multiplied by point weight
                final double dist = getPoint2ClusterSqDist(center, point);

                // Compute sensitivity for each point.
                final double s_p = sensitivityFunction.sensitivity(point, dist, clusterWeight, totalVariance);

                // Adjust new weights for cluster points.
                changePointWeight(clusters, point, s_p);
            }
        }

        final List<SparseWeightableVector> sample = samplingAlg.getSampleOfSize(dataset, sampleSize);

        // Since sampling with repetitions sampleSize could be greater or equal to real sample.size()
        // therefore loop below used to balance weights
        double t_bound = 8 * clusters.size() + 2;
        for (SparseWeightableVector point : sample) {
            double s_p = point.getProbability() * t_bound;
            point.setWeight(point.getWeight() / (sample.size() * s_p));
        }

        return sample;
    }

    private void changePointWeight(List<SparseCentroidCluster> clusters, SparseWeightableVector point, double s_p) {
        double t_bound = 8 * clusters.size() + 2; // \sum_{p \in P} s(p) = 8 * k + 2

        point.setProbability(s_p / t_bound);
        /**
         * Adjust new weight, s.t. \sum_{p \in P}dist(p, Q)^2 = E[\sum_{p \in S} dist(p, Q)^2]
         *
         * Based on sensitivity framework new weight should be u(p) := \frac{t_bound * w(p)}{|S|*s(p)} or
         *
         * u_p = (t_bound * w_p)/(sampleSize * s_p)
         */
        point.setWeight((t_bound * point.getWeight()));
//        point.setWeight((t_bound * point.getWeight()) / (sampleSize * s_p));
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
