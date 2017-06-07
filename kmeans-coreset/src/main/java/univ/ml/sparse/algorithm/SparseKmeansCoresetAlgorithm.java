package univ.ml.sparse.algorithm;

import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.RealVector;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

public class SparseKmeansCoresetAlgorithm implements SparseCoresetAlgorithm {

    private static final long serialVersionUID = 6378586987151382195L;

    private int sampleSize;

    public SparseKmeansCoresetAlgorithm() {
        this(0);
    }

    /**
     *
     * @param sampleSize
     */
    public SparseKmeansCoresetAlgorithm(int sampleSize) {
        this.sampleSize = sampleSize;
    }

    @Override
    public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
        if (pointset.size() <= sampleSize)
            return pointset;

        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(sampleSize);
        System.out.println("XXX: Running k-means with k = " + sampleSize);
        final long before = System.currentTimeMillis();
        final List<SparseCentroidCluster> clusters = kmeans.cluster(pointset);
        System.out.println("Took " + (System.currentTimeMillis() - before) + " ms." );

        final List<SparseWeightableVector> results = new ArrayList<>();
        for (final SparseCentroidCluster cluster : clusters) {
            final RealVector center = cluster.getCenter().getVector();
            double weight = 0d;
            // Get cluster total weight
            for (SparseWeightableVector point : cluster.getPoints()) {
                weight += point.getWeight();
            }

            results.add(new SparseWeightableVector(center, weight));
        }

        return results;
    }
}
