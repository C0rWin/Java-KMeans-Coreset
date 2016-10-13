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

    public SparseKmeansCoresetAlgorithm(int sampleSize) {
        this.sampleSize = sampleSize;
    }

    @Override
    public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
        if (pointset.size() <= sampleSize)
            return pointset;

        final SparseWeightedKMeansPlusPlus kmeans = new SparseWeightedKMeansPlusPlus(sampleSize);
        final List<SparseCentroidCluster> clusters = kmeans.cluster(pointset);

        final List<SparseWeightableVector> results = new ArrayList<>();
        for (final SparseCentroidCluster cluster : clusters) {
            final RealVector center = cluster.getCenter().getVector();
            final int weight = cluster.getPoints().size();
            results.add(new SparseWeightableVector(center, weight));
        }

        return results;
    }
}
