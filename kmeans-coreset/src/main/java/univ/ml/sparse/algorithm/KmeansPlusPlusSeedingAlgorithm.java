package univ.ml.sparse.algorithm;

import java.io.Serializable;
import java.util.List;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

/**
 * A decorator class for k-means algorithm, so it could philosophically substitute any of {@link SparseSeedingAlgorithm}
 * implementations, mostly used to configure {@link SparseNonUniformCoreset} the non-uniform coreset implementation algorithm.
 */
public class KmeansPlusPlusSeedingAlgorithm implements SparseSeedingAlgorithm, Serializable {

    private static final long serialVersionUID = -7486278241166556175L;

    final private SparseWeightedKMeansPlusPlus kmeans;

    public KmeansPlusPlusSeedingAlgorithm(final SparseWeightedKMeansPlusPlus kmeans) {
        this.kmeans = kmeans;
    }

    @Override
    public List<SparseCentroidCluster> seed(final List<SparseWeightableVector> vectors) {
        return kmeans.cluster(vectors);
    }
}
