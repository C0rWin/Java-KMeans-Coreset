package univ.ml.sparse.algorithm;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

import java.util.List;

public interface SparseSeedingAlgorithm {

    /**
     * Seed initial cluster centers.
     *
     * @param vectors vectors to select initial centers from. Method should not change input parameter.
     * @return list of the cluster centers.
     */
    List<SparseCentroidCluster> seed(final List<SparseWeightableVector> vectors);

}
