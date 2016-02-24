package univ.ml.sparse.algorithm;

import univ.ml.sparse.SparseWeightableVector;

import java.util.List;

public interface SparseCoresetAlgorithm {

    List<SparseWeightableVector> takeSample(final List<SparseWeightableVector>  pointset);

}
