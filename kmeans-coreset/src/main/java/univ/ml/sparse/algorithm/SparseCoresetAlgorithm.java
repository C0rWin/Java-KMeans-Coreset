package univ.ml.sparse.algorithm;

import java.io.Serializable;
import java.util.List;

import univ.ml.sparse.SparseWeightableVector;

public interface SparseCoresetAlgorithm extends Serializable {

    List<SparseWeightableVector> takeSample(final List<SparseWeightableVector>  pointset);

}
