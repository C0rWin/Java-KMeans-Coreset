package univ.ml.sparse.algorithm;

import univ.ml.sparse.SparseWeightableVector;

import java.io.Serializable;
import java.util.List;

public interface SparseCoresetAlgorithm extends Serializable {

    List<SparseWeightableVector> takeSample(final List<SparseWeightableVector>  pointset);

}
