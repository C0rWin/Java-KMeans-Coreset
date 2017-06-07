package univ.ml.sparse;

import java.util.List;

public interface RandomSampleAlgorithm {

    List<SparseWeightableVector> getSampleOfSize(List<SparseWeightableVector> dataset, final int t);

}
