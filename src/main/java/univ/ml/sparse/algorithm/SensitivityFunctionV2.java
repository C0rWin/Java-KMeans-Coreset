package univ.ml.sparse.algorithm;

import univ.ml.sparse.SparseWeightableVector;

public class SensitivityFunctionV2 extends  SensitivityFunction {

    @Override
    public double sensitivity(SparseWeightableVector point, double dist, double clusterWeight, double totalVariance) {
        return 8d / clusterWeight + 2d * dist / totalVariance;
    }
}
