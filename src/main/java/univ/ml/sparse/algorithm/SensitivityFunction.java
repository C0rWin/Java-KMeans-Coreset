package univ.ml.sparse.algorithm;

import univ.ml.sparse.SparseWeightableVector;

public class SensitivityFunction {

    public double sensitivity(SparseWeightableVector point, double dist, double clusterWeight, double totalVariance) {
        return (8d * point.getWeight()) / clusterWeight + 2d * dist / totalVariance;
    }
}
