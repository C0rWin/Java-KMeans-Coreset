package univ.ml.sparse.algorithm;

import java.io.Serializable;

import univ.ml.sparse.SparseWeightableVector;

public class SensitivityFunction implements Serializable {

    private static final long serialVersionUID = -2966005364936959356L;

    public double sensitivity(SparseWeightableVector point, double dist, double clusterWeight, double totalVariance) {
        return 8d * point.getWeight() / clusterWeight + 2d * dist / totalVariance;
    }
}
