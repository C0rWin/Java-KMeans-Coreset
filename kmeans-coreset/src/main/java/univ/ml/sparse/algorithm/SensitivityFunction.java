package univ.ml.sparse.algorithm;

import java.io.Serializable;

import univ.ml.sparse.SparseWeightableVector;

public class SensitivityFunction implements Serializable {

    private static final long serialVersionUID = -2966005364936959356L;

    public double sensitivity(SparseWeightableVector point, double dist, double clusterWeight, double totalVariance) {
        // dist already includes weight, hence no need to multiply it second time here
        // s_p := 8 * w_p / sum_p(w_p) + 2 * w_p * dist(p, c)^2 / sum_p(w_p * dist(p, c)^2)
        return 8d * point.getWeight() / clusterWeight + 2d * dist / totalVariance;
    }
}
