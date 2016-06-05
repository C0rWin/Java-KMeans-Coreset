package univ.ml.sparse.algorithm;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;

import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWeightableVector;

public class SparseUniformCoreset implements SparseCoresetAlgorithm {

    private static final long serialVersionUID = -6996893318896045066L;

    private int t;

    public SparseUniformCoreset(final int t) {
        this.t = t;
    }

    @Override
    public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
        final List<SparseWeightableVector> copy = Lists.newArrayList(pointset);
        if (pointset.size() <= t) {
            for (SparseWeightableVector each : copy) {
                each.setWeight(1);
                each.setProbability(1.0/pointset.size());
            }
            return copy;
        }

        Collections.sort(copy);
        SparseRandomSample sample = new SparseRandomSample(copy);
        List<SparseWeightableVector> result = sample.getSampleOfSize(t);
        for (SparseWeightableVector point : result) {
            point.setProbability((1.0*pointset.size())/t);
            point.setWeight((1.0*pointset.size())/t);
        }
        return result;
    }
}
