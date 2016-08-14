package univ.ml.sparse.algorithm;

import java.util.Collections;
import java.util.List;

import com.google.common.collect.Lists;

import univ.ml.sparse.CTRandomSample;
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

        double totalWeight = 0d;
        for (SparseWeightableVector vector : pointset) {
            totalWeight += vector.getWeight();
        }

        for (SparseWeightableVector point : copy) {
            point.setProbability((1d * t) / totalWeight);
            point.setWeight(totalWeight / t);
        }

        if (pointset.size() <= t) {
            return copy;
        }

        Collections.shuffle(copy);

        CTRandomSample sample = new CTRandomSample();

        return sample.getSampleOfSize(copy, t);
    }
}
