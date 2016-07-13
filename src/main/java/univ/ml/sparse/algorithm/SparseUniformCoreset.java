package univ.ml.sparse.algorithm;

import java.util.Collections;
import java.util.List;

import org.apache.commons.math3.stat.descriptive.summary.Sum;

import com.google.common.collect.Lists;

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

        final Sum totalWeight = new Sum();
        for (SparseWeightableVector vector : pointset) {
            totalWeight.increment(vector.getWeight());
        }

        for (SparseWeightableVector point : copy) {
            point.setProbability((1d * t) / pointset.size());
            point.setWeight((1.0 * totalWeight.getResult()) / t);
//            point.setWeight((1.0 * pointset.size()) / t);
        }

        if (pointset.size() <= t) {
            return copy;
        }

        Collections.shuffle(copy);

//        CTRandomSample sample = new CTRandomSample(copy);

        List<SparseWeightableVector> result = copy.subList(0, t);

        return result;
    }
}
