package univ.ml.sparse.algorithm;

import java.util.List;

import com.google.common.collect.Lists;

import univ.ml.sparse.RandomSampleAlgorithm;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWeightableVector;

public class SparseUniformCoreset implements SparseCoresetAlgorithm {

    private static final long serialVersionUID = -6996893318896045066L;

    private int sampleSize;

    public SparseUniformCoreset(final int sampleSize) {
        this.sampleSize = sampleSize;
    }

    @Override
    public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
        final List<SparseWeightableVector> dataset = Lists.newArrayList();

        // Copy initial set of points to not affect original points weights and probabilities
        // (OpenMapRealVector.copy() drops the weight, so it is passed explicitly)
        for (SparseWeightableVector vector : pointset) {
            dataset.add(new SparseWeightableVector(vector.copy(), vector.getWeight()));
        }

        if (dataset.size() <= sampleSize) {
            return dataset;
        }

        double totalWeight = 0d;
        for (SparseWeightableVector vector : dataset) {
            totalWeight += vector.getWeight();
        }

        // Drawing p with probability w_p / W and giving each draw weight W / m makes E[weight of p] = w_p.
        for (SparseWeightableVector point : dataset) {
            point.setProbability(point.getWeight() / totalWeight);
            point.setWeight(totalWeight / sampleSize);
        }

        RandomSampleAlgorithm sample = new SparseRandomSample();

        return sample.getSampleOfSize(dataset, sampleSize);
    }
}
