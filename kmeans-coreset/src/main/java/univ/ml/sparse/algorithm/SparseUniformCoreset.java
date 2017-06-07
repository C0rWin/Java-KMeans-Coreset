package univ.ml.sparse.algorithm;

import java.util.Collections;
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
        for (SparseWeightableVector vector : pointset) {
            dataset.add(new SparseWeightableVector(vector.copy()));
        }

        double totalWeight = 0d;
        for (SparseWeightableVector vector : dataset) {
            totalWeight += vector.getWeight();
        }

        for (SparseWeightableVector point : dataset) {
            point.setProbability((1d * sampleSize) / totalWeight);
            point.setWeight(totalWeight / sampleSize);
        }

        if (dataset.size() <= sampleSize) {
            return dataset;
        }

        Collections.shuffle(dataset);

        RandomSampleAlgorithm sample = new SparseRandomSample();

        return sample.getSampleOfSize(dataset, sampleSize);
    }
}
