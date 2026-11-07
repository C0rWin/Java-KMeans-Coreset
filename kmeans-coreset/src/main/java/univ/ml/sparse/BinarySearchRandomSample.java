package univ.ml.sparse;

import com.google.common.collect.Lists;

import java.util.List;

public class BinarySearchRandomSample implements RandomSampleAlgorithm {

    public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {
        final List<SparseWeightableVector> result = Lists.newArrayList();

        final double[] cdf = Cdf.of(dataset);
        for (int i = 0; i < t; i++) {
            result.add(dataset.get(Cdf.draw(cdf, Math.random())));
        }

        return result;
    }

}
