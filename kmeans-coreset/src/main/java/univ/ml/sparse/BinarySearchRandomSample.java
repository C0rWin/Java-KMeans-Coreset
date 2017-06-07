package univ.ml.sparse;

import com.google.common.collect.Lists;

import java.util.Arrays;
import java.util.List;

public class BinarySearchRandomSample implements RandomSampleAlgorithm {

    public List<SparseWeightableVector> getSampleOfSize(final List<SparseWeightableVector> dataset, final int t) {
        final List<SparseWeightableVector> result = Lists.newArrayList();

        double [] cdf = new double[dataset.size()];
        for (int i = 0; i < cdf.length; i++) {
            cdf[i] = dataset.get(i).getProbability();
        }

        for (int i = 1; i < cdf.length; i++) {
            cdf[i] += cdf[i - 1];
        }

        for (int i = 0; i < t; i++) {
            result.add(dataset.get(Math.min(Math.abs(Arrays.binarySearch(cdf, Math.random())), dataset.size() -1)));
        }

        return result;
    }

}
