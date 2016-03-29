package univ.ml.sparse;

import com.google.common.collect.Lists;

import java.util.List;
import java.util.Random;

public class SparseRandomSample {

    private List<SparseWeightableVector> dataset;

    public SparseRandomSample(final List<SparseWeightableVector> dataset) {
        this.dataset = dataset;
    }

    public List<SparseWeightableVector> getSampleOfSize(final int t) {
        if (t >= dataset.size())
            return dataset;

        final List<SparseWeightableVector> result = Lists.newArrayListWithExpectedSize(t);

        double total = 0;

        for (SparseWeightableVector vector : dataset) {
            total += vector.getProbability();
        }

        final Random rnd = new Random();

        for (int j = 0; j < t; j++) {
        	double pr = rnd.nextDouble();
            double cdf = total * pr;
            
            int i  = 0;
            for (double sum = 0; sum < cdf; i++) {
                sum += dataset.get(i).getProbability();
            }
            result.add(dataset.get(Math.max(0, i-1)));
        }

        return result;
    }

}
