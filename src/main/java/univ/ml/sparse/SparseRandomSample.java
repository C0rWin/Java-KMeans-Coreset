package univ.ml.sparse;

import com.google.common.collect.Lists;

import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

public class SparseRandomSample {

    private List<SparseWeightableVector> dataset;

    public SparseRandomSample(final List<SparseWeightableVector> dataset) {
        this.dataset = dataset;
    }

    public List<SparseWeightableVector> getSampleOfSize(final int t) {
        if (t <= dataset.size())
            return dataset;

        final List<SparseWeightableVector> result = Lists.newArrayListWithExpectedSize(t);

        final double total = dataset.stream()
                .map(point -> point.getProbability())
                .collect(Collectors.summingDouble(x -> x));

        final Random rnd = new Random();

        for (int j = 0; j < t; j++) {
            double cdf = total * rnd.nextDouble();
            int i  = 0;
            for (double sum = 0; sum < cdf; i++) {
                sum += dataset.get(i).getProbability();
            }
            result.add(dataset.get(i-1));
        }

        return result;
    }

}
