package univ.ml;

import com.google.common.collect.Lists;

import java.util.List;
import java.util.Random;

public class RandomSample<T extends Sample> {

    private List<T> dataset;

    public RandomSample(final List<T> dataset) {
        this.dataset = dataset;
    }

    public List<T> getSampleOfSize(final int t) {
        if (t <= dataset.size())
            return dataset;

        final List<T> result = Lists.newArrayListWithExpectedSize(t);

        double total = 0;

        for (T point : dataset) {
            total += point.getProbability();
        }

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
