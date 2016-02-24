package univ.ml;

import com.google.common.collect.Lists;

import java.util.Collections;
import java.util.List;

public class UniformCoreset<T extends Sample> extends BaseCoreset<T>{

    private int t;

    public UniformCoreset(final int t) {
        this.t = t;
    }

    @Override
    public List<T> takeSample(final List<T> pointset) {
        final List<T> copy = Lists.newArrayList(pointset);
        if (pointset.size() <= t) {
            for (T each : copy) {
                each.setWeight(1);
                each.setProbability(1.0/pointset.size());
            }
            return copy;
        }

        Collections.sort(copy);
        RandomSample<T> sample = new RandomSample<>(copy);
        List<T> result = sample.getSampleOfSize(t);
        for (T point : result) {
            point.setProbability((1.0*pointset.size())/t);
            point.setWeight((1.0*pointset.size())/t);
        }
        return result;
    }
}
