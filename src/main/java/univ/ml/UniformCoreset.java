package univ.ml;

import com.clearspring.analytics.util.Lists;

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
