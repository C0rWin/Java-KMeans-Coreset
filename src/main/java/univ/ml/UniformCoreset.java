package univ.ml;

import java.util.Collections;
import java.util.List;

public class UniformCoreset<T extends Sample> extends BaseCoreset<T>{

    private int t;


    public UniformCoreset(final int t) {
        this.t = t;
    }

    @Override
    public List<T> reduce(final List<T> pointset) {
        Collections.sort(pointset);
        RandomSample<T> sample = new RandomSample<>(pointset);
        List<T> result = sample.getSampleOfSize(t);
        for (T point : result) {
            point.setProbability((1.0*pointset.size())/t);
            point.setWeight((1.0*pointset.size())/t);
        }
        return result;
    }
}
