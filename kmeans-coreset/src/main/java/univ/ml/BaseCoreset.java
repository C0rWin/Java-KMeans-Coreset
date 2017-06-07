package univ.ml;

import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

import java.util.Collections;
import java.util.List;

public abstract class BaseCoreset<T extends Sample> implements CoresetAlgorithm<T> {

    @Override
    public List<T> concat(List<T> p1, List<T> p2) {
        final List<T> result = Lists.newArrayList(p1);
        Iterables.addAll(result, p2);
        Collections.sort(result);
        return result;
    }
}
