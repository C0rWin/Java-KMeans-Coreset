package univ.ml;

import java.util.List;

public interface CoresetAlgorithm<T extends Sample> {

    List<T> merge(final List<T> p1, final List<T> p2);

    List<T> reduce(final List<T>  pointset);
}
