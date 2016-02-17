package univ.ml;

import java.util.List;

public interface CoresetAlgorithm<T extends Sample> {

    List<T> concat(final List<T> p1, final List<T> p2);

    List<T> takeSample(final List<T>  pointset);
}
