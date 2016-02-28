package univ.ml;

import java.io.Serializable;
import java.util.List;

public interface CoresetAlgorithm<T extends Sample> extends Serializable {

    List<T> concat(final List<T> p1, final List<T> p2);

    List<T> takeSample(final List<T>  pointset);
}
