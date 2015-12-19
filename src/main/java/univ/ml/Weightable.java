package univ.ml;

import org.apache.commons.math3.ml.clustering.Clusterable;

public interface Weightable<T> extends Clusterable, Labelable<String>, Comparable<T> {

    double getWeight();

    void setWeight(final double weight);
}
