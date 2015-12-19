package univ.ml;

public interface Sample<T> extends Weightable<T> {

    double getProbability();

    void setProbability(final double probability);
}
