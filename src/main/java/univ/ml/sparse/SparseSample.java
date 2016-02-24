package univ.ml.sparse;

public interface SparseSample extends SparseWeightable {

    void setProbability(final double prob);

    double getProbability();

}
