package univ.ml.sparse;

import org.apache.commons.math3.linear.OpenMapRealVector;
import org.apache.commons.math3.linear.RealVector;

import java.util.Map;

public class SparseWeightableVector extends OpenMapRealVector implements SparseSample, SparseWeightable, SparseClusterable, Comparable<SparseWeightableVector> {

    private static final long serialVersionUID = 3069201894898346800L;

    private double weight;

    private double probability;

    public SparseWeightableVector() {

    }

    public SparseWeightableVector(final Map<Integer, Double> values, final double weight, final int dim) {
        super(dim);
        for (Map.Entry<Integer, Double> each : values.entrySet()) {
            this.setEntry(each.getKey(), each.getValue());
        }
        this.weight = weight;
    }

    public SparseWeightableVector(final Map<Integer, Double> values, final int dim) {
        super(dim);
        for (Map.Entry<Integer, Double> each : values.entrySet()) {
            this.setEntry(each.getKey(), each.getValue());
        }
    }

    public SparseWeightableVector(int dimension) {
        super(dimension);
        this.weight = 1.0;
    }

    public SparseWeightableVector(final RealVector vector) {
        super(vector);
    }

    public SparseWeightableVector(final RealVector vector, double weight) {
        super(vector);
        this.weight = weight;
    }

    public SparseWeightableVector(double[] coords, double weight) {
        super(coords);
        this.weight = weight;
    }

    @Override
    public void setProbability(double prob) {
        this.probability = prob;
    }

    @Override
    public double getProbability() {
        return probability;
    }

    @Override
    public void setWeight(double weight) {
        this.weight = weight;
    }

    @Override
    public double getWeight() {
        return weight;
    }

    @Override
    public RealVector getVector() {
        return this;
    }

    @Override
    public int compareTo(SparseWeightableVector o) {
        return Double.valueOf(probability).compareTo(o.getProbability());
    }
}
