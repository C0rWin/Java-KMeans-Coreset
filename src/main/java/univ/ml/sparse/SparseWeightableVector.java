package univ.ml.sparse;

import java.io.Serializable;
import java.util.Iterator;
import java.util.Map;

import org.apache.commons.math3.linear.OpenMapRealVector;
import org.apache.commons.math3.linear.RealVector;

public class SparseWeightableVector extends OpenMapRealVector implements SparseSample, SparseWeightable,
        SparseClusterable, Comparable<SparseWeightableVector>, Serializable {

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
        this.weight = 1.0;
    }

    public SparseWeightableVector(int dimension) {
        super(dimension);
        this.weight = 1.0;
    }

    public SparseWeightableVector(final RealVector vector) {
        this(vector, 1.0);
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
    public String toString() {
    	StringBuilder buf = new StringBuilder();
    	Iterator<Entry> it = getVector().iterator();
    	
    	while (it.hasNext()) {
    		Entry ent = it.next();
    		assert(ent != null);
    		
    		if (ent.getValue() != 0.0) {
    			if (buf.length() > 0) {
        			buf.append(',');
    			}
    			
    			buf.append(String.format("(%d,%f)", ent.getIndex(), ent.getValue()));
    		}
    	}
    	
    	return String.format("w=%f,p=%f,vec=[%s]", getWeight(), getProbability(), buf.toString());
    }

    @Override
    public int compareTo(SparseWeightableVector o) {
        return Double.valueOf(probability).compareTo(o.getProbability());
    }
}
