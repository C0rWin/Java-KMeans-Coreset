package univ.ml;

import java.io.Serializable;
import java.util.Arrays;

public class WeightedDoublePoint implements Sample<WeightedDoublePoint>, Serializable {

    /** Serializable version identifier. */
    private static final long serialVersionUID = 3946024775784901369L;

    /** Point coordinates. */
    private final double[] point;

    private double weight;

    private double probability;

    private String label;

    /**
     * Build an instance wrapping an double array.
     * <p>
     * The wrapped array is referenced, it is <em>not</em> copied.
     *
     * @param point the n-dimensional point in double space
     */
    public WeightedDoublePoint(final double[] point, final double weight, final String label) {
        this.point = point;
        this.weight = weight;
        this.label = label;
    }

    /** {@inheritDoc} */
    public double[] getPoint() {
        return point;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(final Object other) {
        if (!(other instanceof WeightedDoublePoint)) {
            return false;
        }
        return Arrays.equals(point, ((WeightedDoublePoint) other).point);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Arrays.hashCode(point);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return Arrays.toString(point);
    }

    public double getWeight() {
        return weight;
    }

    @Override
    public void setWeight(final double weight) {
        this.weight = weight;
    }

    public String getLabel() {
        return label;
    }

    @Override
    public int compareTo(WeightedDoublePoint o) {
        return Double.valueOf(probability).compareTo(o.getProbability());
    }

    @Override
    public double getProbability() {
        return probability;
    }

    @Override
    public void setProbability(double probability) {
        this.probability = probability;
    }
}