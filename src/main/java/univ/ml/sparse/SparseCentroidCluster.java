package univ.ml.sparse;

import org.apache.commons.math3.stat.descriptive.moment.Variance;
import org.apache.commons.math3.util.FastMath;

import java.io.Serializable;

public class SparseCentroidCluster extends SparseCluster implements Serializable {

    private static final long serialVersionUID = -321133774182676323L;

    /** Center of the cluster. */
    private final SparseClusterable center;

    private Variance clusterVariance = new Variance();

    public SparseCentroidCluster(final SparseClusterable center) {
        super();
        this.center = center;
    }

    /**
     * Get the point chosen to be the center of this cluster.
     * @return chosen cluster center
     */
    public SparseClusterable getCenter() {
        return center;
    }

    @Override
    public void addPoint(SparseWeightableVector point) {
        super.addPoint(point);
        clusterVariance.increment(FastMath.sqrt(point.getWeight())*center.getVector().getDistance(point));
    }

    public double getClusterVariance() {
        return clusterVariance.getResult();
    }
}
