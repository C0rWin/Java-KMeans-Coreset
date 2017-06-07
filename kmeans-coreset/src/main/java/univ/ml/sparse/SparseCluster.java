package univ.ml.sparse;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

public class SparseCluster implements Serializable {

    private static final long serialVersionUID = -6260657669283135821L;

    /** The points contained in this cluster. */
    private final List<SparseWeightableVector> points;

    /**
     * Build a cluster centered at a specified point.
     */
    public SparseCluster() {
        points = new ArrayList<>();
    }

    /**
     * Add a point to this cluster.
     * @param point point to add
     */
    public void addPoint(final SparseWeightableVector point) {
        points.add(point);
    }

    /**
     * Get the points contained in the cluster.
     * @return points contained in the cluster
     */
    public List<SparseWeightableVector> getPoints() {
        return points;
    }
}
