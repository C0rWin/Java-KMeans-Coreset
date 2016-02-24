package univ.ml.sparse;

import java.io.Serializable;

public class SparseCentroidCluster extends SparseCluster implements Serializable {

    private static final long serialVersionUID = -321133774182676323L;

    /** Center of the cluster. */
    private final SparseClusterable center;

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

}
