// SPDX-License-Identifier: MIT
package univ.ml.sparse.algorithm;

import java.util.List;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

public class SparseCoresetUtility {

    /**
     * Weighted k-means cost of {@code points} w.r.t. the cluster centers: sum_p w_p * min_c ||p - c||^2.
     */
    public static double getEnergy(final List<SparseCentroidCluster> centers, final List<SparseWeightableVector> points) {
        double energy = 0;
        for (final SparseWeightableVector point : points) {
            double minSquaredDistance = Double.POSITIVE_INFINITY;
            for (final SparseCentroidCluster center : centers) {
                final double distance = center.getDistanceToCenter(point);
                minSquaredDistance = Math.min(minSquaredDistance, distance * distance);
            }
            energy += point.getWeight() * minSquaredDistance;
        }
        return energy;
    }
}
