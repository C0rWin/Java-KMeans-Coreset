package univ.ml.sparse;

import org.apache.commons.math3.util.FastMath;

import java.util.List;

public class SparseWSSE {

    public double getCost(final List<SparseCentroidCluster> clusters) {
        double cost = 0;
        for (SparseCentroidCluster cluster : clusters) {
            double minDistance = Double.MAX_VALUE;
            for (SparseWeightableVector vector : cluster.getPoints()) {
                double d = FastMath.pow(cluster.getCenter().getVector().mapMultiply(-1.0).add(vector).getNorm(), 2);
                minDistance  = FastMath.min(d, minDistance);
            }
            // Compute the cost of given clustering
            cost += minDistance;
        }
        return cost;
    }

    public double getCost(List<SparseWeightableVector> centers, List<SparseWeightableVector> pointSet) {
        double cost = 0;
        for (SparseWeightableVector center : centers) {
            // For each center and for each point compute pair wise distance and
            // pick the smallest one to include it in cost.
            double minDistance = Double.MAX_VALUE;
            for (SparseWeightableVector vector : pointSet) {
                double d = FastMath.pow(center.getVector().mapMultiply(-1.0).add(vector).getNorm(), 2);
                minDistance  = FastMath.min(d, minDistance);
            }
            // Compute the cost of given clustering
            cost += minDistance;
        }
        return cost;
    }

}
