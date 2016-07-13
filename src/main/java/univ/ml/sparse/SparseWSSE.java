package univ.ml.sparse;

import java.util.Collection;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.util.FastMath;

public class SparseWSSE {

    public double getCost(final Collection<SparseCentroidCluster> clusters) {
        double cost = 0;
        for (SparseCentroidCluster cluster : clusters) {
            final RealVector center = cluster.getCenter().getVector();
            for (SparseWeightableVector vector : cluster.getPoints()) {
                cost += FastMath.pow(center.getDistance(vector), 2);
            }
        }
        return cost;
    }

    public double getCost(Collection<SparseWeightableVector> centers, Collection<SparseWeightableVector> pointSet) {
        double cost = 0;
        for (SparseWeightableVector vector : pointSet) {
            // For each center and for each point compute pair wise distance and
            // pick the smallest one to include it in cost.
            double minDistance = Double.MAX_VALUE;
            for (SparseWeightableVector center : centers) {

                double d = FastMath.pow(center.getVector().getDistance(vector), 2);
                minDistance = FastMath.min(d, minDistance);
            }
            // Compute the cost of given clustering
            cost += minDistance;
        }
        return cost;
    }

}
