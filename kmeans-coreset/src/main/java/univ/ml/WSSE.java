package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.util.FastMath;

import java.util.List;

public class WSSE implements ClusteringCostFunction {

    private final DistanceMeasure measure;

    public WSSE(final DistanceMeasure measure) {
        this.measure = measure;
    }

    @Override
    public double getCost(final List<CentroidCluster<WeightedDoublePoint>> clusters) {
        double cost = 0;
        for (CentroidCluster<WeightedDoublePoint> cluster : clusters) {
            for (WeightedDoublePoint point : cluster.getPoints()) {
                cost += FastMath.pow(measure.compute(cluster.getCenter().getPoint(), point.getPoint()), 2);
            }
        }
        return cost;
    }

    @Override
    public double getCost(List<WeightedDoublePoint> centers, List<WeightedDoublePoint> pointSet) {
        double cost = 0;
        for (WeightedDoublePoint center : centers) {
            double minDist = Double.MAX_VALUE;
            for (WeightedDoublePoint point : pointSet) {
                minDist = FastMath.min(minDist, Math.pow(measure.compute(center.getPoint(), point.getPoint()), 2));
            }
            cost += minDist;
        }
        return cost;
    }
}
