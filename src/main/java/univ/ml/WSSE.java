package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

import java.util.List;
import java.util.stream.Collectors;

public class WSSE implements ClusteringCostFunction {

    private final DistanceMeasure measure;

    public WSSE(final DistanceMeasure measure) {
        this.measure = measure;
    }

    @Override
    public double getCost(final List<CentroidCluster<WeightedDoublePoint>> clusters) {
        return clusters.stream()
                .map(cluster ->
                        cluster.getPoints().stream()
                                .map(point ->
                                        Math.pow(measure.compute(cluster.getCenter().getPoint(), point.getPoint()), 2)
                                ).collect(Collectors.summingDouble(x -> x))
                ).collect(Collectors.summingDouble(x -> x));
    }
}
