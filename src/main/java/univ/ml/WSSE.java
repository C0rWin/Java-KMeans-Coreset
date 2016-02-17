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
                                ).collect(Collectors.summingDouble(x -> (double)x))
                ).collect(Collectors.summingDouble(x -> x));
    }

    @Override
    public double getCost(List<WeightedDoublePoint> centers, List<WeightedDoublePoint> pointSet) {
        return pointSet.stream()
                .map(point ->
                        centers.stream().map(center -> Math.pow(measure.compute(center.getPoint(), point.getPoint()), 2))
                                .min(Double::compare).get())
                .collect(Collectors.summingDouble(x -> x));
    }
}
