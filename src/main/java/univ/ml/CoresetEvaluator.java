package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.List;
import java.util.stream.Collectors;

public class CoresetEvaluator {

    private final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer;

    private final DistanceMeasure measure = new EuclideanDistance();

    public CoresetEvaluator(final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer) {
        this.clusterer = clusterer;
    }

    public double evalute(final CoresetAlgorithm<WeightedDoublePoint> algorithm, final List<WeightedDoublePoint> pointSet) {
        List<WeightedDoublePoint> sample = algorithm.reduce(pointSet);
        List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(sample);

        return pointSet.stream()
                .map(point -> clusters.stream().map(cluster ->
                                Math.pow(measure.compute(cluster.getCenter().getPoint(), point.getPoint()), 2)
                        ).min(Double::compareTo).get()
                ).collect(Collectors.summingDouble(x->x));
    }
}