package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.List;

public class CoresetEvaluator {

    private final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer;

    private final DistanceMeasure measure = new EuclideanDistance();

    public CoresetEvaluator(final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer) {
        this.clusterer = clusterer;
    }

    public double evalute(final CoresetAlgorithm<WeightedDoublePoint> algorithm, final List<WeightedDoublePoint> pointSet) {
        final List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(algorithm.reduce(pointSet));
        final ClusteringCostFunction costFunction = new WSSE(measure);

        return costFunction.getCost(clusters);
    }
}