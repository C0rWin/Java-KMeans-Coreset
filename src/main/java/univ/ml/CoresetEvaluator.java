package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.List;
import java.util.stream.Collectors;

public class CoresetEvaluator {

    private final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer;

    private final ClusteringCostFunction costFunction = new WSSE(new EuclideanDistance());

    public CoresetEvaluator(final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer) {
        this.clusterer = clusterer;
    }

    public double evalute(final CoresetAlgorithm<WeightedDoublePoint> algorithm, final List<WeightedDoublePoint> pointSet) {
        final List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(algorithm.reduce(pointSet));

        final List<WeightedDoublePoint> centers = clusters.stream()
                .map(cluster -> new WeightedDoublePoint(cluster.getCenter().getPoint(), 1, ""))
                .collect(Collectors.toList());

        return costFunction.getCost(centers, pointSet);
    }
}