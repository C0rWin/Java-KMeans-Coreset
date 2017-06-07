package univ.ml;

import com.google.common.collect.Lists;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.List;

public class CoresetEvaluator {

    private final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer;

    private final ClusteringCostFunction costFunction = new WSSE(new EuclideanDistance());

    public CoresetEvaluator(final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer) {
        this.clusterer = clusterer;
    }

    public double evalute(final CoresetAlgorithm<WeightedDoublePoint> algorithm, final List<WeightedDoublePoint> pointSet) {
        final List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(algorithm.takeSample(pointSet));

        final List<WeightedDoublePoint> centers = Lists.newArrayList();

        for (CentroidCluster<WeightedDoublePoint> cluster : clusters) {
            centers.add(new WeightedDoublePoint(cluster.getCenter().getPoint(), 1, ""));
        }

        return costFunction.getCost(centers, pointSet);
    }
}