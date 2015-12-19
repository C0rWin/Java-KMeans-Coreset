package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;

import java.util.List;
import java.util.stream.Collectors;

public class KmeansCoreset extends BaseCoreset<WeightedDoublePoint> {

    private int sampleSize;

    public KmeansCoreset(final int sampleSize) {
        this.sampleSize = sampleSize;
    }

    @Override
    public List<WeightedDoublePoint> reduce(final List<WeightedDoublePoint> pointset) {
        final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer = new WeightedKMeansPlusPlusClusterer<>(sampleSize);
        final List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(pointset);
        return clusters.stream().map(cluster -> {
            WeightedDoublePoint center = (WeightedDoublePoint) cluster.getCenter();
            center.setWeight(cluster.getPoints().size());
            return center;
        }).collect(Collectors.toList());
    }
}
