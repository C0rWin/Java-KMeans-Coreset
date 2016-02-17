package univ.ml;

import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.List;
import java.util.stream.Collectors;

public class KmeansCoreset extends BaseCoreset<WeightedDoublePoint> {

    private int sampleSize;

    public KmeansCoreset(final int sampleSize) {
        this.sampleSize = sampleSize;
    }

    @Override
    public List<WeightedDoublePoint> takeSample(final List<WeightedDoublePoint> pointset) {
        final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer = new WeightedKMeansPlusPlusClusterer<>(sampleSize);
        final List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(pointset);
        return clusters.stream().map(cluster -> {
            DoublePoint center = (DoublePoint) cluster.getCenter();
            return new WeightedDoublePoint(center.getPoint(), cluster.getPoints().size(), "");
        }).collect(Collectors.toList());
    }
}
