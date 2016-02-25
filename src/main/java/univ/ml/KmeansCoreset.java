package univ.ml;

import com.google.common.collect.Lists;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.DoublePoint;

import java.util.List;

public class KmeansCoreset extends BaseCoreset<WeightedDoublePoint> {

    private int sampleSize;

    public KmeansCoreset(final int sampleSize) {
        this.sampleSize = sampleSize;
    }

    @Override
    public List<WeightedDoublePoint> takeSample(final List<WeightedDoublePoint> pointset) {
        final WeightedKMeansPlusPlusClusterer<WeightedDoublePoint> clusterer = new WeightedKMeansPlusPlusClusterer<>(sampleSize);
        final List<CentroidCluster<WeightedDoublePoint>> clusters = clusterer.cluster(pointset);

        final List<WeightedDoublePoint> result = Lists.newArrayList();

        for (CentroidCluster<WeightedDoublePoint> cluster : clusters) {
            DoublePoint center = (DoublePoint) cluster.getCenter();
            result.add(new WeightedDoublePoint(center.getPoint(), cluster.getPoints().size(), ""));
        }

        return result;
    }
}
