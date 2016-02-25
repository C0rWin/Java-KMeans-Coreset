package univ.ml;

import com.google.common.collect.Lists;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.Collections;
import java.util.List;

public class NonUniformCoreset<T extends Sample> extends BaseCoreset<T> {

    private int k;

    private int sampleSize;

    private DistanceMeasure measure = new EuclideanDistance();

    public NonUniformCoreset(final int k, final int t) {
        this.k = k;
        this.sampleSize = t;
    }

    @Override
    public List<T> takeSample(final List<T> pointset) {
        if (pointset.size() <= sampleSize) {
            final List<T> result = Lists.newArrayList(pointset);
            for (T each : result) {
                each.setWeight(1);
                each.setProbability(1.0/pointset.size());
            }
            return result;
        }
        final WeightedKMeansPlusPlusClusterer<T> clusterer = new WeightedKMeansPlusPlusClusterer<>(k, 1);
        final List<CentroidCluster<T>> clusters = clusterer.cluster(pointset);

        double totalVariance = 0;

        for (CentroidCluster<T> cluster : clusters) {
            for (T point : cluster.getPoints()) {
                double d = measure.compute(cluster.getCenter().getPoint(), point.getPoint());
                totalVariance += point.getWeight() * d * d;
            }
        }

        double totalSensitivity = 0;

        for (final CentroidCluster<T> cluster : clusters) {
            final double[] center = cluster.getCenter().getPoint();
            double clusterSize = 0;

            for (T point : cluster.getPoints()) {
                clusterSize += point.getWeight();
            }

            for (final T each : cluster.getPoints()) {
                final double[] point = each.getPoint();
                double d = measure.compute(center, point);
                double sensitivity = 8.0 / clusterSize + 2.0 * (each.getWeight() * d * d) / totalVariance;
                totalSensitivity += sensitivity; // not sure we need to compute this value
                each.setProbability(sensitivity * sampleSize);
//                each.setProbability(sensitivity);
            }
        }

        final List<T> allPoints = Lists.newArrayList();

        for (final CentroidCluster<T> cluster : clusters) {
            for (final T each : cluster.getPoints()) {
                each.setWeight(totalSensitivity/each.getProbability() / sampleSize);
                allPoints.add(each);
            }
        }

        Collections.sort(allPoints);
        RandomSample<T> randomSample = new RandomSample<>(allPoints);

        return randomSample.getSampleOfSize(sampleSize);
    }
}
