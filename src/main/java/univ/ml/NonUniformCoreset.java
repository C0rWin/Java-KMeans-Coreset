package univ.ml;

import com.google.common.collect.Lists;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;

import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

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

        final Double totalVariance = clusters.stream()
                .map(cluster ->
                        cluster.getPoints()
                                .stream()
                                .map(point -> {
                                    double d = measure.compute(cluster.getCenter().getPoint(), point.getPoint());
                                    return point.getWeight() * d * d;
                                }).collect(Collectors.summingDouble(x -> x)) // Single cluster distance variance
                ).collect(Collectors.summingDouble(x -> x));// Total clusters distance variance

        double totalSensitivity = 0;

        for (final CentroidCluster<T> cluster : clusters) {
            final double[] center = cluster.getCenter().getPoint();
            double clusterSize = cluster.getPoints().stream()
                    .map(point ->
                            point.getWeight()).collect(Collectors.summingDouble(x->x)
                    );

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
