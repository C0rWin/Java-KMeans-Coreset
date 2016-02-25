package univ.ml.sparse.algorithm;

import com.google.common.collect.Lists;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.util.FastMath;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.SparseWeightedKMeansPlusPlus;

import java.util.Collections;
import java.util.List;

public class SparseNonUniformCoreset implements SparseCoresetAlgorithm {
    private int k;

    private int sampleSize;

    private DistanceMeasure measure = new EuclideanDistance();

    public SparseNonUniformCoreset(final int k, final int t) {
        this.k = k;
        this.sampleSize = t;
    }

    @Override
    public List<SparseWeightableVector> takeSample(final List<SparseWeightableVector> pointset) {
        if (pointset.size() <= sampleSize) {
            final List<SparseWeightableVector> result = Lists.newArrayList(pointset);
            for (SparseWeightableVector each : result) {
                each.setWeight(1);
                each.setProbability(1.0/pointset.size());
            }
            return result;
        }
        final SparseWeightedKMeansPlusPlus clusterer = new SparseWeightedKMeansPlusPlus(k, 1);
        final List<SparseCentroidCluster> clusters = clusterer.cluster(pointset);

        double totalVariance = 0;

        for (SparseCentroidCluster cluster : clusters) {
            for (SparseWeightableVector point : cluster.getPoints()) {
                double d = point.getVector().mapMultiply(-1.0).add(cluster.getCenter().getVector()).getNorm();
                totalVariance += point.getWeight() * d * d;
            }
        } // Total clusters distance variance

        double totalSensitivity = 0;

        for (final SparseCentroidCluster cluster : clusters) {
            final RealVector center = cluster.getCenter().getVector();
            double clusterSize = 0;

            for (SparseWeightableVector vector : cluster.getPoints()) {
                clusterSize += vector.getWeight();
            }

            for (final SparseWeightableVector each : cluster.getPoints()) {
                double d = FastMath.pow(each.getVector().mapMultiply(-1.0).add(center).getNorm(), 2);
                double sensitivity = 8.0 / clusterSize + 2.0 * (each.getWeight() * d * d) / totalVariance;
                totalSensitivity += sensitivity; // not sure we need to compute this value
                each.setProbability(sensitivity * sampleSize);
//                each.setProbability(sensitivity);
            }
        }

        final List<SparseWeightableVector> allPoints = Lists.newArrayList();

        for (final SparseCentroidCluster cluster : clusters) {
            for (final SparseWeightableVector each : cluster.getPoints()) {
                each.setWeight(totalSensitivity/each.getProbability() / sampleSize);
                allPoints.add(each);
            }
        }

        Collections.sort(allPoints);
        SparseRandomSample randomSample = new SparseRandomSample(allPoints);

        return randomSample.getSampleOfSize(sampleSize);
    }
}
