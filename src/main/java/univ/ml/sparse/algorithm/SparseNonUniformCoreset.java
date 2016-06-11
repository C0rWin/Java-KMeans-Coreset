package univ.ml.sparse.algorithm;

import java.util.List;

import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.ml.distance.DistanceMeasure;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.util.FastMath;

import com.google.common.collect.Lists;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.SparseWeightedKMeansPlusPlus;

public class SparseNonUniformCoreset implements SparseCoresetAlgorithm {

    private static final long serialVersionUID = -390692758659255533L;

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
                each.setProbability(1.0/pointset.size());
            }
            return result;
        }

        final SparseWeightedKMeansPlusPlus clusterer = new SparseWeightedKMeansPlusPlus(k, 1);
        final List<SparseCentroidCluster> clusters = clusterer.cluster(pointset);

        double totalVariance = 0;

        double[] clusterWeights = new double[k];
        double[][] pointClusterDist = new double[k][];

        for (int i = 0; i < clusters.size(); i++) {
            final SparseCentroidCluster cluster = clusters.get(i);
            final RealVector center = cluster.getCenter().getVector();
            final int pointNo = cluster.getPoints().size();
            pointClusterDist[i] = new double[pointNo];
            for (int j = 0; j < pointNo; j++) {
                final SparseWeightableVector point = cluster.getPoints().get(j);
                double d = point.getVector().getDistance(center);

                pointClusterDist[i][j] = d;

                totalVariance += point.getWeight() * d * d;
                clusterWeights[i] += point.getWeight();
            }
        } // Total clusters distance variance

        double totalSensitivity = 0;

        for (int i = 0; i < clusters.size(); i++) {
            final SparseCentroidCluster cluster = clusters.get(i);
            for (int j = 0; j < cluster.getPoints().size(); j++) {
                final SparseWeightableVector point = cluster.getPoints().get(j);
                double d = FastMath.pow(pointClusterDist[i][j], 2);
                // Sensitivity has to be s(p) = 8 / |P_i| + 2 dist(p, c_i)^2/ (sum_i sum_j dist(p_i, c_j)^2)
                double sensitivity = 1d / clusterWeights[i];
                sensitivity += d / totalVariance;
                sensitivity *= point.getWeight();
                totalSensitivity += sensitivity;

                point.setWeight(point.getWeight()/(sampleSize * sensitivity));

                point.setProbability(sensitivity);
            }
        }

        for (final SparseCentroidCluster cluster : clusters) {
            for (final SparseWeightableVector each : cluster.getPoints()) {
                each.setProbability(each.getProbability()/totalSensitivity);
                each.setWeight(totalSensitivity*each.getWeight());
            }
        }

        SparseRandomSample randomSample = new SparseRandomSample(pointset);

        return randomSample.getSampleOfSize(sampleSize);
    }
}
