package univ.ml.sparse.algorithm;

import com.google.common.collect.Lists;
import org.apache.commons.math3.linear.RealVector;
import univ.ml.sparse.CTRandomSample;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

import java.util.List;

public class SparseNonUniformCoreset implements SparseCoresetAlgorithm {

    private static final long serialVersionUID = -390692758659255533L;

    private int k;

    private int sampleSize;

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

        final BiCriteriaAlgorithm clusterer = new BiCriteriaAlgorithm(k, 0.5);
//        final SparseWeightedKMeansPlusPlus clusterer = new SparseWeightedKMeansPlusPlus(k, 1);
        final List<SparseCentroidCluster> clusters = clusterer.takeSample(pointset);
//        final List<SparseCentroidCluster> clusters = clusterer.cluster(pointset);

        double totalVariance = 0;

        double[] clusterWeights = new double[clusters.size()];
        double[][] pointClusterDist = new double[clusters.size()][];

        for (int i = 0; i < clusters.size(); i++) {
            final SparseCentroidCluster cluster = clusters.get(i);
            final RealVector center = cluster.getCenter().getVector();
            final int pointNo = cluster.getPoints().size();
            pointClusterDist[i] = new double[pointNo];
            for (int j = 0; j < pointNo; j++) {
                final SparseWeightableVector point = cluster.getPoints().get(j);
                double d = point.getVector().getDistance(center);

                pointClusterDist[i][j] = d * d;

                totalVariance       += point.getWeight() * pointClusterDist[i][j];
                clusterWeights[i]   += point.getWeight();
            }
        } // Total clusters distance variance

//        double totalSensitivity = 0;

        for (int i = 0; i < clusters.size(); i++) {
            final SparseCentroidCluster cluster = clusters.get(i);
            for (int j = 0; j < cluster.getPoints().size(); j++) {
                final SparseWeightableVector point = cluster.getPoints().get(j);

                // Sensitivity has to be s(p) = (8 * w_i) / |P_i| + 2 * w_i * dist(p_i, c_i)^2/ (sum_i sum_j w_i * dist(p_i, c_j)^2)

                double sensitivity = 8d / clusterWeights[i];
                sensitivity += 2d * pointClusterDist[i][j] / totalVariance;
                sensitivity *= point.getWeight();


//                totalSensitivity += sensitivity;

                point.setWeight((8*clusters.size()+2)*point.getWeight()/(sampleSize * sensitivity));
                point.setProbability(sensitivity/(8*clusters.size()+2));
            }
        }

//        System.out.println("XXX@@@: Total sensitivity = " + totalSensitivity + " $$$");
//
//        for (final SparseCentroidCluster cluster : clusters) {
//            for (final SparseWeightableVector each : cluster.getPoints()) {
//                each.setProbability(each.getProbability()/totalSensitivity);
//                each.setWeight(totalSensitivity*each.getWeight());
//            }
//        }

        CTRandomSample randomSample = new CTRandomSample(pointset);
        return randomSample.getSampleOfSize(sampleSize);
    }
}
