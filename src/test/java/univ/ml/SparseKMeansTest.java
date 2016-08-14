package univ.ml;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.commons.math3.util.FastMath;
import org.junit.Test;
import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

import java.util.List;
import java.util.Map;
import java.util.Random;

public class SparseKMeansTest {

    @Test
    public void kmeans() {

        List<SparseWeightableVector> regularPoints = Lists.newArrayList();
        List<SparseWeightableVector> weightedPoints = Lists.newArrayList();

        final Random rnd = new Random(System.currentTimeMillis());

        for (int i = 0; i < 100; i++) {
            Map<Integer, Double> coordinates = Maps.newHashMap();
            for (int j = 0; j < 3; j++) {
                coordinates.put(j, 100*rnd.nextDouble());
            }
            regularPoints.add(new SparseWeightableVector(coordinates, 1d, 3));
            weightedPoints.add(new SparseWeightableVector(coordinates, 100d * rnd.nextDouble(), 3));
        }

        final SparseWeightedKMeansPlusPlus kMeans = new SparseWeightedKMeansPlusPlus(20);
        final List<SparseCentroidCluster> regularClusters = kMeans.cluster(regularPoints);
        final List<SparseCentroidCluster> weightedClusters = kMeans.cluster(weightedPoints);

        double regularEnergy = 0d;

        for (SparseCentroidCluster cluster : regularClusters) {
            for (SparseWeightableVector vector : cluster.getPoints()) {
                regularEnergy += FastMath.pow(cluster.getCenter().getVector().getDistance(vector.getVector()), 2);
            }
        }
        System.out.println(regularClusters);
        System.out.println(regularClusters.get(0).getPoints());

        double weightedEnergy = 0d;
        for (SparseCentroidCluster cluster : weightedClusters) {
            for (SparseWeightableVector vector : cluster.getPoints()) {
                weightedEnergy += FastMath.pow(cluster.getCenter().getVector().getDistance(vector.getVector()), 2);
            }
        }

        System.out.println(weightedClusters);
        System.out.println(weightedClusters.get(0).getPoints());

        System.out.println("Non weighted energy = " + regularEnergy);
        System.out.println("Weighted energy = " + weightedEnergy);
    }
}
