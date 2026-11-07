package univ.ml.sparse;

import static org.junit.Assert.assertEquals;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

import univ.ml.sparse.algorithm.SparseCoresetUtility;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

/**
 * k-means cost of a weighted set is sum_p w_p * min_c ||p - c||^2.
 * Fixture: points 0 (w=1) and 4 (w=2), single center 1, so cost = 1*1 + 2*9 = 19.
 */
public class WeightedCostTest {

    private static SparseWeightableVector point(final double x, final double weight) {
        return new SparseWeightableVector(new double[] {x}, weight);
    }

    private static List<SparseWeightableVector> points() {
        return Arrays.asList(point(0, 1), point(4, 2));
    }

    private static SparseCentroidCluster clusterAt(final double x, final List<SparseWeightableVector> members) {
        final SparseCentroidCluster cluster = new SparseCentroidCluster(point(x, 1));
        members.forEach(cluster::addPoint);
        return cluster;
    }

    @Test
    public void wsseAgainstCentersIsWeighted() {
        assertEquals(19.0, new SparseWSSE().getCost(Collections.singletonList(point(1, 1)), points()), 0.0);
    }

    @Test
    public void wsseOfClustersIsWeighted() {
        assertEquals(19.0, new SparseWSSE().getCost(Collections.singletonList(clusterAt(1, points()))), 0.0);
    }

    @Test
    public void utilityEnergyIsWeighted() {
        final List<SparseCentroidCluster> centers = Collections.singletonList(clusterAt(1, points()));
        assertEquals(19.0, SparseCoresetUtility.getEnergy(centers, points()), 0.0);
        assertEquals(2 * SparseCoresetUtility.getEnergy(centers, Collections.singletonList(point(4, 1))),
                SparseCoresetUtility.getEnergy(centers, Collections.singletonList(point(4, 2))), 0.0);
    }

    @Test
    public void evaluatorMeasuresWeightedCostOfCoresetCenters() {
        // The "coreset" is {0, 10}; with k = 2, k-means++ must pick exactly those two points as centers.
        final List<SparseWeightableVector> sample = Arrays.asList(point(0, 1), point(10, 1));
        final List<SparseWeightableVector> pointSet = Arrays.asList(point(0, 1), point(4, 2), point(10, 3));
        final double cost = new SparseCoresetEvaluator(new SparseWeightedKMeansPlusPlus(2))
                .evalute(ignored -> sample, pointSet);
        // Only point 4 is off-center: nearest center 0, so cost = 2 * 16.
        assertEquals(32.0, cost, 0.0);
    }
}
