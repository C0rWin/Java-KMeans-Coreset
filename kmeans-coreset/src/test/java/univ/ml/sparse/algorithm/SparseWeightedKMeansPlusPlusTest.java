package univ.ml.sparse.algorithm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.linear.RealVector;
import org.junit.jupiter.api.Test;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

/**
 * The clusterer seeds from an unseeded RNG, so each test either forces its outcome through the
 * input or states the probability of a spurious result.
 */
class SparseWeightedKMeansPlusPlusTest {

    private static final Duration HANG_LIMIT = Duration.ofSeconds(30);

    private static SparseWeightableVector point(double x, double y, double weight) {
        return new SparseWeightableVector(new double[] {x, y}, weight);
    }

    private static List<SparseCentroidCluster> cluster(int k, List<SparseWeightableVector> points) {
        return assertTimeoutPreemptively(HANG_LIMIT, () -> new SparseWeightedKMeansPlusPlus(k).cluster(points));
    }

    /** Clusters partition the input (by identity), are non-empty and have finite centers. */
    private static void assertWellFormed(List<SparseWeightableVector> points, List<SparseCentroidCluster> clusters) {
        final Map<SparseWeightableVector, Integer> seen = new IdentityHashMap<>();
        for (SparseCentroidCluster c : clusters) {
            final RealVector center = c.getCenter().getVector();
            assertFalse(center.isNaN() || center.isInfinite(), "non-finite center " + center);
            assertFalse(c.getPoints().isEmpty(), "empty cluster at " + center);
            for (SparseWeightableVector p : c.getPoints()) {
                seen.merge(p, 1, Integer::sum);
            }
        }
        assertEquals(points.size(), seen.size());
        for (SparseWeightableVector p : points) {
            assertEquals(1, seen.get(p));
        }
    }

    private static double cost(List<SparseCentroidCluster> clusters) {
        double cost = 0;
        for (SparseCentroidCluster c : clusters) {
            for (SparseWeightableVector p : c.getPoints()) {
                final double d = c.getCenter().getVector().getDistance(p);
                cost += p.getWeight() * d * d;
            }
        }
        return cost;
    }

    @Test
    void emptyClustersNeverSurfaceAsNaNCenters() {
        // With k close to n Lloyd regularly empties a cluster. On this layout (seed 1543) the previous
        // implementation returned an empty cluster with a NaN center in ~2% of runs (42 of 2000), so
        // 1000 runs miss it with probability 0.979^1000 < 1e-9.
        final Random rng = new Random(1543);
        final List<SparseWeightableVector> points = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            points.add(point(rng.nextInt(1000), rng.nextInt(1000), 1 + rng.nextInt(5)));
        }

        for (int run = 0; run < 1000; run++) {
            final List<SparseCentroidCluster> clusters = cluster(60, points);
            assertWellFormed(points, clusters);
            assertTrue(Double.isFinite(cost(clusters)));
        }
    }

    @Test
    void fewerDistinctPointsThanKYieldsOneClusterPerDistinctPoint() {
        final List<SparseWeightableVector> points = new ArrayList<>();
        for (int i = 0; i < 1000; i++) {
            points.add(point(0, 0, 1));
        }
        points.add(point(7, 0, 1));

        final List<SparseCentroidCluster> clusters = cluster(3, points);

        assertEquals(2, clusters.size());
        assertWellFormed(points, clusters);
        assertEquals(0d, cost(clusters));
    }

    @Test
    void fewerPointsThanKYieldsOneClusterPerPoint() {
        final List<SparseWeightableVector> points = List.of(point(0, 0, 1), point(5, 5, 2));

        final List<SparseCentroidCluster> clusters = cluster(5, points);

        assertEquals(2, clusters.size());
        assertWellFormed(points, clusters);
    }

    @Test
    void emptyInputIsRejected() {
        assertThrows(NumberIsTooSmallException.class, () -> new SparseWeightedKMeansPlusPlus(2).cluster(new ArrayList<>()));
    }

    @Test
    void centerIsTheWeightedMeanAndPointsAreTheInputObjects() {
        final SparseWeightableVector a = point(0, 0, 1);
        final SparseWeightableVector b = point(4, 0, 3);

        final List<SparseCentroidCluster> clusters = cluster(1, List.of(a, b));

        assertEquals(1, clusters.size());
        final RealVector center = clusters.get(0).getCenter().getVector();
        assertEquals(3d, center.getEntry(0), 1e-12);
        assertEquals(0d, center.getEntry(1), 1e-12);
        assertSame(a, clusters.get(0).getPoints().get(0));
        assertSame(b, clusters.get(0).getPoints().get(1));
    }

    @Test
    void zeroWeightPointsDoNotProduceNaNCenters() {
        final List<SparseWeightableVector> points = List.of(point(0, 0, 1), point(10, 0, 0), point(11, 0, 0));

        final List<SparseCentroidCluster> clusters = cluster(2, points);

        assertWellFormed(points, clusters);
        assertEquals(0d, cost(clusters));
    }

    @Test
    void wellSeparatedGroupsAreRecovered() {
        // Groups 1e6 apart with unit spread: kmeans++ puts two seeds in one group with probability
        // below 1e-11, so the optimal partition is found.
        final List<SparseWeightableVector> points = new ArrayList<>();
        final double[][] offsets = {{0, 0}, {1e6, 0}, {0, 1e6}};
        for (double[] o : offsets) {
            points.add(point(o[0] - 1, o[1], 1));
            points.add(point(o[0] + 1, o[1], 1));
            points.add(point(o[0], o[1] + 1, 2));
        }

        final List<SparseCentroidCluster> clusters = cluster(3, points);

        assertEquals(3, clusters.size());
        assertWellFormed(points, clusters);
        for (SparseCentroidCluster c : clusters) {
            assertEquals(3, c.getPoints().size());
        }
        // Per group: mean (0, 0.5) gives 1.25 + 1.25 + 2 * 0.25.
        assertEquals(3 * 3.0, cost(clusters), 1e-6);
    }
}
