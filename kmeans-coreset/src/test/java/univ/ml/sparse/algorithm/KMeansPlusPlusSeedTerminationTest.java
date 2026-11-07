package univ.ml.sparse.algorithm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

/**
 * KMeansPlusPlusSeed uses an unseeded RNG, so every input here is built so that the result does not
 * depend on the random draws: once all D^2 mass is zero there is nothing left to sample.
 */
class KMeansPlusPlusSeedTerminationTest {

    private static final Duration HANG_LIMIT = Duration.ofSeconds(10);

    private static SparseWeightableVector point(double x, double y, double weight) {
        return new SparseWeightableVector(new double[] {x, y}, weight);
    }

    private static List<SparseCentroidCluster> seed(int k, List<SparseWeightableVector> points) {
        return assertTimeoutPreemptively(HANG_LIMIT, () -> new KMeansPlusPlusSeed(k).seed(points));
    }

    /** Every input point is assigned to exactly one returned cluster. */
    private static void assertPartition(List<SparseWeightableVector> points, List<SparseCentroidCluster> clusters) {
        final Map<SparseWeightableVector, Integer> seen = new IdentityHashMap<>();
        for (SparseCentroidCluster c : clusters) {
            for (SparseWeightableVector p : c.getPoints()) {
                seen.merge(p, 1, Integer::sum);
            }
        }
        assertEquals(points.size(), seen.size());
        for (SparseWeightableVector p : points) {
            assertEquals(1, seen.get(p));
        }
    }

    @Test
    void duplicatesOnlyInputYieldsOneCenter() {
        final List<SparseWeightableVector> points = new ArrayList<>();
        for (int i = 0; i < 5; i++) {
            points.add(point(1, 2, 1));
        }

        final List<SparseCentroidCluster> clusters = seed(3, points);

        assertEquals(1, clusters.size());
        assertPartition(points, clusters);
    }

    @Test
    void fewerDistinctPointsThanKYieldsOneCenterPerDistinctPoint() {
        final List<SparseWeightableVector> points = new ArrayList<>();
        for (int i = 0; i < 2; i++) {
            points.add(point(0, 0, 1));
            points.add(point(3, 0, 1));
            points.add(point(0, 4, 1));
        }

        final List<SparseCentroidCluster> clusters = seed(points.size(), points);

        assertEquals(3, clusters.size());
        assertPartition(points, clusters);
        for (SparseCentroidCluster c : clusters) {
            assertEquals(2, c.getPoints().size());
        }
    }

    @Test
    void fewerPointsThanKYieldsEveryPointAsCenter() {
        final List<SparseWeightableVector> points = List.of(point(0, 0, 1), point(5, 5, 2));

        final List<SparseCentroidCluster> clusters = seed(5, points);

        assertEquals(2, clusters.size());
        assertPartition(points, clusters);
        for (SparseCentroidCluster c : clusters) {
            assertEquals(1, c.getPoints().size());
            assertSame(c.getCenter(), c.getPoints().get(0));
        }
    }

    @Test
    void zeroWeightPointsAreNeverSampledAfterTheFirstCenter() {
        final SparseWeightableVector heavy = point(0, 0, 1);
        final List<SparseWeightableVector> points = List.of(heavy, point(1, 0, 0), point(2, 0, 0));

        final List<SparseCentroidCluster> clusters = seed(3, points);

        // The first center is uniform; afterwards only `heavy` carries D^2 mass.
        assertTrue(clusters.size() <= 2, "got " + clusters.size() + " centers");
        assertTrue(clusters.stream().anyMatch(c -> c.getCenter() == heavy));
        assertPartition(points, clusters);
    }
}
