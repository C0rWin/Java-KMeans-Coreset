package univ.ml.sparse.algorithm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseWeightableVector;

/** BiCriteriaSeedingAlgorithm uses an unseeded RNG; the inputs force the asserted outcome. */
class BiCriteriaSeedingAlgorithmTest {

    private static final Duration HANG_LIMIT = Duration.ofSeconds(10);

    private static SparseWeightableVector point(double x) {
        return new SparseWeightableVector(new double[] {x}, 1);
    }

    @Test
    void everyRoundAddsCentersEvenAfterAFarPointWasPicked() {
        // 1023 points on 1..1023 plus one far point. Every remaining point has positive D^2 mass,
        // so each outer round must add a center. Percentile 50 removes at most about half of the
        // remaining points per round (ties add at most 2 per center), which leaves enough points
        // for at least 6 rounds: 1 initial center + >= 6.
        final List<SparseWeightableVector> points = new ArrayList<>();
        points.add(point(1e9));
        for (int x = 1; x <= 1023; x++) {
            points.add(point(x));
        }

        final List<SparseCentroidCluster> clusters =
                assertTimeoutPreemptively(HANG_LIMIT, () -> new BiCriteriaSeedingAlgorithm(1, 50).seed(points));

        assertTrue(clusters.size() >= 7, "expected at least 7 centers, got " + clusters.size());
    }

    @Test
    void duplicatesOnlyInputTerminates() {
        final List<SparseWeightableVector> points = new ArrayList<>();
        for (int i = 0; i < 50; i++) {
            points.add(point(3));
        }

        final List<SparseCentroidCluster> clusters =
                assertTimeoutPreemptively(HANG_LIMIT, () -> new BiCriteriaSeedingAlgorithm(2, 50).seed(points));

        assertEquals(50, clusters.stream().mapToInt(c -> c.getPoints().size()).sum());
    }
}
