package io.github.c0rwin.coreset.core.algo;

import static io.github.c0rwin.coreset.core.algo.TestData.point;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/**
 * Review M1: finite weights and coordinates whose sums overflow must be rejected when compress starts, with a message
 * naming the overflow, instead of surfacing later as an unrelated error or a NaN.
 */
class InputOverflowTest {

    private static final List<CoresetAlgorithm> ALGORITHMS = List.of(
            Algorithms.create("uniform", Map.of("m", "4")),
            Algorithms.create("sensitivity", Map.of("k", "2", "m", "4")),
            Algorithms.create("sensitivity", Map.of("k", "2", "m", "4", "seeding", "bicriteria")),
            Algorithms.create("mean", Map.of("m", "4")),
            Algorithms.create("mean", Map.of("k", "2", "eps", "0.5", "maxM", "4")));

    @Test
    void weightsWhoseSumOverflowsAreRejectedUpFront() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 6; i++) {
            pts.add(point(1e308, i, 1));
        }
        assertRejected(Coreset.of(pts));
    }

    @Test
    void coordinatesWhoseSquaresOverflowAreRejectedUpFront() {
        List<WeightedPoint> pts = new ArrayList<>();
        pts.add(point(0.0, 1e200, 0)); // zero weight: 0 * infinity used to turn the Lloyd cost into NaN
        for (int i = 0; i < 6; i++) {
            pts.add(point(1.0, i, 2));
        }
        assertRejected(Coreset.of(pts));
    }

    private static void assertRejected(Coreset input) {
        for (CoresetAlgorithm a : ALGORITHMS) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                    () -> a.compress(input, new SplittableRandom(1), new RecordingMetrics()), a.toString());
            assertTrue(e.getMessage().contains("overflow"), a + ": " + e.getMessage());
        }
    }
}
