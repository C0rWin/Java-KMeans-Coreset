package io.github.c0rwin.coreset.core.algo;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.CancellationException;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;

/**
 * Review H4: long loops (seeding, Lloyd, sampling) must notice an interrupted thread and stop with a clear exception,
 * leaving the interrupt flag set, so a closing node can stop its workers.
 */
class InterruptionTest {

    private static final Coreset INPUT = TestData.mixture(500, 4, 3, true, 1);

    @Test
    void everyAlgorithmStopsOnAnInterruptedThread() {
        List<CoresetAlgorithm> algorithms = List.of(
                Algorithms.create("uniform", Map.of("m", "50")),
                Algorithms.create("sensitivity", Map.of("k", "4", "m", "50")),
                Algorithms.create("sensitivity", Map.of("k", "4", "m", "50", "lloydIterations", "5")),
                Algorithms.create("sensitivity", Map.of("k", "4", "m", "50", "seeding", "bicriteria")),
                Algorithms.create("mean", Map.of("m", "50")),
                Algorithms.create("mean", Map.of("k", "4", "eps", "0.2", "maxM", "64")));
        for (CoresetAlgorithm a : algorithms) {
            assertCancelled(() -> a.compress(INPUT, new SplittableRandom(2)), a.toString());
        }
    }

    @Test
    void kmeansBuildingBlocksStopOnAnInterruptedThread() {
        assertCancelled(() -> KMeansPlusPlus.seed(INPUT.points(), 4, new SplittableRandom(3)), "k-means++");
        assertCancelled(() -> WeightedLloyd.cluster(INPUT.points(), 4, 10, new SplittableRandom(4)), "lloyd");
        assertCancelled(() -> Seeding.bicriteria().cluster(INPUT.points(), 4, new SplittableRandom(5),
                MetricsSink.NOOP), "bicriteria");
    }

    private static void assertCancelled(Executable call, String what) {
        Thread.currentThread().interrupt();
        try {
            CancellationException e = assertThrows(CancellationException.class, call, what);
            assertTrue(e.getMessage().contains("interrupted"), what + ": " + e.getMessage());
            assertTrue(Thread.currentThread().isInterrupted(), what + " must keep the interrupt flag");
        } finally {
            Thread.interrupted();
        }
    }
}
