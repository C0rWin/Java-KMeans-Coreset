package io.github.c0rwin.coreset.core.algo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class AlgorithmsTest {

    @Test
    void buildsEachAlgorithmWithDefaults() {
        assertEquals("uniform(m=10)", Algorithms.create("uniform", Map.of("m", "10")).toString());
        assertEquals("sensitivity(k=3, m=50, seeding=kmeans++(lloydIterations=0))",
                Algorithms.create("sensitivity", Map.of("k", "3", "m", "50")).toString());
        assertEquals("sensitivity(k=3, m=50, seeding=bicriteria(removeFraction=0.5))",
                Algorithms.create("sensitivity", Map.of("k", "3", "m", "50", "seeding", "bicriteria")).toString());
        assertEquals("mean(m=8, lloydIterations=20)", Algorithms.create("mean", Map.of("m", "8")).toString());
        assertEquals("mean(k=2, eps=0.2, maxM=64, lloydIterations=5)", Algorithms.create("mean",
                Map.of("k", "2", "eps", "0.2", "maxM", "64", "lloydIterations", "5")).toString());
    }

    @Test
    void rejectsBadIdsAndParametersWithClearMessages() {
        assertMessage("unknown coreset algorithm 'kmeans'", () -> Algorithms.create("kmeans", Map.of()));
        assertMessage("unknown parameter(s) [k]", () -> Algorithms.create("uniform", Map.of("m", "5", "k", "2")));
        assertMessage("missing required parameter m", () -> Algorithms.create("uniform", Map.of()));
        assertMessage("must be an integer, got 'ten'", () -> Algorithms.create("uniform", Map.of("m", "ten")));
        assertMessage("m must be >= 1", () -> Algorithms.create("uniform", Map.of("m", "0")));
        assertMessage("unknown seeding 'random'",
                () -> Algorithms.create("sensitivity", Map.of("k", "2", "m", "5", "seeding", "random")));
        assertMessage("lloydIterations applies only to seeding=kmeans++", () -> Algorithms.create("sensitivity",
                Map.of("k", "2", "m", "5", "seeding", "bicriteria", "lloydIterations", "3")));
        assertMessage("give either m, or k, eps and maxM",
                () -> Algorithms.create("mean", Map.of("m", "5", "eps", "0.1")));
        assertMessage("give either m, or k, eps and maxM", () -> Algorithms.create("mean", Map.of()));
        assertMessage("missing required parameter maxM",
                () -> Algorithms.create("mean", Map.of("k", "2", "eps", "0.1")));
    }

    private static void assertMessage(String expected, org.junit.jupiter.api.function.Executable call) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, call);
        assertTrue(e.getMessage().contains(expected), e.getMessage());
    }
}
