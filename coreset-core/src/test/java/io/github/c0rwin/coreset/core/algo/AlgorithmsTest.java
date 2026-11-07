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
        String alpha3 = "standard(alpha=" + SensitivityBound.kMeansPlusPlusAlpha(3) + ")";
        assertEquals("sensitivity(k=3, m=50, seeding=kmeans++(lloydIterations=0), bound=" + alpha3 + ")",
                Algorithms.create("sensitivity", Map.of("k", "3", "m", "50")).toString());
        assertEquals("sensitivity(k=3, m=50, seeding=bicriteria(removeFraction=0.5), bound=" + alpha3 + ")",
                Algorithms.create("sensitivity", Map.of("k", "3", "m", "50", "seeding", "bicriteria")).toString());
        assertEquals("sensitivity(k=3, m=50, seeding=kmeans++(lloydIterations=0), bound=legacy)",
                Algorithms.create("sensitivity", Map.of("k", "3", "m", "50", "bound", "legacy")).toString());
        assertEquals("sensitivity(k=3, m=50, seeding=kmeans++(lloydIterations=0), bound=standard(alpha=4.0))",
                Algorithms.create("sensitivity", Map.of("k", "3", "m", "50", "alpha", "4")).toString());
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
        assertMessage("unknown bound 'tight'",
                () -> Algorithms.create("sensitivity", Map.of("k", "2", "m", "5", "bound", "tight")));
        assertMessage("alpha applies only to bound=standard",
                () -> Algorithms.create("sensitivity", Map.of("k", "2", "m", "5", "bound", "legacy", "alpha", "3")));
        assertMessage("alpha must be finite and >= 1",
                () -> Algorithms.create("sensitivity", Map.of("k", "2", "m", "5", "alpha", "0.5")));
        assertMessage("give either m, or k, eps and maxM",
                () -> Algorithms.create("mean", Map.of("m", "5", "eps", "0.1")));
        assertMessage("give either m, or k, eps and maxM", () -> Algorithms.create("mean", Map.of()));
        assertMessage("missing required parameter maxM",
                () -> Algorithms.create("mean", Map.of("k", "2", "eps", "0.1")));
    }

    /** Review H4: a remote caller must not be able to ask for 2^31 centers, draws or Lloyd steps. */
    @Test
    void rejectsParametersAboveTheDefaultLimits() {
        assertMessage("m must be <= 1000000", () -> Algorithms.create("uniform", Map.of("m", "1000001")));
        assertMessage("m must be <= 1000000",
                () -> Algorithms.create("sensitivity", Map.of("k", "2", "m", "2147483647")));
        assertMessage("k must be <= 100000",
                () -> Algorithms.create("sensitivity", Map.of("k", "100001", "m", "5")));
        assertMessage("lloydIterations must be <= 1000", () -> Algorithms.create("sensitivity",
                Map.of("k", "2", "m", "5", "lloydIterations", "2147483647")));
        assertMessage("m must be <= 1000000", () -> Algorithms.create("mean", Map.of("m", "1000001")));
        assertMessage("lloydIterations must be <= 1000",
                () -> Algorithms.create("mean", Map.of("m", "5", "lloydIterations", "1001")));
        assertMessage("maxM must be <= 1000000",
                () -> Algorithms.create("mean", Map.of("k", "2", "eps", "0.1", "maxM", "1000001")));
        assertMessage("k must be <= 100000",
                () -> Algorithms.create("mean", Map.of("k", "100001", "eps", "0.1", "maxM", "5")));
        // The limits themselves are accepted.
        Algorithms.create("sensitivity", Map.of("k", "100000", "m", "1000000", "lloydIterations", "1000"));
    }

    private static void assertMessage(String expected, org.junit.jupiter.api.function.Executable call) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, call);
        assertTrue(e.getMessage().contains(expected), e.getMessage());
    }
}
