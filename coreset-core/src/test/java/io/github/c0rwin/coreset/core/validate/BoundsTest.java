package io.github.c0rwin.coreset.core.validate;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class BoundsTest {

    @Test
    void errorBudgetComposesBackToTheTarget() {
        ErrorBudget b = ErrorBudget.forStream(0.1, 0.01, 16_000, 500, false);
        assertEquals(5, b.levels()); // 32 batches
        assertEquals(31, b.reduceOps()); // 32 - bitCount(32)
        assertEquals(1.1, Math.pow(1 + b.epsPerLevel(), b.levels()), 1e-12);
        assertEquals(0.01 / 31, b.deltaPerReduce(), 1e-18);
        assertEquals(Math.pow(0.1 / b.epsPerLevel(), 2), b.sampleSizeFactor(), 1e-12);

        ErrorBudget withFinal = ErrorBudget.forStream(0.1, 0.01, 16_001, 500, true);
        assertEquals(6 + 1, withFinal.levels()); // 33 batches -> ceil(log2 33) = 6, plus the final reduce
        assertEquals(33 - 2 + 1, withFinal.reduceOps());
    }

    @Test
    void errorBudgetWithoutReductionsKeepsTheTarget() {
        ErrorBudget b = ErrorBudget.forStream(0.2, 0.05, 300, 500, false);
        assertEquals(0, b.levels());
        assertEquals(0, b.reduceOps());
        assertEquals(0.2, b.epsPerLevel());
        assertEquals(0.05, b.deltaPerReduce());
    }

    /** The worked example of research notes 6a: S = 40, eps = 0.1, delta = 0.01 needs about 4.2e5 draws. */
    @Test
    void singleQuerySampleSizeMatchesHoeffding() {
        assertEquals(6 * 32.0 + 4 * 5, TheoryBounds.kMeansTotalSensitivity(5, 32.0)); // Lemma 2.2: 6 alpha + 4 k
        assertEquals((long) Math.ceil(1600 / 0.02 * Math.log(200)), TheoryBounds.singleQuerySampleSize(40, 0.1, 0.01));
        assertEquals(423_866, TheoryBounds.singleQuerySampleSize(40, 0.1, 0.01));
    }

    @Test
    void allQueriesSampleSizeScalesWithPseudoDimension() {
        double dPrime = TheoryBounds.kMeansPseudoDimension(10, 8); // 10 * 8 * 3
        assertEquals(240.0, dPrime, 1e-12);
        long m = TheoryBounds.allQueriesSampleSize(40, 0.1, 0.01, dPrime, 1.0);
        assertEquals((long) Math.ceil(1600 / 0.01 * (240 + Math.log(100))), m);
    }

    @Test
    void rejectsInvalidArguments() {
        assertThrows(IllegalArgumentException.class, () -> ErrorBudget.forStream(0, 0.1, 10, 5, false));
        assertThrows(IllegalArgumentException.class, () -> ErrorBudget.forStream(0.1, 1.0, 10, 5, false));
        assertThrows(IllegalArgumentException.class, () -> ErrorBudget.forStream(0.1, 0.1, 0, 5, false));
        assertThrows(IllegalArgumentException.class, () -> TheoryBounds.singleQuerySampleSize(0, 0.1, 0.1));
        assertThrows(IllegalArgumentException.class, () -> TheoryBounds.allQueriesSampleSize(1, 0.1, 0.1, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> TheoryBounds.kMeansTotalSensitivity(0, 2.0));
        assertThrows(IllegalArgumentException.class, () -> TheoryBounds.kMeansTotalSensitivity(3, 0.9));
    }
}
