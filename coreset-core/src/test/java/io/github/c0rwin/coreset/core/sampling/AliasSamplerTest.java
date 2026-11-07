package io.github.c0rwin.coreset.core.sampling;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class AliasSamplerTest {

    /** Asserts each count is within 6 standard deviations of its binomial mean (a correct sampler fails w.p. < 1e-8). */
    private static void assertFrequencies(double[] weights, int[] counts, int m) {
        double max = 0.0;
        for (double w : weights) {
            max = Math.max(max, w);
        }
        double sum = 0.0;
        for (double w : weights) {
            sum += w / max;
        }
        for (int i = 0; i < weights.length; i++) {
            double p = weights[i] / max / sum;
            double mean = m * p;
            double sd = Math.sqrt(m * p * (1 - p));
            assertTrue(Math.abs(counts[i] - mean) <= 6 * sd + 1e-9,
                    "index " + i + ": count " + counts[i] + ", expected " + mean + " +- " + 6 * sd);
        }
    }

    @Test
    void frequenciesMatchNonNormalisedWeights() {
        double[] w = {1, 2, 3, 4, 0, 10};
        int m = 200_000;
        int[] counts = AliasSampler.drawCounts(w, m, new SplittableRandom(1));
        assertFrequencies(w, counts, m);
        assertEquals(0, counts[4]);
    }

    @Test
    void chiSquareOverManyCategoriesWithZeros() {
        SplittableRandom gen = new SplittableRandom(2);
        int n = 1000;
        double[] w = new double[n];
        int positive = 0;
        for (int i = 0; i < n; i++) {
            w[i] = i % 7 == 0 ? 0.0 : gen.nextDouble() * (i % 3 == 0 ? 100.0 : 1.0);
            positive += w[i] > 0 ? 1 : 0;
        }
        int m = 1_000_000;
        int[] counts = AliasSampler.drawCounts(w, m, new SplittableRandom(3));
        double sum = 0.0;
        for (double v : w) {
            sum += v;
        }
        double chi2 = 0.0;
        for (int i = 0; i < n; i++) {
            if (w[i] == 0.0) {
                assertEquals(0, counts[i], "zero-weight index drawn");
                continue;
            }
            double e = m * w[i] / sum;
            chi2 += (counts[i] - e) * (counts[i] - e) / e;
        }
        int df = positive - 1;
        // chi2 has mean df and sd sqrt(2 df); 8 sd is far beyond any plausible fluctuation.
        assertTrue(chi2 < df + 8 * Math.sqrt(2.0 * df), "chi2=" + chi2 + " df=" + df);
    }

    @Test
    void zeroWeightsAreNeverDrawn() {
        double[] w = new double[64];
        w[5] = 1e-12;
        w[40] = 3e-12;
        AliasSampler s = AliasSampler.of(w);
        SplittableRandom rng = new SplittableRandom(4);
        int[] counts = new int[w.length];
        for (int j = 0; j < 100_000; j++) {
            counts[s.next(rng)]++;
        }
        assertEquals(100_000, counts[5] + counts[40]);
        assertFrequencies(w, counts, 100_000);
    }

    @Test
    void singlePositiveWeightIsAlwaysDrawn() {
        int[] counts = AliasSampler.drawCounts(new double[] {0, 0, 5, 0}, 1000, new SplittableRandom(5));
        assertEquals(1000, counts[2]);
    }

    @Test
    void extremeMagnitudesAreRescaled() {
        int m = 100_000;
        double[] huge = {Double.MAX_VALUE, Double.MAX_VALUE};
        assertFrequencies(huge, AliasSampler.drawCounts(huge, m, new SplittableRandom(6)), m);
        double[] subnormal = {Double.MIN_VALUE, 3 * Double.MIN_VALUE};
        assertFrequencies(subnormal, AliasSampler.drawCounts(subnormal, m, new SplittableRandom(7)), m);
        double[] mixed = {1e-300, 1.0, 1.0};
        int[] counts = AliasSampler.drawCounts(mixed, m, new SplittableRandom(8));
        assertEquals(0, counts[0]); // probability 5e-301 per draw
        assertFrequencies(mixed, counts, m);
    }

    @Test
    void countsSumToM() {
        int[] counts = AliasSampler.drawCounts(new double[] {0.2, 0.0, 0.7}, 12_345, new SplittableRandom(9));
        assertEquals(12_345, counts[0] + counts[1] + counts[2]);
        assertEquals(0, AliasSampler.drawCounts(new double[] {1.0}, 0, new SplittableRandom(9))[0]);
    }

    @Test
    void rejectsInvalidInput() {
        SplittableRandom rng = new SplittableRandom(10);
        assertThrows(IllegalArgumentException.class, () -> AliasSampler.of(new double[0]));
        assertThrows(IllegalArgumentException.class, () -> AliasSampler.of(new double[] {0, 0}));
        assertThrows(IllegalArgumentException.class, () -> AliasSampler.of(new double[] {1, -1}));
        assertThrows(IllegalArgumentException.class, () -> AliasSampler.of(new double[] {1, Double.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> AliasSampler.of(new double[] {1, Double.POSITIVE_INFINITY}));
        assertThrows(IllegalArgumentException.class, () -> AliasSampler.drawCounts(new double[] {1}, -1, rng));
    }
}
