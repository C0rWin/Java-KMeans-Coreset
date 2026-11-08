package io.github.c0rwin.coreset.core.sampling;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class AliasSamplerTest {

    /** Vose's construction with two explicit stack arrays, as before the worklists moved into the alias table. */
    private static Object[] referenceTables(double[] weights) {
        final int n = weights.length;
        double max = 0.0;
        int argMax = -1;
        for (int i = 0; i < n; i++) {
            if (weights[i] > max) {
                max = weights[i];
                argMax = i;
            }
        }
        double sum = 0.0;
        for (double w : weights) {
            sum += w / max;
        }
        final double scale = n / sum;
        double[] prob = new double[n];
        int[] alias = new int[n];
        int[] small = new int[n];
        int[] large = new int[n];
        int ns = 0;
        int nl = 0;
        for (int i = 0; i < n; i++) {
            double p = weights[i] / max * scale;
            prob[i] = p;
            if (p < 1.0) {
                small[ns++] = i;
            } else {
                large[nl++] = i;
            }
        }
        while (ns > 0 && nl > 0) {
            int s = small[--ns];
            int l = large[--nl];
            alias[s] = l;
            double rest = (prob[l] + prob[s]) - 1.0;
            prob[l] = rest;
            if (rest < 1.0) {
                small[ns++] = l;
            } else {
                large[nl++] = l;
            }
        }
        while (nl > 0) {
            int l = large[--nl];
            prob[l] = 1.0;
            alias[l] = l;
        }
        while (ns > 0) {
            int s = small[--ns];
            if (weights[s] > 0.0) {
                prob[s] = 1.0;
                alias[s] = s;
            } else {
                prob[s] = 0.0;
                alias[s] = argMax;
            }
        }
        return new Object[] {prob, alias};
    }

    private static Object field(AliasSampler s, String name) throws ReflectiveOperationException {
        Field f = AliasSampler.class.getDeclaredField(name);
        f.setAccessible(true);
        return f.get(s);
    }

    /**
     * The in-place construction must give exactly the reference tables (zero probabilities may differ only in sign,
     * which no draw can see), including underflowing and exactly cancelling weights.
     */
    @Test
    void tablesMatchTheTwoStackConstruction() throws ReflectiveOperationException {
        List<double[]> cases = new ArrayList<>();
        cases.add(new double[] {1, 1, 0, 0});
        cases.add(new double[] {1, 1, 1, 0});
        cases.add(new double[] {0, 3, 0, 1, 0});
        cases.add(new double[] {1e-320, 1e300, 1.0, 0.0, 5e-324});
        cases.add(new double[] {Double.MAX_VALUE, Double.MAX_VALUE, 0.0});
        cases.add(new double[] {-0.0, 2.0, -0.0});
        SplittableRandom rnd = new SplittableRandom(31);
        for (int t = 0; t < 400; t++) {
            double[] w = new double[1 + rnd.nextInt(60)];
            for (int i = 0; i < w.length; i++) {
                w[i] = switch (t % 4) {
                    case 0 -> rnd.nextInt(4); // small integers: exact sums, frequent exact cancellations
                    case 1 -> rnd.nextInt(3) == 0 ? 0.0 : Math.pow(1.0 - rnd.nextDouble(), -1.5);
                    case 2 -> 2.5;
                    default -> rnd.nextInt(5) == 0 ? 1e-310 * rnd.nextDouble() : rnd.nextDouble() * 1e290;
                };
            }
            w[rnd.nextInt(w.length)] = 1.0 + rnd.nextInt(3);
            cases.add(w);
        }
        for (double[] w : cases) {
            double[] before = w.clone();
            Object[] ref = referenceTables(w);
            AliasSampler s = AliasSampler.of(w);
            assertArrayEquals(before, w, "of must not modify its argument");
            double[] prob = (double[]) field(s, "prob");
            double[] refProb = (double[]) ref[0];
            for (int i = 0; i < w.length; i++) {
                assertTrue(prob[i] == refProb[i], Arrays.toString(w) + " prob " + i);
            }
            assertArrayEquals((int[]) ref[1], (int[]) field(s, "alias"), Arrays.toString(w));
        }
    }

    @Test
    void sortedDrawsAreTheSortedSequenceOfNext() {
        double[] w = {0.5, 0, 3, 1, 7, 0.25};
        AliasSampler s = AliasSampler.of(w);
        SplittableRandom a = new SplittableRandom(32);
        SplittableRandom b = new SplittableRandom(32);
        int[] expected = new int[5000];
        for (int j = 0; j < expected.length; j++) {
            expected[j] = s.next(a);
        }
        Arrays.sort(expected);
        assertArrayEquals(expected, s.sortedDraws(expected.length, b));
        assertEquals(a.nextLong(), b.nextLong());
        assertEquals(0, s.sortedDraws(0, b).length);
    }

    /** The equal-weight shortcut must replay an actual sampler bit for bit and leave the generator in the same state. */
    @Test
    void sortedUniformDrawsReplayAnEqualWeightSampler() {
        for (int n : new int[] {1, 2, 7, 1000}) {
            for (double weight : new double[] {1.0, 0.1, 3e-300, 7e300}) {
                double[] w = new double[n];
                Arrays.fill(w, weight);
                SplittableRandom a = new SplittableRandom(n * 31L);
                SplittableRandom b = new SplittableRandom(n * 31L);
                assertArrayEquals(AliasSampler.of(w).sortedDraws(3000, a), AliasSampler.sortedUniformDraws(n, 3000, b),
                        "n=" + n + " w=" + weight);
                assertEquals(a.nextLong(), b.nextLong());
            }
        }
        assertThrows(IllegalArgumentException.class, () -> AliasSampler.sortedUniformDraws(0, 1, new SplittableRandom()));
    }

    @Test
    void ofOwnedReusesTheGivenArrays() throws ReflectiveOperationException {
        double[] w = {1, 0, 2};
        int[] alias = new int[3];
        AliasSampler s = AliasSampler.ofOwned(w, alias);
        assertTrue(field(s, "prob") == w && field(s, "alias") == alias);
        assertThrows(IllegalArgumentException.class, () -> AliasSampler.ofOwned(new double[] {1, 2}, new int[1]));
    }

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
