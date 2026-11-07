package io.github.c0rwin.coreset.core.sampling;

import io.github.c0rwin.coreset.core.Guards;
import java.util.random.RandomGenerator;

/**
 * Discrete distribution over {@code 0..n-1} proportional to non-negative weights, sampled with Vose's alias method:
 * O(n) construction, O(1) per draw (one {@code nextInt} and one {@code nextDouble}).
 *
 * <p>Weights need not be normalised; zero-weight indices are never drawn, and tiny or huge weights are handled by
 * rescaling with the maximum weight before normalising. Instances are immutable and thread-safe.
 */
public final class AliasSampler {

    private final double[] prob;
    private final int[] alias;

    private AliasSampler(double[] prob, int[] alias) {
        this.prob = prob;
        this.alias = alias;
    }

    /**
     * @throws IllegalArgumentException if {@code weights} is empty, contains a negative or non-finite value, or has
     *     no positive value
     */
    public static AliasSampler of(double[] weights) {
        final int n = weights.length;
        if (n == 0) {
            throw new IllegalArgumentException("weights must not be empty");
        }
        double max = 0.0;
        int argMax = -1;
        for (int i = 0; i < n; i++) {
            double w = weights[i];
            if (!(w >= 0.0) || w == Double.POSITIVE_INFINITY) {
                throw new IllegalArgumentException("weight " + i + " must be finite and >= 0: " + w);
            }
            if (w > max) {
                max = w;
                argMax = i;
            }
        }
        if (argMax < 0) {
            throw new IllegalArgumentException("at least one weight must be positive");
        }
        // Dividing by the max first keeps the sum finite and away from subnormals.
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
        // Leftovers differ from 1 only by rounding; a zero weight must still never be drawn.
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
        return new AliasSampler(prob, alias);
    }

    /** Number of outcomes {@code n}. */
    public int size() {
        return prob.length;
    }

    /** Draws one index with probability {@code weights[i] / sum(weights)}. */
    public int next(RandomGenerator rng) {
        int i = rng.nextInt(prob.length);
        return rng.nextDouble() < prob[i] ? i : alias[i];
    }

    /**
     * Multinomial sampling with replacement: performs {@code m} independent draws and returns how many times each
     * index was drawn (the counts sum to {@code m}). O(n + m).
     *
     * @throws IllegalArgumentException if {@code m < 0} or {@code probabilities} is invalid (see {@link #of})
     * @throws java.util.concurrent.CancellationException if the thread is interrupted (checked every 4096 draws)
     */
    public static int[] drawCounts(double[] probabilities, int m, RandomGenerator rng) {
        if (m < 0) {
            throw new IllegalArgumentException("m must be >= 0: " + m);
        }
        AliasSampler sampler = of(probabilities);
        int[] counts = new int[probabilities.length];
        for (int j = 0; j < m; j++) {
            if ((j & 4095) == 0) {
                Guards.checkInterrupted("sampling");
            }
            counts[sampler.next(rng)]++;
        }
        return counts;
    }
}
