package io.github.c0rwin.coreset.core.sampling;

import io.github.c0rwin.coreset.core.Guards;
import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Discrete distribution over {@code 0..n-1} proportional to non-negative weights, sampled with Vose's alias method:
 * O(n) construction, O(1) per draw (one {@code nextInt} and one {@code nextDouble}).
 *
 * <p>Weights need not be normalised; zero-weight indices are never drawn, and tiny or huge weights are handled by
 * rescaling with the maximum weight before normalising. Construction needs no work arrays beyond the two tables:
 * the small and large worklists are linked lists threaded through the alias table. Instances are immutable and
 * thread-safe.
 */
public final class AliasSampler {

    private static final long NEGATIVE_ZERO_BITS = Double.doubleToRawLongBits(-0.0);

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
        return ofOwned(weights.clone(), new int[weights.length]);
    }

    /**
     * {@link #of} taking ownership of both arrays, so construction allocates nothing: {@code weights} is overwritten
     * with the probability table and {@code alias} (same length, contents ignored) with the alias table. Callers
     * must not use either array afterwards. The table is the same as {@link #of} builds.
     *
     * @throws IllegalArgumentException as {@link #of}, or if the lengths differ
     */
    public static AliasSampler ofOwned(double[] weights, int[] alias) {
        final int n = weights.length;
        if (n == 0) {
            throw new IllegalArgumentException("weights must not be empty");
        }
        if (alias.length != n) {
            throw new IllegalArgumentException("alias has " + alias.length + " entries for " + n + " weights");
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
        final double[] prob = weights;
        // Stacks as linked lists through alias[]: an index's slot is free until it is finalised. Same LIFO order as
        // array stacks, so the same table.
        int small = -1;
        int large = -1;
        for (int i = 0; i < n; i++) {
            double w = weights[i];
            // -0.0 marks a zero weight; positive weights never produce it, even when they underflow or cancel.
            double p = w == 0.0 ? -0.0 : w / max * scale;
            prob[i] = p;
            if (p < 1.0) {
                alias[i] = small;
                small = i;
            } else {
                alias[i] = large;
                large = i;
            }
        }
        while (small >= 0 && large >= 0) {
            int s = small;
            small = alias[s];
            int l = large;
            large = alias[l];
            alias[s] = l;
            double rest = (prob[l] + prob[s]) - 1.0;
            prob[l] = rest;
            if (rest < 1.0) {
                alias[l] = small;
                small = l;
            } else {
                alias[l] = large;
                large = l;
            }
        }
        // Leftovers differ from 1 only by rounding; a zero weight must still never be drawn.
        while (large >= 0) {
            int l = large;
            large = alias[l];
            prob[l] = 1.0;
            alias[l] = l;
        }
        while (small >= 0) {
            int s = small;
            small = alias[s];
            if (Double.doubleToRawLongBits(prob[s]) != NEGATIVE_ZERO_BITS) {
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
     * {@code m} independent draws ({@link #next}, in order), returned sorted ascending: O(m log m) and O(m) memory
     * whatever n is. Runs of equal indices give the draw counts.
     *
     * @throws IllegalArgumentException if {@code m < 0}
     * @throws java.util.concurrent.CancellationException if the thread is interrupted (checked every 4096 draws)
     */
    public int[] sortedDraws(int m, RandomGenerator rng) {
        int[] out = new int[checkDraws(m)];
        for (int j = 0; j < m; j++) {
            if ((j & 4095) == 0) {
                Guards.checkInterrupted("sampling");
            }
            out[j] = next(rng);
        }
        Arrays.sort(out);
        return out;
    }

    /**
     * The draws of {@link #sortedDraws} for a sampler over {@code n} equal positive weights, without building it:
     * every probability is exactly 1, so a draw is {@code nextInt(n)} plus one {@code nextDouble} that never
     * redirects. Consumes the generator identically.
     *
     * @throws IllegalArgumentException if {@code n < 1} or {@code m < 0}
     */
    public static int[] sortedUniformDraws(int n, int m, RandomGenerator rng) {
        if (n < 1) {
            throw new IllegalArgumentException("n must be >= 1: " + n);
        }
        int[] out = new int[checkDraws(m)];
        for (int j = 0; j < m; j++) {
            if ((j & 4095) == 0) {
                Guards.checkInterrupted("sampling");
            }
            out[j] = rng.nextInt(n);
            rng.nextDouble();
        }
        Arrays.sort(out);
        return out;
    }

    /**
     * Multinomial sampling with replacement: performs {@code m} independent draws and returns how many times each
     * index was drawn (the counts sum to {@code m}). O(n + m).
     *
     * @throws IllegalArgumentException if {@code m < 0} or {@code probabilities} is invalid (see {@link #of})
     * @throws java.util.concurrent.CancellationException if the thread is interrupted (checked every 4096 draws)
     */
    public static int[] drawCounts(double[] probabilities, int m, RandomGenerator rng) {
        checkDraws(m);
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

    private static int checkDraws(int m) {
        if (m < 0) {
            throw new IllegalArgumentException("m must be >= 0: " + m);
        }
        return m;
    }
}
