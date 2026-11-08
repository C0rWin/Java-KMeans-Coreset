package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.Arrays;
import java.util.Objects;

/**
 * A coreset built by i.i.d. importance sampling, with the provenance needed for per-query error estimates: coreset
 * point {@code j} was drawn {@code drawCounts[j]} times out of {@code draws}, each time with probability
 * {@code probabilities[j]}, so its weight is {@code w_j * drawCounts[j] / (draws * probabilities[j])} for input weight
 * {@code w_j}.
 *
 * <p>When {@code draws == 0} no sampling happened (the input was small enough to keep, or had no positive weight): the
 * coreset is exact and both arrays are empty.
 *
 * @param coreset           sampled points (input order, duplicates merged) and the input offset
 * @param drawCounts        per coreset point, times drawn; accessor returns a copy
 * @param probabilities     per coreset point, per-draw selection probability; accessor returns a copy
 * @param draws             number of draws m, or 0 for an exact coreset
 * @param totalSensitivity  {@code S = sum_p s_p} over the input, or 0 for an exact coreset
 */
public record SampledCoreset(Coreset coreset, int[] drawCounts, double[] probabilities, int draws,
                             double totalSensitivity) {

    public SampledCoreset {
        Objects.requireNonNull(coreset, "coreset");
        drawCounts = drawCounts.clone();
        probabilities = probabilities.clone();
        int expected = draws == 0 ? 0 : coreset.size();
        if (draws < 0 || drawCounts.length != expected || probabilities.length != expected) {
            throw new IllegalArgumentException("provenance arrays must have one entry per coreset point (none when "
                    + "draws == 0): draws=" + draws + ", points=" + coreset.size() + ", counts=" + drawCounts.length
                    + ", probabilities=" + probabilities.length);
        }
    }

    static SampledCoreset exact(Coreset coreset) {
        return new SampledCoreset(coreset, new int[0], new double[0], 0, 0.0);
    }

    /** True when the coreset is the (unsampled) input and its costs are exact. */
    public boolean exact() {
        return draws == 0;
    }

    @Override
    public int[] drawCounts() {
        return drawCounts.clone();
    }

    @Override
    public double[] probabilities() {
        return probabilities.clone();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof SampledCoreset s
                && coreset.equals(s.coreset)
                && Arrays.equals(drawCounts, s.drawCounts)
                && Arrays.equals(probabilities, s.probabilities)
                && draws == s.draws
                && Double.compare(totalSensitivity, s.totalSensitivity) == 0;
    }

    @Override
    public int hashCode() {
        return Objects.hash(coreset, Arrays.hashCode(drawCounts), Arrays.hashCode(probabilities), draws,
                totalSensitivity);
    }

    @Override
    public String toString() {
        return "SampledCoreset[points=" + coreset.size() + ", offset=" + coreset.offset() + ", draws=" + draws
                + ", totalSensitivity=" + totalSensitivity + "]";
    }
}
