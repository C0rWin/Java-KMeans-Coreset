package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Sensitivity (importance) sampling in the Feldman-Langberg framework. A {@link Seeding} gives a rough clustering B, a
 * {@link SensitivityBound} turns it into per-point bounds {@code s_p} (by default
 * {@link SensitivityBound#standard} with {@code alpha = }{@link SensitivityBound#DEFAULT_ALPHA}),
 * m points are drawn i.i.d. with {@code q_p = s_p / S} where {@code S = sum_p s_p} is the actual total, and a point
 * drawn {@code c_p} times gets weight {@code w_p * c_p / (m * q_p)}. This makes {@code cost(C, Q)} unbiased for every
 * Q and the total weight unbiased, whatever the bound; the bound only has to dominate the true sensitivities for
 * Hoeffding-type guarantees. Inputs with at most m points are returned unchanged; the offset passes through.
 *
 * <p>Time: the seeding (O(n k nnz) for k-means++) plus O(n + m) for sensitivities and sampling: two passes over the
 * points for the bounds and one to build the output. Stateless and thread-safe.
 */
public final class SensitivityCoreset implements CoresetAlgorithm {

    private final int k;
    private final int m;
    private final Seeding seeding;
    private final SensitivityBound bound;

    /**
     * Uses {@link SensitivityBound#standard} with {@link SensitivityBound#DEFAULT_ALPHA}; pass
     * {@code standard(kMeansPlusPlusAlpha(k))} for a bound that holds with probability 1/2 per seeding.
     *
     * @throws IllegalArgumentException if {@code k < 1} or {@code m < 1}
     */
    public SensitivityCoreset(int k, int m, Seeding seeding) {
        this(k, m, seeding, SensitivityBound.standard(SensitivityBound.DEFAULT_ALPHA));
    }

    /** @throws IllegalArgumentException if {@code k < 1} or {@code m < 1} */
    public SensitivityCoreset(int k, int m, Seeding seeding, SensitivityBound bound) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        if (m < 1) {
            throw new IllegalArgumentException("m must be >= 1: " + m);
        }
        this.k = k;
        this.m = m;
        this.seeding = Objects.requireNonNull(seeding, "seeding");
        this.bound = Objects.requireNonNull(bound, "bound");
    }

    @Override
    public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
        return compressWithProvenance(input, rng, metrics).coreset();
    }

    /** Same coreset as {@link #compress} for the same generator state, plus the sampling provenance. */
    public SampledCoreset compressWithProvenance(Coreset input, RandomGenerator rng, MetricsSink metrics) {
        try (Span ignored = metrics.span(MetricNames.SENSITIVITY_COMPRESS)) {
            PointBlock points = input.block();
            final int n = points.size();
            if (n <= m) {
                metrics.count(MetricNames.SENSITIVITY_PASSTHROUGH, 1);
                return SampledCoreset.exact(input);
            }
            Clustering b = seeding.cluster(points, k, rng, metrics);
            if (b.centers().isEmpty()) {
                return SampledCoreset.exact(new Coreset(PointBlock.EMPTY, input.offset())); // no positive weight
            }
            if (b.cost() == 0.0) {
                metrics.count(MetricNames.SENSITIVITY_ZERO_COST_SEEDINGS, 1);
            }
            double[] s = bound.sensitivities(points, b);
            double total = 0.0;
            for (double v : s) {
                total += v;
            }
            metrics.record(MetricNames.SENSITIVITY_TOTAL, total);

            int[] counts = AliasSampler.drawCounts(s, m, rng);
            int distinct = 0;
            for (int c : counts) {
                distinct += c > 0 ? 1 : 0;
            }
            int[] rows = new int[distinct];
            double[] outWeights = new double[distinct];
            int[] outCounts = new int[distinct];
            double[] outProb = new double[distinct];
            for (int i = 0, j = 0; i < n; i++) {
                int c = counts[i];
                if (c > 0) {
                    double q = s[i] / total;
                    rows[j] = i;
                    outWeights[j] = points.weight(i) * c / (m * q);
                    outCounts[j] = c;
                    outProb[j] = q;
                    j++;
                }
            }
            metrics.record(MetricNames.SENSITIVITY_OUTPUT_POINTS, distinct);
            return new SampledCoreset(new Coreset(points.select(rows, outWeights, distinct), input.offset()),
                    outCounts, outProb, m, total);
        }
    }

    public int k() {
        return k;
    }

    public int m() {
        return m;
    }

    public Seeding seeding() {
        return seeding;
    }

    public SensitivityBound bound() {
        return bound;
    }

    @Override
    public String toString() {
        return "sensitivity(k=" + k + ", m=" + m + ", seeding=" + seeding + ", bound=" + bound + ")";
    }
}
