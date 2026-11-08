package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
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
 * <p>Time: the seeding (O(n k nnz) for k-means++) plus one pass over the points for the bounds and O(m log m) for
 * sampling and the output. Scratch beyond the seeding's 12 bytes per point: the bounds overwrite the seeding's
 * distances, the alias table reuses its assignment array, and only the probability table (8 bytes per point) is
 * new. Stateless and thread-safe.
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
        return sample(input, rng, rng, metrics, true);
    }

    /**
     * Always draws m points i.i.d., even from an input of at most m points (an input without positive weight still
     * gives an exact empty coreset). The seeding, hence the distribution q, comes from {@code seedingRng} and the
     * draws from {@code samplingRng}: replicas sharing {@code seedingRng}'s state sample one q independently.
     */
    public SampledCoreset sample(Coreset input, RandomGenerator seedingRng, RandomGenerator samplingRng,
                                 MetricsSink metrics) {
        return sample(input, seedingRng, samplingRng, metrics, false);
    }

    private SampledCoreset sample(Coreset input, RandomGenerator seedingRng, RandomGenerator samplingRng,
                                  MetricsSink metrics, boolean passThroughSmall) {
        try (Span ignored = metrics.span(MetricNames.SENSITIVITY_COMPRESS)) {
            PointSet points = input.pointSet();
            final int n = points.size();
            if (passThroughSmall && n <= m) {
                metrics.count(MetricNames.SENSITIVITY_PASSTHROUGH, 1);
                return SampledCoreset.exact(input);
            }
            Clustering b = seeding.cluster(points, k, seedingRng, metrics);
            if (b.centers().isEmpty()) {
                return SampledCoreset.exact(new Coreset(PointBlock.EMPTY, input.offset())); // no positive weight
            }
            if (b.cost() == 0.0) {
                metrics.count(MetricNames.SENSITIVITY_ZERO_COST_SEEDINGS, 1);
            }
            // b is private to this call: its distances become the bounds and its assignment the alias table.
            double[] s = b.squaredDistances;
            double total = bound.fill(points, b, s);
            long scanned = b.clusterWeights == null ? 2L * n : n;
            metrics.record(MetricNames.SENSITIVITY_TOTAL, total);

            int[] draws = AliasSampler.ofOwned(s.clone(), b.assignment).sortedDraws(m, samplingRng);
            int distinct = 0;
            for (int j = 0; j < m; j++) {
                distinct += j == 0 || draws[j] != draws[j - 1] ? 1 : 0;
            }
            int[] rows = new int[distinct];
            double[] outWeights = new double[distinct];
            int[] outCounts = new int[distinct];
            double[] outProb = new double[distinct];
            for (int j = 0, d = 0; j < m; d++) {
                int g = draws[j];
                int end = j + 1;
                while (end < m && draws[end] == g) {
                    end++;
                }
                int c = end - j;
                double q = s[g] / total;
                rows[d] = g;
                outWeights[d] = points.weight(g) * c / (m * q);
                outCounts[d] = c;
                outProb[d] = q;
                j = end;
            }
            metrics.count(MetricNames.SENSITIVITY_POINTS_SCANNED, scanned + distinct);
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
