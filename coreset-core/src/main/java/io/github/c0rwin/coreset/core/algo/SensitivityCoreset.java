package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.random.RandomGenerator;

/**
 * Sensitivity (importance) sampling in the Feldman-Langberg framework. A {@link Seeding} gives a rough clustering B;
 * each point gets the sensitivity bound
 * <pre>  s_p = 8 * w_p / W(cluster of p) + 2 * w_p * d(p, B)^2 / cost(B)</pre>
 * (only the first term when {@code cost(B) = 0}), m points are drawn i.i.d. with {@code q_p = s_p / S} where
 * {@code S = sum_p s_p} is the actual total, and a point drawn {@code c_p} times gets weight
 * {@code w_p * c_p / (m * q_p)}. This makes {@code cost(C, Q)} unbiased for every Q and the total weight unbiased.
 * Inputs with at most m points are returned unchanged; the offset passes through.
 *
 * <p>Time: the seeding (O(n k d) for k-means++) plus O(n + m) for sensitivities and sampling. Stateless and
 * thread-safe.
 */
public final class SensitivityCoreset implements CoresetAlgorithm {

    private final int k;
    private final int m;
    private final Seeding seeding;

    /** @throws IllegalArgumentException if {@code k < 1} or {@code m < 1} */
    public SensitivityCoreset(int k, int m, Seeding seeding) {
        if (k < 1) {
            throw new IllegalArgumentException("k must be >= 1: " + k);
        }
        if (m < 1) {
            throw new IllegalArgumentException("m must be >= 1: " + m);
        }
        this.k = k;
        this.m = m;
        this.seeding = Objects.requireNonNull(seeding, "seeding");
    }

    @Override
    public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
        return compressWithProvenance(input, rng, metrics).coreset();
    }

    /** Same coreset as {@link #compress} for the same generator state, plus the sampling provenance. */
    public SampledCoreset compressWithProvenance(Coreset input, RandomGenerator rng, MetricsSink metrics) {
        try (Span ignored = metrics.span(MetricNames.SENSITIVITY_COMPRESS)) {
            List<WeightedPoint> points = input.points();
            final int n = points.size();
            if (n <= m) {
                metrics.count(MetricNames.SENSITIVITY_PASSTHROUGH, 1);
                return SampledCoreset.exact(input);
            }
            Clustering b = seeding.cluster(points, k, rng, metrics);
            if (b.centers().isEmpty()) {
                return SampledCoreset.exact(new Coreset(List.of(), input.offset())); // no positive weight
            }
            final int[] assignment = b.assignment;
            final double[] sq = b.squaredDistances;
            double[] clusterWeight = new double[b.centers().size()];
            for (int i = 0; i < n; i++) {
                double w = points.get(i).weight();
                if (w > 0.0) {
                    clusterWeight[assignment[i]] += w;
                }
            }
            final double costB = b.cost();
            final boolean distanceTerm = costB > 0.0;
            if (!distanceTerm) {
                metrics.count(MetricNames.SENSITIVITY_ZERO_COST_SEEDINGS, 1);
            }
            double[] s = new double[n];
            double total = 0.0;
            for (int i = 0; i < n; i++) {
                double w = points.get(i).weight();
                if (w > 0.0) {
                    double v = 8.0 * w / clusterWeight[assignment[i]];
                    if (distanceTerm) {
                        v += 2.0 * w * sq[i] / costB;
                    }
                    s[i] = v;
                    total += v;
                }
            }
            metrics.record(MetricNames.SENSITIVITY_TOTAL, total);

            int[] counts = AliasSampler.drawCounts(s, m, rng);
            int distinct = 0;
            for (int c : counts) {
                distinct += c > 0 ? 1 : 0;
            }
            List<WeightedPoint> out = new ArrayList<>(distinct);
            int[] outCounts = new int[distinct];
            double[] outProb = new double[distinct];
            for (int i = 0, j = 0; i < n; i++) {
                int c = counts[i];
                if (c > 0) {
                    WeightedPoint p = points.get(i);
                    double q = s[i] / total;
                    out.add(new WeightedPoint(p.vector(), p.weight() * c / (m * q)));
                    outCounts[j] = c;
                    outProb[j] = q;
                    j++;
                }
            }
            metrics.record(MetricNames.SENSITIVITY_OUTPUT_POINTS, distinct);
            return new SampledCoreset(new Coreset(out, input.offset()), outCounts, outProb, m, total);
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

    @Override
    public String toString() {
        return "sensitivity(k=" + k + ", m=" + m + ", seeding=" + seeding + ")";
    }
}
