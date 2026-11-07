package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.random.RandomGenerator;

/**
 * Uniform sampling on weighted input: m i.i.d. draws with {@code q_p = w_p / W}, and a point drawn {@code c_p} times
 * gets weight {@code W * c_p / m}, so {@code E[cost(C, Q)] = cost(P, Q)} for every Q and the total weight is W (up to
 * rounding). Inputs with at most m points are returned unchanged. The offset passes through.
 *
 * <p>O(n + m) time, no distance evaluations. Stateless and thread-safe.
 */
public final class UniformCoreset implements CoresetAlgorithm {

    private final int m;

    /** @throws IllegalArgumentException if {@code m < 1} */
    public UniformCoreset(int m) {
        if (m < 1) {
            throw new IllegalArgumentException("m must be >= 1: " + m);
        }
        this.m = m;
    }

    @Override
    public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
        try (Span ignored = metrics.span(MetricNames.UNIFORM_COMPRESS)) {
            PointSet points = input.pointSet();
            final int n = points.size();
            if (n <= m) {
                metrics.count(MetricNames.UNIFORM_PASSTHROUGH, 1);
                return input;
            }
            double[] weights = new double[n];
            double total = 0.0;
            double moment = 0.0;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock p = points.segment(t);
                for (int i = 0, size = p.size(); i < size; i++, g++) {
                    weights[g] = p.weight(i);
                    total += weights[g];
                    moment += weights[g] * p.squaredNorm(i);
                }
            }
            Guards.requireSummable(total, moment);
            if (total == 0.0) {
                return new Coreset(PointBlock.EMPTY, input.offset());
            }
            int[] counts = AliasSampler.drawCounts(weights, m, rng);
            int[] rows = new int[Math.min(n, m)];
            double[] outWeights = new double[rows.length];
            int distinct = 0;
            for (int i = 0; i < n; i++) {
                if (counts[i] > 0) {
                    rows[distinct] = i;
                    outWeights[distinct++] = total * counts[i] / m;
                }
            }
            metrics.record(MetricNames.UNIFORM_OUTPUT_POINTS, distinct);
            return new Coreset(points.select(rows, outWeights, distinct), input.offset());
        }
    }

    public int m() {
        return m;
    }

    @Override
    public String toString() {
        return "uniform(m=" + m + ")";
    }
}
