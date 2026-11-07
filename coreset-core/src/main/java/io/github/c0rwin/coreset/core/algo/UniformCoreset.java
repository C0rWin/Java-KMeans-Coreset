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
import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Uniform sampling on weighted input: m i.i.d. draws with {@code q_p = w_p / W}, and a point drawn {@code c_p} times
 * gets weight {@code W * c_p / m}, so {@code E[cost(C, Q)] = cost(P, Q)} for every Q and the total weight is W (up to
 * rounding). Inputs with at most m points are returned unchanged. The offset passes through.
 *
 * <p>O(n + m log m) time, one pass over the points, no distance evaluations. Scratch is O(m) when all weights are
 * equal (the draws are replayed without an alias table), otherwise 12 bytes per point. Stateless and thread-safe.
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
            double[] weights = null; // allocated at the first weight that differs from the first one
            final double first = points.weight(0);
            double total = 0.0;
            double moment = 0.0;
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock p = points.segment(t);
                int i = 0;
                final int size = p.size();
                if (weights == null) {
                    for (; i < size; i++, g++) {
                        double w = p.weight(i);
                        if (w != first) {
                            weights = new double[n];
                            Arrays.fill(weights, 0, g, first);
                            break; // row i is summed and stored below
                        }
                        total += w;
                        moment += w * p.squaredNorm(i);
                    }
                }
                for (; i < size; i++, g++) {
                    double w = p.weight(i);
                    weights[g] = w;
                    total += w;
                    moment += w * p.squaredNorm(i);
                }
            }
            Guards.requireSummable(total, moment);
            if (total == 0.0) {
                return new Coreset(PointBlock.EMPTY, input.offset());
            }
            // Equal positive weights make every alias probability exactly 1: the replay is bit-identical.
            int[] draws = weights == null ? AliasSampler.sortedUniformDraws(n, m, rng)
                    : AliasSampler.ofOwned(weights, new int[n]).sortedDraws(m, rng);
            int distinct = 0;
            for (int j = 0; j < m; j++) {
                distinct += j == 0 || draws[j] != draws[j - 1] ? 1 : 0;
            }
            int[] rows = new int[distinct];
            double[] outWeights = new double[distinct];
            for (int j = 0, d = 0; j < m; d++) {
                int row = draws[j];
                int end = j + 1;
                while (end < m && draws[end] == row) {
                    end++;
                }
                rows[d] = row;
                outWeights[d] = total * (end - j) / m;
                j = end;
            }
            metrics.count(MetricNames.UNIFORM_POINTS_SCANNED, (long) n + distinct);
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
