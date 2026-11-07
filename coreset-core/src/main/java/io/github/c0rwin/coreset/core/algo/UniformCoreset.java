package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.ArrayList;
import java.util.List;
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
            List<WeightedPoint> points = input.points();
            final int n = points.size();
            if (n <= m) {
                metrics.count(MetricNames.UNIFORM_PASSTHROUGH, 1);
                return input;
            }
            double[] weights = new double[n];
            double total = 0.0;
            double moment = 0.0;
            for (int i = 0; i < n; i++) {
                WeightedPoint p = points.get(i);
                weights[i] = p.weight();
                total += weights[i];
                moment += weights[i] * p.vector().squaredNorm();
            }
            Guards.requireSummable(total, moment);
            if (total == 0.0) {
                return new Coreset(List.of(), input.offset());
            }
            int[] counts = AliasSampler.drawCounts(weights, m, rng);
            List<WeightedPoint> out = new ArrayList<>(Math.min(n, m));
            for (int i = 0; i < n; i++) {
                if (counts[i] > 0) {
                    out.add(new WeightedPoint(points.get(i).vector(), total * counts[i] / m));
                }
            }
            metrics.record(MetricNames.UNIFORM_OUTPUT_POINTS, out.size());
            return new Coreset(out, input.offset());
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
