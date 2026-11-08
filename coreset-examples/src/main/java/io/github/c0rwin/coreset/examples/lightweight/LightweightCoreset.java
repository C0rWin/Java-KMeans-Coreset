package io.github.c0rwin.coreset.examples.lightweight;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.random.RandomGenerator;

/**
 * Lightweight coreset (Bachem, Lucic, Krause, "Scalable k-Means Clustering via Lightweight Coresets", KDD 2018), on
 * weighted input. With {@code mu} the weighted mean, {@code W} the total weight and
 * {@code q(x) = 1/2 * w(x) / W + 1/2 * w(x) d(x, mu)^2 / sum_y w(y) d(y, mu)^2}, it draws {@code m} points i.i.d. from
 * {@code q}; a point drawn {@code c} times gets weight {@code w(x) c / (m q(x))}. Every k-means cost, and the total
 * weight, are therefore unbiased estimates of the input's. Inputs with at most {@code m} points are returned
 * unchanged; the offset passes through.
 *
 * <p>Two passes, O(n + m log m) time plus one dense mean of the input dimension. Written against the public
 * coreset-core API only, as a template for plugging in an algorithm. Stateless and thread-safe.
 */
public final class LightweightCoreset implements CoresetAlgorithm {

    /** Span name of one compression. */
    public static final String SPAN = "examples.lightweight.compress";

    private final int m;

    /** @throws IllegalArgumentException if {@code m < 1} */
    public LightweightCoreset(int m) {
        if (m < 1) {
            throw new IllegalArgumentException("m must be >= 1: " + m);
        }
        this.m = m;
    }

    @Override
    public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
        try (Span ignored = metrics.span(SPAN)) {
            PointSet points = input.pointSet();
            final int n = points.size();
            if (n <= m) {
                return input;
            }
            double[] weights = new double[n];
            double[] mean = new double[points.dim()];
            double total = 0.0;
            double moment = 0.0;
            for (int s = 0, g = 0; s < points.segmentCount(); s++) {
                PointBlock block = points.segment(s);
                for (int i = 0; i < block.size(); i++, g++) {
                    double w = block.weight(i);
                    weights[g] = w;
                    total += w;
                    moment += w * block.squaredNorm(i);
                    block.addTo(i, mean, w);
                }
            }
            Guards.requireSummable(total, moment);
            if (total == 0.0) {
                return new Coreset(PointBlock.EMPTY, input.offset());
            }
            double meanNorm = 0.0;
            for (int c = 0; c < mean.length; c++) {
                mean[c] /= total;
                meanNorm += mean[c] * mean[c];
            }
            double[] q = new double[n];
            double spread = 0.0;
            for (int s = 0, g = 0; s < points.segmentCount(); s++) {
                Guards.checkInterrupted("lightweight coreset");
                PointBlock block = points.segment(s);
                for (int i = 0; i < block.size(); i++, g++) {
                    q[g] = weights[g] * Math.max(0.0, block.squaredDistance(i, mean, meanNorm));
                    spread += q[g];
                }
            }
            double sum = 0.0;
            for (int g = 0; g < n; g++) {
                // All points at the mean leave only the weight term.
                q[g] = spread > 0.0 ? 0.5 * weights[g] / total + 0.5 * q[g] / spread : weights[g] / total;
                sum += q[g];
            }
            int[] draws = AliasSampler.of(q).sortedDraws(m, rng);
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
                // q[row] / sum is the exact draw probability, whatever the rounding in the two halves.
                outWeights[d] = weights[row] * (end - j) * sum / (m * q[row]);
                j = end;
            }
            return new Coreset(points.select(rows, outWeights, distinct), input.offset());
        }
    }

    public int m() {
        return m;
    }

    @Override
    public String toString() {
        return "lightweight(m=" + m + ")";
    }
}
