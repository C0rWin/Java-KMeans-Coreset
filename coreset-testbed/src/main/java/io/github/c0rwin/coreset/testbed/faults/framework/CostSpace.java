package io.github.c0rwin.coreset.testbed.faults.framework;

import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.random.RandomGenerator;

/**
 * A weighted family with sampling law {@code q = s / S} and fixed queries; stores {@code g_t(Q) = (w_t / q_t) f_t(Q)},
 * the cost a scale-free record carries, so every decoder is a sum of {@code g} over received draws.
 */
final class CostSpace {

    final int size;
    final int queries;
    final int classes;
    final int[] rows;
    final double[] weight;
    final int[] cls;
    final double[] q;
    final double[] g;
    final double[] cost;
    private final AliasSampler sampler;

    /** {@code s}: unnormalized sampling law; {@code table}: per-row costs, {@code queries} per row; null classes = one. */
    CostSpace(int[] rows, double[] w, double[] s, double[] table, int queries, int[] clsByRow, int classes) {
        this.size = rows.length;
        this.queries = queries;
        this.classes = clsByRow == null ? 1 : classes;
        this.rows = rows;
        this.weight = w;
        this.sampler = AliasSampler.of(s);
        double total = 0.0;
        for (double v : s) {
            total += v;
        }
        this.q = new double[size];
        this.cls = new int[size];
        this.g = new double[size * queries];
        this.cost = new double[queries];
        for (int t = 0; t < size; t++) {
            q[t] = s[t] / total;
            cls[t] = clsByRow == null ? 0 : clsByRow[rows[t]];
            int base = rows[t] * queries;
            double unit = q[t] > 0.0 ? w[t] / q[t] : 0.0;
            for (int j = 0; j < queries; j++) {
                double f = table[base + j];
                cost[j] += w[t] * f;
                g[t * queries + j] = unit * f;
            }
        }
    }

    /** Identity rows {@code 0..n-1}. */
    static int[] identity(int n) {
        int[] r = new int[n];
        for (int i = 0; i < n; i++) {
            r[i] = i;
        }
        return r;
    }

    int draw(RandomGenerator rng) {
        return sampler.next(rng);
    }

    void draws(int[] out, int m, RandomGenerator rng) {
        for (int j = 0; j < m; j++) {
            out[j] = sampler.next(rng);
        }
    }

    double rel(double estimate, int j) {
        return estimate / cost[j] - 1.0;
    }
}
