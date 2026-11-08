package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.algo.Clustering;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityBound;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * An importance sample of a node's rows that remembers the global row of every item, so its cost on any candidate is a
 * sum over the instance's {@link CostTable}. {@link #sensitivity} and {@link #uniform} replay the steps of
 * {@code SensitivityCoreset} (k-means++ seeding, standard bound with the default alpha) and {@code UniformCoreset}
 * call for call, so with the same generator they select the same points with the same weights (checked by tests).
 *
 * <p>Item {@code j} is global row {@code rows[j]} drawn {@code counts[j]} times; each draw carries weight
 * {@code unit[j] = w / (m q_j)}. An exact sample ({@code m == 0}) keeps every input row with {@code unit[j]} as its
 * weight.
 */
final class Sample {

    static final SensitivityBound BOUND = SensitivityBound.standard(SensitivityBound.DEFAULT_ALPHA);

    final int[] rows;
    final int[] counts;
    final double[] unit;
    final int m;

    Sample(int[] rows, int[] counts, double[] unit, int m) {
        this.rows = rows;
        this.counts = counts;
        this.unit = unit;
        this.m = m;
    }

    /** Importance proposal over a node's rows: bounds {@code s} (row order) and their total. */
    record Proposal(int[] rows, double[] inWeight, double[] s, double total) {

        double q(int local) {
            return s[local] / total;
        }
    }

    int size() {
        return rows.length;
    }

    boolean exact() {
        return m == 0;
    }

    double weight(int j) {
        return m == 0 ? unit[j] : unit[j] * counts[j];
    }

    /** Rows {@code global} (ascending) of {@code full} with the given weights ({@code null}: their own) as a set. */
    static PointSet input(PointSet full, int[] global, double[] weights) {
        double[] w = weights;
        if (w == null) {
            w = new double[global.length];
            for (int t = 0; t < global.length; t++) {
                w[t] = full.weight(global[t]);
            }
        }
        return PointSet.of(full.select(global, w, global.length));
    }

    static Sample exact(PointSet input, int[] global) {
        double[] w = new double[global.length];
        for (int t = 0; t < global.length; t++) {
            w[t] = input.weight(t);
        }
        return new Sample(global.clone(), new int[global.length], w, 0);
    }

    static Proposal proposal(PointSet input, int[] global, int k, RandomGenerator rng) {
        Clustering b = Seeding.kMeansPlusPlus(0).cluster(input, k, rng, MetricsSink.NOOP);
        double[] s = BOUND.sensitivities(input, b);
        double total = 0.0;
        for (double x : s) {
            total += x;
        }
        double[] w = new double[global.length];
        for (int t = 0; t < global.length; t++) {
            w[t] = input.weight(t);
        }
        return new Proposal(global, w, s, total);
    }

    /** m i.i.d. draws from the proposal, merged per row (as {@code SensitivityCoreset} returns them). */
    static Sample draw(Proposal p, int m, RandomGenerator rng) {
        return merge(p, AliasSampler.of(p.s).sortedDraws(m, rng), m);
    }

    /** Merges sorted local draws into items. */
    static Sample merge(Proposal p, int[] sortedLocal, int m) {
        int distinct = 0;
        for (int j = 0; j < sortedLocal.length; j++) {
            distinct += j == 0 || sortedLocal[j] != sortedLocal[j - 1] ? 1 : 0;
        }
        int[] rows = new int[distinct];
        int[] counts = new int[distinct];
        double[] unit = new double[distinct];
        for (int j = 0, d = 0; j < sortedLocal.length; d++) {
            int g = sortedLocal[j];
            int end = j + 1;
            while (end < sortedLocal.length && sortedLocal[end] == g) {
                end++;
            }
            rows[d] = p.rows[g];
            counts[d] = end - j;
            unit[d] = p.inWeight[g] / (m * p.q(g));
            j = end;
        }
        return new Sample(rows, counts, unit, m);
    }

    /** Same selection as {@code new SensitivityCoreset(k, m, Seeding.kMeansPlusPlus(0)).compress(input, rng)}. */
    static Sample sensitivity(PointSet input, int[] global, int k, int m, RandomGenerator rng) {
        if (input.size() <= m) {
            return exact(input, global);
        }
        return draw(proposal(input, global, k, rng), m, rng);
    }

    /** Same selection as {@code new UniformCoreset(m).compress(input, rng)}. */
    static Sample uniform(PointSet input, int[] global, int m, RandomGenerator rng) {
        final int n = input.size();
        if (n <= m) {
            return exact(input, global);
        }
        double[] w = new double[n];
        boolean equal = true;
        double total = 0.0;
        for (int t = 0; t < n; t++) {
            w[t] = input.weight(t);
            equal &= w[t] == w[0];
            total += w[t];
        }
        int[] draws = equal ? AliasSampler.sortedUniformDraws(n, m, rng) : AliasSampler.of(w).sortedDraws(m, rng);
        double[] s = equal ? filled(n, 1.0) : w;
        double sum = equal ? n : total;
        // q = w / W, so the unit weight w / (m q) is W / m for every draw.
        Sample out = merge(new Proposal(global, w, s, sum), draws, m);
        Arrays.fill(out.unit, total / m);
        return out;
    }

    private static double[] filled(int n, double v) {
        double[] a = new double[n];
        Arrays.fill(a, v);
        return a;
    }

    /** {@code acc[j] += scale * sum_items weight * cost(row, j)}. */
    void addCost(CostTable table, double scale, double[] acc) {
        for (int t = 0; t < rows.length; t++) {
            table.addRow(rows[t], scale * weight(t), acc);
        }
    }

    /** The sample as weighted rows of {@code full}, weights multiplied by {@code scale}. */
    PointBlock materialize(PointSet full, double scale) {
        double[] w = new double[rows.length];
        for (int t = 0; t < rows.length; t++) {
            w[t] = scale * weight(t);
        }
        return full.select(rows, w, rows.length);
    }

    /** The draws one by one, in random order: item index per draw. */
    int[] drawSequence(RandomGenerator rng) {
        int[] seq = new int[m];
        for (int j = 0, at = 0; j < rows.length; j++) {
            for (int c = 0; c < counts[j]; c++) {
                seq[at++] = j;
            }
        }
        Partitions.shuffle(seq, rng);
        return seq;
    }
}
