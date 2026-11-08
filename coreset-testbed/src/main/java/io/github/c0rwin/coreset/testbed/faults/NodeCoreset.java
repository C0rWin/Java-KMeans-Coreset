package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.algo.MeanCoreset;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.random.RandomGenerator;

/**
 * One node's coreset message: its points (weights as built), its cost on every candidate (offset included) and its
 * size on the wire. Sampled coresets also keep their {@link Sample} so per-draw faults can be applied later.
 */
final class NodeCoreset {

    enum Algo {
        SENSITIVITY, UNIFORM, MEAN;

        String id() {
            return name().toLowerCase(Locale.ROOT);
        }
    }

    /** Lloyd iterations of the mean coreset (as the standard suite). */
    static final int MEAN_LLOYD = 3;

    final PointSet points;
    final double offset;
    final double[] cost;
    final long bytes;
    final Sample sample;

    private NodeCoreset(PointSet points, double offset, double[] cost, long bytes, Sample sample) {
        this.points = points;
        this.offset = offset;
        this.cost = cost;
        this.bytes = bytes;
        this.sample = sample;
    }

    static NodeCoreset build(Instance inst, Algo algo, int[] rows, int m, RandomGenerator rng) {
        PointSet input = Sample.input(inst.points, rows, null);
        return switch (algo) {
            case SENSITIVITY -> ofSample(inst, Sample.sensitivity(input, rows, inst.k, m, rng));
            case UNIFORM -> ofSample(inst, Sample.uniform(input, rows, m, rng));
            case MEAN -> ofCoreset(inst, input.size() <= m ? Coreset.of(input)
                    : new MeanCoreset(m, MEAN_LLOYD).compress(Coreset.of(input), rng, MetricsSink.NOOP));
        };
    }

    static NodeCoreset ofSample(Instance inst, Sample s) {
        double[] cost = new double[inst.candidates()];
        s.addCost(inst.table, 1.0, cost);
        PointSet pts = PointSet.of(s.materialize(inst.points, 1.0));
        return new NodeCoreset(pts, 0.0, cost, Measure.messageBytes(Coreset.of(pts)), s);
    }

    static NodeCoreset ofCoreset(Instance inst, Coreset c) {
        double[] cost = CostTable.of(c.pointSet(), inst.candidates, null).weightedSums(c.pointSet());
        for (int j = 0; j < cost.length; j++) {
            cost[j] += c.offset();
        }
        return new NodeCoreset(c.pointSet(), c.offset(), cost, Measure.messageBytes(c), null);
    }

    int size() {
        return points.size();
    }

    /** Points of the given messages with every weight multiplied by {@code scale} (one copy). */
    static PointSet union(List<NodeCoreset> parts, double scale) {
        List<PointSet> sets = new ArrayList<>(parts.size());
        for (NodeCoreset p : parts) {
            if (!p.points.isEmpty()) {
                sets.add(p.points);
            }
        }
        PointSet all = PointSet.concat(sets);
        if (scale == 1.0 || all.isEmpty()) {
            return all;
        }
        int n = all.size();
        int[] rows = Partitions.identity(n);
        double[] w = new double[n];
        for (int r = 0; r < n; r++) {
            w[r] = scale * all.weight(r);
        }
        return PointSet.of(all.select(rows, w, n));
    }
}
