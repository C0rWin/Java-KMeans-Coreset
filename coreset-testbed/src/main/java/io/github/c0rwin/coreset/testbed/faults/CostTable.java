package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * Unweighted k-means cost of every row against every candidate, {@code min_c ||p - c||^2}, row-major; callers
 * multiply by whatever weight the row carries. Computed once in one fused pass, it turns every fault realization into
 * sums over rows: no further distance evaluations.
 */
final class CostTable {

    static final int CHUNK = 4096;

    final int rows;
    final int cols;
    final double[] cost;

    private CostTable(int rows, int cols, double[] cost) {
        this.rows = rows;
        this.cols = cols;
        this.cost = cost;
    }

    /** @param pool runs the chunks; {@code null} evaluates on the calling thread */
    static CostTable of(PointSet points, List<Centers> candidates, ForkJoinPool pool) {
        final int n = points.size();
        final int cols = candidates.size();
        final Centers[] cands = candidates.toArray(new Centers[0]);
        int maxK = 1;
        for (Centers c : cands) {
            maxK = Math.max(maxK, c.size());
        }
        final int scratch = maxK;
        final double[] out = new double[n * cols];
        final int chunks = (n + CHUNK - 1) / CHUNK;
        Runnable all = () -> IntStream.range(0, chunks).parallel().forEach(t -> fill(points, cands, out, t, scratch));
        if (pool == null || chunks < 2) {
            for (int t = 0; t < chunks; t++) {
                fill(points, cands, out, t, scratch);
            }
        } else {
            try {
                pool.submit(all).get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("cost table interrupted", e);
            } catch (ExecutionException e) {
                throw new IllegalStateException("cost table failed", e.getCause());
            }
        }
        return new CostTable(n, cols, out);
    }

    private static void fill(PointSet points, Centers[] cands, double[] out, int chunk, int maxK) {
        Problem problem = Problem.kMeans();
        double[] scratch = new double[maxK];
        int from = chunk * CHUNK;
        int to = Math.min(points.size(), from + CHUNK);
        int cols = cands.length;
        for (int g = from; g < to; g++) {
            int s = points.segmentOf(g);
            PointBlock b = points.segment(s);
            int i = g - points.start(s);
            var v = b.vector(i);
            for (int j = 0; j < cols; j++) {
                out[g * cols + j] = problem.pointCost(v, cands[j], scratch);
            }
        }
    }

    /** {@code acc[j] += scale * cost(row, j)}. */
    void addRow(int row, double scale, double[] acc) {
        int o = row * cols;
        for (int j = 0; j < cols; j++) {
            acc[j] += scale * cost[o + j];
        }
    }

    /** Column sums over {@code rows}, in the given order. */
    double[] sumRows(int[] rowIds) {
        double[] acc = new double[cols];
        for (int r : rowIds) {
            addRow(r, 1.0, acc);
        }
        return acc;
    }

    /** {@code sum_r w_r cost(r, j)} over every row, weights from {@code points}. */
    double[] weightedSums(PointSet points) {
        double[] acc = new double[cols];
        for (int r = 0; r < rows; r++) {
            addRow(r, points.weight(r), acc);
        }
        return acc;
    }
}
