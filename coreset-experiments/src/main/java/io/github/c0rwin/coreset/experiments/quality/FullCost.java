package io.github.c0rwin.coreset.experiments.quality;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.experiments.data.ChunkReader;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Exact k-means costs of several center sets on all rows of a source in one streaming pass: {@code threads} contiguous
 * row ranges, read in chunks, each row's distance to every center computed once per set. Partial sums are added in
 * range order, so equal inputs and thread counts give bit-identical results.
 */
public final class FullCost {

    public static final int CHUNK_ROWS = 8192;

    /**
     * @param costs  one cost per center set, in input order
     * @param weight total input weight
     */
    public record Result(double[] costs, double weight) {
    }

    private FullCost() {
    }

    public static Result evaluate(PointSource src, List<List<SparseVector>> centerSets, int threads) {
        int sets = centerSets.size();
        SparseVector[][] cs = new SparseVector[sets][];
        for (int s = 0; s < sets; s++) {
            if (centerSets.get(s).isEmpty()) {
                throw new IllegalArgumentException("center set " + s + " is empty");
            }
            cs[s] = centerSets.get(s).toArray(new SparseVector[0]);
        }
        int t = (int) Math.max(1, Math.min(threads, src.size()));
        ExecutorService pool = Executors.newFixedThreadPool(t, r -> {
            Thread th = new Thread(r, "full-cost");
            th.setDaemon(true);
            return th;
        });
        try {
            List<Future<double[]>> parts = new ArrayList<>(t);
            for (int i = 0; i < t; i++) {
                long from = PointSource.shardStart(src.size(), i, t);
                long to = PointSource.shardStart(src.size(), i + 1, t);
                parts.add(pool.submit(() -> range(src, from, to, cs)));
            }
            double[] costs = new double[sets];
            double weight = 0.0;
            for (Future<double[]> f : parts) {
                double[] p = f.get();
                for (int s = 0; s < sets; s++) {
                    costs[s] += p[s];
                }
                weight += p[sets];
            }
            return new Result(costs, weight);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        } catch (ExecutionException e) {
            throw e.getCause() instanceof UncheckedIOException u ? u : new IllegalStateException(e.getCause());
        } finally {
            pool.shutdownNow();
        }
    }

    /** Per-set costs of rows {@code [from, to)} followed by their total weight. */
    private static double[] range(PointSource src, long from, long to, SparseVector[][] cs) {
        double[] acc = new double[cs.length + 1];
        try (ChunkReader r = src.open(from, to)) {
            for (PointBlock b = r.next(CHUNK_ROWS); b != null; b = r.next(CHUNK_ROWS)) {
                for (int i = 0, n = b.size(); i < n; i++) {
                    double w = b.weight(i);
                    acc[cs.length] += w;
                    if (w <= 0.0) {
                        continue;
                    }
                    for (int s = 0; s < cs.length; s++) {
                        SparseVector[] c = cs[s];
                        double best = b.squaredDistance(i, c[0]);
                        for (int j = 1; j < c.length; j++) {
                            double d = b.squaredDistance(i, c[j]);
                            if (d < best) {
                                best = d;
                            }
                        }
                        acc[s] += w * best;
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return acc;
    }
}
