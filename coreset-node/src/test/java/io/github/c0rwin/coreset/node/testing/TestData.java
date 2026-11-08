package io.github.c0rwin.coreset.node.testing;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.IngestResult;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.locks.LockSupport;

/** Seeded batch generators with integer weights, so weight sums are exact in any order. */
public final class TestData {

    private TestData() {
    }

    /** {@code size} sparse points of dimension {@code dim} with integer coordinates and weights in [1, 9]. */
    public static List<WeightedPoint> batch(SplittableRandom rng, int size, int dim) {
        List<WeightedPoint> out = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            int nnz = 1 + rng.nextInt(Math.min(3, dim));
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int k = 0; k < nnz; k++) {
                idx[k] = rng.nextInt(dim);
                val[k] = 1 + rng.nextInt(20);
            }
            out.add(new WeightedPoint(SparseVector.of(dim, idx, val), 1 + rng.nextInt(9)));
        }
        return out;
    }

    /**
     * Offers a batch the way a real producer does: a BUSY answer (full worker ring, which depends on how fast the
     * workers get scheduled) is retried with the same seq until another answer comes back or {@code timeout} passes.
     */
    public static IngestResult offer(CoresetNode node, String session, String producer, long seq,
                                     List<WeightedPoint> batch, Duration timeout) {
        long deadline = System.nanoTime() + timeout.toNanos();
        for (;;) {
            IngestResult r = node.ingest(session, producer, seq, batch);
            if (r.status() != IngestResult.Status.BUSY || System.nanoTime() - deadline > 0) {
                return r;
            }
            LockSupport.parkNanos(100_000L);
        }
    }

    public static double weight(List<WeightedPoint> points) {
        double s = 0.0;
        for (WeightedPoint p : points) {
            s += p.weight();
        }
        return s;
    }

    /** Live worker threads of node sessions whose id starts with {@code sessionIdPrefix}. */
    public static List<Thread> liveWorkers(String sessionIdPrefix) {
        String prefix = "coreset-node-" + sessionIdPrefix;
        return Thread.getAllStackTraces().keySet().stream()
                .filter(t -> t.isAlive() && t.getName().startsWith(prefix))
                .toList();
    }
}
