package io.github.c0rwin.coreset.testbed.eval;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.IntStream;

/**
 * Evaluates many center sets over a point set in one pass, in parallel over fixed-size chunks. Chunk partial sums are
 * reduced in chunk order, so results are bit-identical whatever the thread count or scheduling.
 */
public final class CostEvaluator {

    /** Points per parallel task; chunks never straddle a segment boundary. */
    public static final int CHUNK = 4096;

    private final Problem problem;
    private final ForkJoinPool pool;
    private final AtomicLong distanceEvaluations = new AtomicLong();
    private final AtomicLong pointsScanned = new AtomicLong();

    /** @param pool runs the chunks; {@code null} evaluates on the calling thread */
    public CostEvaluator(Problem problem, ForkJoinPool pool) {
        this.problem = Objects.requireNonNull(problem, "problem");
        this.pool = pool;
    }

    public Problem problem() {
        return problem;
    }

    /** Point-to-center distance evaluations performed so far (the cost of validation, machine-independently). */
    public long distanceEvaluations() {
        return distanceEvaluations.get();
    }

    /** Points read by cost passes so far; divide by the input size for the number of full-data passes. */
    public long pointsScanned() {
        return pointsScanned.get();
    }

    /** Costs of each candidate on {@code c}, offset included. */
    public double[] costs(Coreset c, List<Centers> candidates) {
        double[] out = segmentCosts(c.points(), new int[] {c.size()}, candidates)[0];
        double offset = problem.offset(c);
        for (int j = 0; j < out.length; j++) {
            out[j] += offset;
        }
        return out;
    }

    public double cost(Coreset c, Centers centers) {
        return costs(c, List.of(centers))[0];
    }

    /**
     * {@code result[s][j]} is the cost (no offset) of candidate {@code j} on points {@code [ends[s-1], ends[s])}, with
     * {@code ends[-1] = 0}.
     *
     * @throws IllegalArgumentException unless {@code ends} is non-decreasing, non-negative and its last entry is
     *     {@code points.size()}
     */
    public double[][] segmentCosts(List<WeightedPoint> points, int[] ends, List<Centers> candidates) {
        if (ends.length == 0 || ends[ends.length - 1] != points.size()) {
            throw new IllegalArgumentException("last segment end must be " + points.size());
        }
        List<int[]> chunks = new ArrayList<>();
        int start = 0;
        for (int s = 0; s < ends.length; s++) {
            if (ends[s] < start) {
                throw new IllegalArgumentException("segment ends must be non-decreasing: " + ends[s] + " < " + start);
            }
            for (int from = start; from < ends[s]; from += CHUNK) {
                chunks.add(new int[] {s, from, Math.min(from + CHUNK, ends[s])});
            }
            start = ends[s];
        }
        final Centers[] cands = candidates.toArray(new Centers[0]);
        final double[][] partial = new double[chunks.size()][];
        if (pool == null || chunks.size() < 2) {
            for (int i = 0; i < chunks.size(); i++) {
                partial[i] = chunk(points, chunks.get(i), cands);
            }
        } else {
            try {
                pool.submit(() -> IntStream.range(0, chunks.size()).parallel()
                        .forEach(i -> partial[i] = chunk(points, chunks.get(i), cands))).get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("cost evaluation interrupted", e);
            } catch (ExecutionException e) {
                throw new IllegalStateException("cost evaluation failed", e.getCause());
            }
        }
        double[][] out = new double[ends.length][cands.length];
        for (int i = 0; i < partial.length; i++) {
            double[] seg = out[chunks.get(i)[0]];
            for (int j = 0; j < cands.length; j++) {
                seg[j] += partial[i][j];
            }
        }
        long centers = 0;
        for (Centers c : cands) {
            centers += c.size();
        }
        distanceEvaluations.addAndGet(centers * points.size());
        pointsScanned.addAndGet(points.size());
        return out;
    }

    /** One sequential sweep over the chunk; per point, every candidate is evaluated while the point is hot. */
    private double[] chunk(List<WeightedPoint> points, int[] range, Centers[] cands) {
        int maxK = 0;
        for (Centers c : cands) {
            maxK = Math.max(maxK, c.size());
        }
        double[] scratch = new double[maxK];
        double[] sums = new double[cands.length];
        for (int i = range[1]; i < range[2]; i++) {
            WeightedPoint p = points.get(i);
            double w = p.weight();
            if (w > 0.0) {
                for (int j = 0; j < cands.length; j++) {
                    sums[j] += w * problem.pointCost(p.vector(), cands[j], scratch);
                }
            }
        }
        return sums;
    }
}
