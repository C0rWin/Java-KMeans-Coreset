package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.Span;
import java.util.random.RandomGenerator;

/**
 * Deterministic coreset of Barger and Feldman (SDM 2016, Algorithm 1 with the mean as the 1-coreset of each part):
 * partition P by an m-means clustering (weighted Lloyd), replace each part P_i by its weighted mean with weight
 * {@code W(P_i)}, and add {@code phi = sum_i cost(P_i, mean_i)} to the offset. By Lemma 1, {@code cost(C, Q) >=
 * cost(P, Q)} for every Q, and total weight, first moment and {@code offset + sum w ||p||^2} are preserved exactly.
 * Inputs with at most m points are returned unchanged.
 *
 * <p>Time: Lloyd, O(n m nnz (iterations + 1)), plus {@link WeightedLloyd#meansOf} for the means and phi (two passes
 * and a sort of each part's stored entries); the adaptive variant runs Lloyd for m = 1, k, k^2, ... up to maxM.
 * Memory follows the stored entries, never m * dim for high-dimensional sparse input. The only randomness is the
 * Lloyd seeding. Stateless and thread-safe.
 */
public final class MeanCoreset implements CoresetAlgorithm {

    private final int m; // the fixed m, or maxM when adaptive
    private final int lloydIterations;
    private final int k;
    private final double eps;

    private MeanCoreset(int m, int lloydIterations, int k, double eps) {
        if (lloydIterations < 0) {
            throw new IllegalArgumentException("lloydIterations must be >= 0: " + lloydIterations);
        }
        this.m = m;
        this.lloydIterations = lloydIterations;
        this.k = k;
        this.eps = eps;
    }

    /** Fixed number of means. @throws IllegalArgumentException if {@code m < 1} or {@code lloydIterations < 0} */
    public MeanCoreset(int m, int lloydIterations) {
        this(requirePositive("m", m), lloydIterations, 0, 0.0);
    }

    /**
     * Picks m as in Algorithm 1 line 1 with Lloyd standing in for opt: the smallest {@code m = k^t} with
     * {@code opt(P, k^t) - opt(P, k^(t+1)) <= eps^2 * opt(P, k)}, where any {@code k^(t+1)} above {@code maxM} is
     * clamped to {@code maxM} and the search stops once m reaches {@code maxM}. Lloyd costs are only upper bounds on
     * opt, so this is a heuristic; the one-sided guarantee holds for whatever m is chosen.
     *
     * @throws IllegalArgumentException if {@code k < 1}, {@code maxM < 1}, {@code eps} not in (0, 1) or
     *     {@code lloydIterations < 0}
     */
    public static MeanCoreset adaptive(int k, double eps, int maxM, int lloydIterations) {
        requirePositive("k", k);
        if (!(eps > 0.0 && eps < 1.0)) {
            throw new IllegalArgumentException("eps must be in (0, 1): " + eps);
        }
        return new MeanCoreset(requirePositive("maxM", maxM), lloydIterations, k, eps);
    }

    private static int requirePositive(String name, int v) {
        if (v < 1) {
            throw new IllegalArgumentException(name + " must be >= 1: " + v);
        }
        return v;
    }

    @Override
    public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
        try (Span ignored = metrics.span(MetricNames.MEAN_COMPRESS)) {
            PointBlock points = input.block();
            if (points.size() <= (isAdaptive() ? 1 : m)) {
                metrics.count(MetricNames.MEAN_PASSTHROUGH, 1);
                return input;
            }
            KMeansResult partition = isAdaptive()
                    ? adaptivePartition(points, rng, metrics)
                    : WeightedLloyd.cluster(points, m, lloydIterations, rng, metrics);
            if (partition.centers().size() >= points.size()) {
                metrics.count(MetricNames.MEAN_PASSTHROUGH, 1);
                return input;
            }
            Coreset means = WeightedLloyd.meansOf(points, partition.assignment(), partition.centers().size());
            metrics.record(MetricNames.MEAN_ADDED_OFFSET, means.offset());
            Coreset out = new Coreset(means.block(), input.offset() + means.offset());
            metrics.record(MetricNames.MEAN_OUTPUT_POINTS, out.size());
            return out;
        }
    }

    private boolean isAdaptive() {
        return k > 0;
    }

    private KMeansResult adaptivePartition(PointBlock points, RandomGenerator rng, MetricsSink metrics) {
        final int maxM = m;
        KMeansResult optK = WeightedLloyd.cluster(points, Math.min(k, maxM), lloydIterations, rng, metrics);
        final double threshold = eps * eps * optK.cost();
        KMeansResult current = WeightedLloyd.cluster(points, 1, lloydIterations, rng, metrics);
        long t = 1;
        while (t < maxM) {
            long next = Math.min(t * k, maxM);
            if (next == t) {
                break; // k == 1: m stays 1
            }
            KMeansResult candidate = next == Math.min(k, maxM)
                    ? optK
                    : WeightedLloyd.cluster(points, (int) next, lloydIterations, rng, metrics);
            if (current.cost() - candidate.cost() <= threshold) {
                break;
            }
            current = candidate;
            t = next;
        }
        metrics.record(MetricNames.MEAN_ADAPTIVE_M, t);
        return current;
    }

    @Override
    public String toString() {
        return isAdaptive()
                ? "mean(k=" + k + ", eps=" + eps + ", maxM=" + m + ", lloydIterations=" + lloydIterations + ")"
                : "mean(m=" + m + ", lloydIterations=" + lloydIterations + ")";
    }
}
