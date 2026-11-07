package io.github.c0rwin.coreset.core.validate;

/**
 * Splits an end-to-end (eps, delta) target across a merge-and-reduce tree. Ratio bounds multiply across levels, so
 * L levels of {@code (1 +- eps_l)} compose to {@code (1 +- eps_l)^L}; requiring {@code (1 + eps_l)^L = 1 + eps} gives
 * {@code eps_l = (1 + eps)^(1/L) - 1}. Failure probabilities are union-bounded over the reduce operations:
 * {@code delta_l = delta / reduceOps}.
 *
 * @param eps            end-to-end relative error target
 * @param delta          end-to-end failure probability
 * @param levels         L = ceil(log2(n / m)) reduce levels, plus one with a final reduce of the snapshot
 * @param reduceOps      merges a binary-counter tree performs over {@code ceil(n / m)} batches, plus the final reduce
 * @param epsPerLevel    {@code eps_l}; equals eps when there are no levels
 * @param deltaPerReduce {@code delta_l}; equals delta when there are no reduce operations
 */
public record ErrorBudget(double eps, double delta, int levels, long reduceOps, double epsPerLevel,
                          double deltaPerReduce) {

    /**
     * @param n           input points
     * @param m           points per batch / coreset size
     * @param finalReduce whether the tree's snapshot is compressed once more (as by {@code compressedSnapshot})
     * @throws IllegalArgumentException unless {@code eps > 0}, {@code 0 < delta < 1}, {@code n >= 1}, {@code m >= 1}
     */
    public static ErrorBudget forStream(double eps, double delta, long n, int m, boolean finalReduce) {
        if (!(eps > 0.0) || Double.isInfinite(eps)) {
            throw new IllegalArgumentException("eps must be finite and > 0: " + eps);
        }
        if (!(delta > 0.0 && delta < 1.0)) {
            throw new IllegalArgumentException("delta must be in (0, 1): " + delta);
        }
        if (n < 1 || m < 1) {
            throw new IllegalArgumentException("n and m must be >= 1: n=" + n + ", m=" + m);
        }
        long batches = (n + m - 1) / m;
        int levels = (64 - Long.numberOfLeadingZeros(batches - 1)) + (finalReduce ? 1 : 0);
        long reduceOps = batches - Long.bitCount(batches) + (finalReduce ? 1 : 0);
        double epsPerLevel = levels == 0 ? eps : Math.expm1(Math.log1p(eps) / levels);
        double deltaPerReduce = reduceOps == 0 ? delta : delta / reduceOps;
        return new ErrorBudget(eps, delta, levels, reduceOps, epsPerLevel, deltaPerReduce);
    }

    /**
     * Factor by which a sampling coreset's size must grow to reach {@code epsPerLevel} instead of eps, since sizes
     * scale as {@code 1 / eps^2}.
     */
    public double sampleSizeFactor() {
        double r = eps / epsPerLevel;
        return r * r;
    }
}
