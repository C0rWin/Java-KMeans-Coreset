package io.github.c0rwin.coreset.core.validate;

/**
 * A-priori sample sizes for sensitivity sampling (Bachem, Lucic, Krause, "Practical Coreset Constructions for Machine
 * Learning", 2017). These are worst-case and loose: useful for trends (m grows as S^2 / eps^2), far above the sizes
 * that certify well a posteriori with {@link Distortion} and {@link CostConfidenceInterval}.
 */
public final class TheoryBounds {

    private TheoryBounds() {
    }

    /** Total sensitivity {@code 8 * clusters + 2} of the bound used by {@code SensitivityCoreset} (cost(B) > 0). */
    public static double kMeansTotalSensitivity(int clusters) {
        if (clusters < 1) {
            throw new IllegalArgumentException("clusters must be >= 1: " + clusters);
        }
        return 8.0 * clusters + 2.0;
    }

    /**
     * Draws so that one fixed query Q has relative error at most eps with probability at least {@code 1 - delta}:
     * {@code m >= S^2 / (2 eps^2) * ln(2 / delta)} (Hoeffding, eq. 2.8).
     */
    public static long singleQuerySampleSize(double totalSensitivity, double eps, double delta) {
        check(totalSensitivity, eps, delta);
        return (long) Math.ceil(totalSensitivity * totalSensitivity / (2.0 * eps * eps) * Math.log(2.0 / delta));
    }

    /**
     * Draws for an eps-coreset (all queries at once) with probability at least {@code 1 - delta}:
     * {@code m >= c * S^2 / eps^2 * (d' + ln(1 / delta))} (Theorem 2.3), d' the pseudo-dimension. The theorem leaves
     * the absolute constant c unspecified, so the result scales with the caller's choice.
     */
    public static long allQueriesSampleSize(double totalSensitivity, double eps, double delta, double pseudoDimension,
                                            double constant) {
        check(totalSensitivity, eps, delta);
        if (!(pseudoDimension >= 0.0) || !(constant > 0.0)) {
            throw new IllegalArgumentException(
                    "pseudoDimension must be >= 0 and constant > 0: " + pseudoDimension + ", " + constant);
        }
        return (long) Math.ceil(constant * totalSensitivity * totalSensitivity / (eps * eps)
                * (pseudoDimension + Math.log(1.0 / delta)));
    }

    /** Pseudo-dimension of k-means in d dimensions, {@code O(d k log k)}, taken here as {@code d k max(1, log2 k)}. */
    public static double kMeansPseudoDimension(int d, int k) {
        if (d < 1 || k < 1) {
            throw new IllegalArgumentException("d and k must be >= 1: d=" + d + ", k=" + k);
        }
        return (double) d * k * Math.max(1.0, Math.log(k) / Math.log(2.0));
    }

    private static void check(double s, double eps, double delta) {
        if (!(s > 0.0) || !(eps > 0.0) || !(delta > 0.0 && delta < 1.0)) {
            throw new IllegalArgumentException(
                    "need S > 0, eps > 0 and 0 < delta < 1: S=" + s + ", eps=" + eps + ", delta=" + delta);
        }
    }
}
