package io.github.c0rwin.coreset.core.kmeans;

/**
 * Rounding margins that let triangle-inequality bounds skip distance evaluations without changing any result: a bound
 * may prune only when it holds even after every computed distance is off by its worst-case relative error.
 */
final class Bounds {

    /** Below this a squared distance may have lost relative accuracy to underflow, so it is never used to prune. */
    static final double TINY_SQUARED = 0x1p-1000;

    private Bounds() {
    }

    /**
     * Relative error bound of a squared distance from the row kernels when the point and the center store at most
     * {@code terms} coordinates together. The worst path takes {@code ||c||^2 - sum_support c_i^2} only above 2^-16 of
     * {@code ||c||^2}, so its O(terms) eps error grows by at most 2^16.
     */
    static double relativeError(long terms) {
        return (terms + 4.0) * 0x1p-36;
    }

    /** Whether {@code relativeError} is small enough for the first-order margins to hold (terms under ~2^24). */
    static boolean usable(double relativeError) {
        return relativeError <= 0x1p-12;
    }
}
