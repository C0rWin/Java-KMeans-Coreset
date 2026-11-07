package io.github.c0rwin.coreset.core;

/**
 * Distance kernels over entry ranges {@code (indices, values)[from, to)} with strictly increasing indices, shared by
 * {@link SparseVector} and {@link PointBlock} rows so both perform the same floating-point operations in the same
 * order. Callers check dimensions.
 */
final class Kernels {

    private Kernels() {
    }

    /** Exact squared distance by merging supports; no cancellation from norm expansion. */
    static double merge(int[] ai, double[] av, int i, int iEnd, int[] bi, double[] bv, int j, int jEnd) {
        double s = 0.0;
        while (i < iEnd && j < jEnd) {
            int x = ai[i];
            int y = bi[j];
            if (x == y) {
                double d = av[i++] - bv[j++];
                s += d * d;
            } else if (x < y) {
                double d = av[i++];
                s += d * d;
            } else {
                double d = bv[j++];
                s += d * d;
            }
        }
        while (i < iEnd) {
            double d = av[i++];
            s += d * d;
        }
        while (j < jEnd) {
            double d = bv[j++];
            s += d * d;
        }
        return s;
    }

    static double dot(int[] ai, double[] av, int i, int iEnd, int[] bi, double[] bv, int j, int jEnd) {
        double s = 0.0;
        while (i < iEnd && j < jEnd) {
            int x = ai[i];
            int y = bi[j];
            if (x == y) {
                s += av[i++] * bv[j++];
            } else if (x < y) {
                i++;
            } else {
                j++;
            }
        }
        return s;
    }

    /** See {@link SparseVector#squaredDistance(double[], double)}. */
    static double dense(int[] ai, double[] av, int from, int to, double[] dense, double denseSquaredNorm) {
        double onSupport = 0.0;
        double denseOnSupport = 0.0;
        for (int k = from; k < to; k++) {
            double c = dense[ai[k]];
            double d = av[k] - c;
            onSupport += d * d;
            denseOnSupport += c * c;
        }
        double offSupport = denseSquaredNorm - denseOnSupport;
        if (offSupport > SparseVector.CANCELLATION_GUARD * denseSquaredNorm) {
            return onSupport + offSupport;
        }
        return onSupport + offSupportDense(ai, from, to, dense);
    }

    /** {@code sum of dense[i]^2} over the coordinates outside the support {@code ai[from, to)}, in index order. */
    private static double offSupportDense(int[] ai, int from, int to, double[] dense) {
        if (to - from == dense.length) {
            return 0.0;
        }
        double s = 0.0;
        int k = from;
        for (int i = 0; i < dense.length; i++) {
            if (k < to && ai[k] == i) {
                k++;
            } else {
                s += dense[i] * dense[i];
            }
        }
        return s;
    }

    /**
     * {@link #dense} to the center {@code (ci, cv)[cFrom, cTo)} with squared norm {@code cNorm}, scattered into
     * {@code scratch} (zero elsewhere): bit-identical to {@link #dense} on {@code scratch}, but O(to - from) lookups.
     * When the guard trips, a center whose whole support was hit has no off-support term; otherwise the exact
     * O(nnz + nnz(c)) merge replaces the O(dim) sweep.
     */
    static double scattered(int[] ai, double[] av, int from, int to, double[] scratch, int[] ci, double[] cv,
                            int cFrom, int cTo, double cNorm) {
        double onSupport = 0.0;
        double centerOnSupport = 0.0;
        int hits = 0;
        for (int k = from; k < to; k++) {
            double x = scratch[ai[k]];
            double d = av[k] - x;
            onSupport += d * d;
            centerOnSupport += x * x;
            hits += x != 0.0 ? 1 : 0; // stored center values are non-zero
        }
        double offSupport = cNorm - centerOnSupport;
        if (offSupport > SparseVector.CANCELLATION_GUARD * cNorm) {
            return onSupport + offSupport;
        }
        return hits == cTo - cFrom ? onSupport : onSupport + offSupportSparse(ai, from, to, ci, cv, cFrom, cTo);
    }

    /** {@code sum of c_i^2} over the center's stored coordinates outside the support {@code ai[from, to)}. */
    static double offSupportSparse(int[] ai, int from, int to, int[] ci, double[] cv, int cFrom, int cTo) {
        double s = 0.0;
        int k = from;
        for (int t = cFrom; t < cTo; t++) {
            int i = ci[t];
            while (k < to && ai[k] < i) {
                k++;
            }
            if (k == to || ai[k] != i) {
                s += cv[t] * cv[t];
            }
        }
        return s;
    }

    static void scatter(int[] ai, double[] av, int from, int to, double[] scratch) {
        for (int k = from; k < to; k++) {
            scratch[ai[k]] = av[k];
        }
    }

    static void clear(int[] ai, int from, int to, double[] scratch) {
        for (int k = from; k < to; k++) {
            scratch[ai[k]] = 0.0;
        }
    }

    static void addTo(int[] ai, double[] av, int from, int to, double[] acc, double scale) {
        for (int k = from; k < to; k++) {
            acc[ai[k]] += scale * av[k];
        }
    }
}
