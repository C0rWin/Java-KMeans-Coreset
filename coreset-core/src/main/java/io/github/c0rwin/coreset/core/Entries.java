package io.github.c0rwin.coreset.core;

import java.util.Arrays;

/** Validation and canonical form of coordinate entries, shared by {@link SparseVector} and {@link PointBlock}. */
final class Entries {

    private Entries() {
    }

    /**
     * Validates the entries {@code src[from, to)} of a vector of dimension {@code dim} and writes their canonical form
     * (sorted by index, duplicates summed in input order, zeros dropped) to {@code dst[at, at + returned)}. The
     * destination may be the source itself when {@code at <= from}; already sorted input is then compacted in place.
     *
     * @throws IllegalArgumentException if an index is outside {@code [0, dim)} or a value (or a sum of duplicates) is
     *     not finite
     */
    static int canonicalize(int dim, int[] srcIdx, double[] srcVal, int from, int to, int[] dstIdx, double[] dstVal,
                            int at) {
        boolean sorted = true;
        for (int k = from; k < to; k++) {
            int i = srcIdx[k];
            if (i < 0 || i >= dim) {
                throw new IllegalArgumentException("index " + i + " out of range [0, " + dim + ")");
            }
            if (!Double.isFinite(srcVal[k])) {
                throw new IllegalArgumentException("value at index " + i + " is not finite: " + srcVal[k]);
            }
            if (k > from && i <= srcIdx[k - 1]) {
                sorted = false;
            }
        }
        int w = at;
        if (sorted) {
            for (int k = from; k < to; k++) {
                double v = srcVal[k];
                if (v != 0.0) {
                    dstIdx[w] = srcIdx[k];
                    dstVal[w] = v;
                    w++;
                }
            }
            return w - at;
        }
        final int n = to - from;
        // (index, position) packed into a long gives a stable primitive sort without boxing.
        long[] keys = new long[n];
        for (int k = 0; k < n; k++) {
            keys[k] = ((long) srcIdx[from + k] << 32) | k;
        }
        Arrays.sort(keys);
        final double[] vals = srcVal == dstVal ? Arrays.copyOfRange(srcVal, from, to) : srcVal;
        final int base = srcVal == dstVal ? 0 : from;
        for (int r = 0; r < n; ) {
            int i = (int) (keys[r] >>> 32);
            double s = vals[base + (int) keys[r++]];
            while (r < n && (int) (keys[r] >>> 32) == i) {
                s += vals[base + (int) keys[r++]];
            }
            if (s != 0.0) {
                if (!Double.isFinite(s)) {
                    throw new IllegalArgumentException("sum of duplicates at index " + i + " is not finite");
                }
                dstIdx[w] = i;
                dstVal[w] = s;
                w++;
            }
        }
        return w - at;
    }

    /** {@code sum v^2} over {@code values[from, to)} in order: the squared norm every vector and row caches. */
    static double squaredNorm(double[] values, int from, int to) {
        double s = 0.0;
        for (int k = from; k < to; k++) {
            s += values[k] * values[k];
        }
        return s;
    }
}
