package io.github.c0rwin.coreset.core;

import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;

/**
 * Immutable sparse vector of fixed dimension, stored as strictly increasing indices with non-zero, finite values.
 *
 * <p>Equality is value-based: two vectors are equal iff they have the same dimension and the same non-zero entries.
 * Instances are safe to share between threads.
 *
 * <p>Java serialization goes through a proxy that is rebuilt with {@link #of}, so a deserialized vector satisfies the
 * same invariants as a constructed one.
 */
public final class SparseVector implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    /** Memory model used by {@link #estimatedBytes()}: 64-bit HotSpot, compressed oops, 8-byte alignment. */
    static final int OBJECT_HEADER_BYTES = 12;
    static final int ARRAY_HEADER_BYTES = 16;
    static final int REFERENCE_BYTES = 4;

    /**
     * Fraction of {@code ||c||^2} below which {@code ||c||^2 - sum over a support of c_i^2} is not trusted. Summing up
     * to 2^20 squares has relative error at most about 2^-33, so a difference above 2^-16 keeps 17 good bits.
     */
    public static final double CANCELLATION_GUARD = 0x1p-16;

    private static final int[] NO_INDICES = new int[0];
    private static final double[] NO_VALUES = new double[0];

    private final int dim;
    private final int[] indices;
    private final double[] values;
    private final double squaredNorm;
    private int hash;

    private SparseVector(int dim, int[] indices, double[] values) {
        this.dim = dim;
        this.indices = indices;
        this.values = values;
        double s = 0.0;
        for (double v : values) {
            s += v * v;
        }
        this.squaredNorm = s;
    }

    /**
     * Builds a vector from coordinate entries. Arrays are copied; entries may be unsorted, duplicate indices are
     * summed (in input order) and resulting zeros are dropped.
     *
     * @throws IllegalArgumentException if {@code dim < 0}, lengths differ, an index is outside {@code [0, dim)}, or a
     *     value (or a sum of duplicates) is not finite
     */
    public static SparseVector of(int dim, int[] indices, double[] values) {
        if (dim < 0) {
            throw new IllegalArgumentException("dim must be >= 0: " + dim);
        }
        if (indices.length != values.length) {
            throw new IllegalArgumentException(
                    "indices and values lengths differ: " + indices.length + " vs " + values.length);
        }
        final int n = indices.length;
        boolean sorted = true;
        for (int k = 0; k < n; k++) {
            int i = indices[k];
            if (i < 0 || i >= dim) {
                throw new IllegalArgumentException("index " + i + " out of range [0, " + dim + ")");
            }
            if (!Double.isFinite(values[k])) {
                throw new IllegalArgumentException("value at index " + i + " is not finite: " + values[k]);
            }
            if (k > 0 && i <= indices[k - 1]) {
                sorted = false;
            }
        }
        int[] idx;
        double[] val;
        if (sorted) {
            idx = indices.clone();
            val = values.clone();
        } else {
            // (index, position) packed into a long gives a stable primitive sort without boxing.
            long[] keys = new long[n];
            for (int k = 0; k < n; k++) {
                keys[k] = ((long) indices[k] << 32) | k;
            }
            Arrays.sort(keys);
            idx = new int[n];
            val = new double[n];
            for (int k = 0; k < n; k++) {
                idx[k] = (int) (keys[k] >>> 32);
                val[k] = values[(int) keys[k]];
            }
        }
        int w = 0;
        for (int r = 0; r < n; ) {
            int i = idx[r];
            double s = val[r++];
            while (r < n && idx[r] == i) {
                s += val[r++];
            }
            if (s != 0.0) {
                if (!Double.isFinite(s)) {
                    throw new IllegalArgumentException("sum of duplicates at index " + i + " is not finite");
                }
                idx[w] = i;
                val[w] = s;
                w++;
            }
        }
        if (w == 0) {
            return new SparseVector(dim, NO_INDICES, NO_VALUES);
        }
        if (w < n) {
            idx = Arrays.copyOf(idx, w);
            val = Arrays.copyOf(val, w);
        }
        return new SparseVector(dim, idx, val);
    }

    /**
     * Builds a vector of dimension {@code values.length} keeping only non-zero entries.
     *
     * @throws IllegalArgumentException if any value is not finite
     */
    public static SparseVector dense(double[] values) {
        int nnz = 0;
        for (double v : values) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("value is not finite: " + v);
            }
            if (v != 0.0) {
                nnz++;
            }
        }
        if (nnz == 0) {
            return new SparseVector(values.length, NO_INDICES, NO_VALUES);
        }
        int[] idx = new int[nnz];
        double[] val = new double[nnz];
        int w = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i] != 0.0) {
                idx[w] = i;
                val[w] = values[i];
                w++;
            }
        }
        return new SparseVector(values.length, idx, val);
    }

    public int dim() {
        return dim;
    }

    /** Number of stored (non-zero) entries. */
    public int nnz() {
        return indices.length;
    }

    /** Index of the {@code k}-th stored entry; indices are strictly increasing in {@code k}. */
    public int indexAt(int k) {
        return indices[k];
    }

    /** Value of the {@code k}-th stored entry. */
    public double valueAt(int k) {
        return values[k];
    }

    /** Coordinate {@code index}, zero if not stored. */
    public double get(int index) {
        if (index < 0 || index >= dim) {
            throw new IndexOutOfBoundsException("index " + index + " out of range [0, " + dim + ")");
        }
        int k = Arrays.binarySearch(indices, index);
        return k >= 0 ? values[k] : 0.0;
    }

    public double squaredNorm() {
        return squaredNorm;
    }

    public double dot(SparseVector o) {
        checkDim(o.dim);
        final int[] ai = indices;
        final int[] bi = o.indices;
        final double[] av = values;
        final double[] bv = o.values;
        int i = 0;
        int j = 0;
        double s = 0.0;
        while (i < ai.length && j < bi.length) {
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

    public double dot(double[] dense) {
        checkDim(dense.length);
        double s = 0.0;
        for (int k = 0; k < indices.length; k++) {
            s += values[k] * dense[indices[k]];
        }
        return s;
    }

    /** Exact squared Euclidean distance computed by merging supports, so no cancellation from norm expansion. */
    public double squaredDistance(SparseVector o) {
        checkDim(o.dim);
        if (o == this) {
            return 0.0;
        }
        final int[] ai = indices;
        final int[] bi = o.indices;
        final double[] av = values;
        final double[] bv = o.values;
        int i = 0;
        int j = 0;
        double s = 0.0;
        while (i < ai.length && j < bi.length) {
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
        while (i < ai.length) {
            double d = av[i++];
            s += d * d;
        }
        while (j < bi.length) {
            double d = bv[j++];
            s += d * d;
        }
        return s;
    }

    /**
     * Squared distance to a dense vector in O(nnz) time, given {@code denseSquaredNorm = ||dense||^2}.
     *
     * <p>The terms over this vector's support are summed directly. The remaining coordinates contribute
     * {@code ||dense||^2} minus the support's share of it; when that difference is below
     * {@value #CANCELLATION_GUARD} of the norm it may be mostly rounding error, so it is recomputed exactly in O(dim)
     * instead. The result is therefore accurate far from the origin, where the plain expansion would cancel.
     */
    public double squaredDistance(double[] dense, double denseSquaredNorm) {
        checkDim(dense.length);
        double onSupport = 0.0;
        double denseOnSupport = 0.0;
        for (int k = 0; k < indices.length; k++) {
            double c = dense[indices[k]];
            double d = values[k] - c;
            onSupport += d * d;
            denseOnSupport += c * c;
        }
        double offSupport = denseSquaredNorm - denseOnSupport;
        if (offSupport > CANCELLATION_GUARD * denseSquaredNorm) {
            return onSupport + offSupport;
        }
        return onSupport + offSupportSquaredNorm(dense);
    }

    /** {@code sum of dense[i]^2} over the coordinates outside this vector's support, in index order. */
    private double offSupportSquaredNorm(double[] dense) {
        if (indices.length == dense.length) {
            return 0.0;
        }
        double s = 0.0;
        int k = 0;
        for (int i = 0; i < dense.length; i++) {
            if (k < indices.length && indices[k] == i) {
                k++;
            } else {
                s += dense[i] * dense[i];
            }
        }
        return s;
    }

    /** {@code acc += scale * this}. */
    public void addTo(double[] acc, double scale) {
        checkDim(acc.length);
        for (int k = 0; k < indices.length; k++) {
            acc[indices[k]] += scale * values[k];
        }
    }

    public double[] toDense() {
        double[] out = new double[dim];
        for (int k = 0; k < indices.length; k++) {
            out[indices[k]] = values[k];
        }
        return out;
    }

    /**
     * Approximate retained heap size in bytes under a 64-bit HotSpot model with compressed oops and 8-byte alignment:
     * object header plus fields, plus both backing arrays (16-byte array header each). Shared empty arrays are counted.
     */
    public long estimatedBytes() {
        long self = align(OBJECT_HEADER_BYTES + 4L + 4L + 2L * REFERENCE_BYTES + 8L);
        long idx = align(ARRAY_HEADER_BYTES + 4L * indices.length);
        long val = align(ARRAY_HEADER_BYTES + 8L * values.length);
        return self + idx + val;
    }

    static long align(long bytes) {
        return (bytes + 7L) & ~7L;
    }

    private void checkDim(int otherDim) {
        if (otherDim != dim) {
            throw new IllegalArgumentException("dimension mismatch: " + dim + " vs " + otherDim);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof SparseVector v)) {
            return false;
        }
        return dim == v.dim && Arrays.equals(indices, v.indices) && Arrays.equals(values, v.values);
    }

    @Override
    public int hashCode() {
        int h = hash;
        if (h == 0) {
            h = 31 * (31 * dim + Arrays.hashCode(indices)) + Arrays.hashCode(values);
            hash = h;
        }
        return h;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("SparseVector[dim=").append(dim).append(", {");
        for (int k = 0; k < indices.length; k++) {
            if (k > 0) {
                sb.append(", ");
            }
            sb.append(indices[k]).append('=').append(values[k]);
        }
        return sb.append("}]").toString();
    }

    @Serial
    private Object writeReplace() {
        return new SerializedForm(dim, indices, values);
    }

    @Serial
    private void readObject(ObjectInputStream in) throws InvalidObjectException {
        throw new InvalidObjectException("SparseVector is deserialized only through its serialized form");
    }

    private record SerializedForm(int dim, int[] indices, double[] values) implements Serializable {

        @Serial
        private Object readResolve() throws InvalidObjectException {
            try {
                return SparseVector.of(dim, indices, values);
            } catch (IllegalArgumentException | NullPointerException e) {
                InvalidObjectException ex = new InvalidObjectException("invalid SparseVector: " + e.getMessage());
                ex.initCause(e);
                throw ex;
            }
        }
    }
}
