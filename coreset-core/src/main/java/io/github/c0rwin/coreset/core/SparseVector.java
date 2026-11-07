package io.github.c0rwin.coreset.core;

import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * Immutable sparse vector of fixed dimension, stored as strictly increasing indices with non-zero, finite values.
 *
 * <p>A vector either owns its two arrays or is a read-only view of one row of a {@link PointBlock}
 * ({@link PointBlock#vector(int)}); a view costs one small object and shares (and keeps reachable) the block's arrays.
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

    static final int[] NO_INDICES = new int[0];
    static final double[] NO_VALUES = new double[0];

    final int dim;
    final int[] indices;
    final double[] values;
    /** Entries are {@code [from, to)} of the arrays. */
    final int from;
    final int to;
    private final double squaredNorm;
    private int hash;

    /** A vector over {@code (indices, values)[from, to)}, which must already be canonical. */
    SparseVector(int dim, int[] indices, double[] values, int from, int to, double squaredNorm) {
        this.dim = dim;
        this.indices = indices;
        this.values = values;
        this.from = from;
        this.to = to;
        this.squaredNorm = squaredNorm;
    }

    private SparseVector(int dim, int[] indices, double[] values) {
        this(dim, indices, values, 0, indices.length, Entries.squaredNorm(values, 0, values.length));
    }

    /**
     * Builds a vector from coordinate entries. Arrays are copied; entries may be unsorted, duplicate indices are
     * summed (in input order) and resulting zeros are dropped.
     *
     * @throws IllegalArgumentException if {@code dim < 0}, lengths differ, an index is outside {@code [0, dim)}, or a
     *     value (or a sum of duplicates) is not finite
     */
    public static SparseVector of(int dim, int[] indices, double[] values) {
        return build(dim, indices, values, false);
    }

    /**
     * {@link #of} that takes ownership of both arrays instead of copying them: the caller must not read or write them
     * afterwards. Validation and canonicalisation are the same; input that is already sorted, duplicate-free and
     * zero-free is wrapped as is. Meant for decoders that build fresh arrays per vector.
     *
     * @throws IllegalArgumentException as {@link #of}
     */
    public static SparseVector ofOwned(int dim, int[] indices, double[] values) {
        return build(dim, indices, values, true);
    }

    private static SparseVector build(int dim, int[] indices, double[] values, boolean owned) {
        if (dim < 0) {
            throw new IllegalArgumentException("dim must be >= 0: " + dim);
        }
        if (indices.length != values.length) {
            throw new IllegalArgumentException(
                    "indices and values lengths differ: " + indices.length + " vs " + values.length);
        }
        final int n = indices.length;
        int[] idx = owned ? indices : new int[n];
        double[] val = owned ? values : new double[n];
        int w = Entries.canonicalize(dim, indices, values, 0, n, idx, val, 0);
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
        return to - from;
    }

    /** Index of the {@code k}-th stored entry; indices are strictly increasing in {@code k}. */
    public int indexAt(int k) {
        return indices[from + Objects.checkIndex(k, to - from)];
    }

    /** Value of the {@code k}-th stored entry. */
    public double valueAt(int k) {
        return values[from + Objects.checkIndex(k, to - from)];
    }

    /** Coordinate {@code index}, zero if not stored. */
    public double get(int index) {
        if (index < 0 || index >= dim) {
            throw new IndexOutOfBoundsException("index " + index + " out of range [0, " + dim + ")");
        }
        int k = Arrays.binarySearch(indices, from, to, index);
        return k >= 0 ? values[k] : 0.0;
    }

    public double squaredNorm() {
        return squaredNorm;
    }

    public double dot(SparseVector o) {
        checkDim(o.dim);
        return Kernels.dot(indices, values, from, to, o.indices, o.values, o.from, o.to);
    }

    public double dot(double[] dense) {
        checkDim(dense.length);
        double s = 0.0;
        for (int k = from; k < to; k++) {
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
        return Kernels.merge(indices, values, from, to, o.indices, o.values, o.from, o.to);
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
        return Kernels.dense(indices, values, from, to, dense, denseSquaredNorm);
    }

    /**
     * Whether a pass of distances from {@code rows} points holding {@code storedEntries} entries to one center should
     * scatter the center into a dense scratch of {@code dim} doubles ({@link #squaredDistanceScattered}) instead of
     * merging supports: only for sparse rows (under half the coordinates stored on average; denser rows merge as
     * cheaply) and only while the scratch is affordable (up to 2^20 coordinates, beyond that no larger than the input's
     * own values, so memory keeps following the stored entries in very high dimensions).
     */
    public static boolean scatterPays(int dim, int rows, long storedEntries) {
        return 2 * storedEntries < (long) rows * dim && dim <= Math.max(1L << 20, storedEntries);
    }

    /**
     * {@link #squaredDistance(double[], double)} to {@code c} whose entries the caller scattered into {@code scratch}
     * (zero elsewhere), bit-identical to it but O(nnz(this)) lookups. When the cancellation guard trips, a center whose
     * whole support was hit has no off-support term; otherwise the exact O(nnz(this) + nnz(c)) merge replaces the
     * O(dim) sweep. Lets one dense scratch serve a whole pass of distances to the same center.
     */
    public double squaredDistanceScattered(double[] scratch, SparseVector c) {
        checkDim(scratch.length);
        checkDim(c.dim);
        return Kernels.scattered(indices, values, from, to, scratch, c.indices, c.values, c.from, c.to,
                c.squaredNorm);
    }

    /** {@code scratch[i] = this_i} over the stored entries; {@link #clearFrom} undoes it in O(nnz). */
    public void scatterInto(double[] scratch) {
        checkDim(scratch.length);
        Kernels.scatter(indices, values, from, to, scratch);
    }

    /** Zeroes {@code scratch} at this vector's stored coordinates. */
    public void clearFrom(double[] scratch) {
        checkDim(scratch.length);
        Kernels.clear(indices, from, to, scratch);
    }

    /** {@code acc += scale * this}. */
    public void addTo(double[] acc, double scale) {
        checkDim(acc.length);
        Kernels.addTo(indices, values, from, to, acc, scale);
    }

    public double[] toDense() {
        double[] out = new double[dim];
        Kernels.scatter(indices, values, from, to, out);
        return out;
    }

    /**
     * Approximate retained heap size in bytes under a 64-bit HotSpot model with compressed oops and 8-byte alignment:
     * object header plus fields, plus both arrays sized to the stored entries (16-byte array header each), as if
     * owned; a {@link PointBlock} view shares its block's arrays instead. Shared empty arrays are counted.
     */
    public long estimatedBytes() {
        long self = align(OBJECT_HEADER_BYTES + 4L * 4 + 2L * REFERENCE_BYTES + 8L);
        long idx = align(ARRAY_HEADER_BYTES + 4L * nnz());
        long val = align(ARRAY_HEADER_BYTES + 8L * nnz());
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
        return dim == v.dim && Arrays.equals(indices, from, to, v.indices, v.from, v.to)
                && Arrays.equals(values, from, to, v.values, v.from, v.to);
    }

    /** Same value as {@code 31 * (31 * dim + Arrays.hashCode(indices)) + Arrays.hashCode(values)} on owned arrays. */
    @Override
    public int hashCode() {
        int h = hash;
        if (h == 0) {
            int hi = 1;
            int hv = 1;
            for (int k = from; k < to; k++) {
                hi = 31 * hi + indices[k];
                long bits = Double.doubleToLongBits(values[k]);
                hv = 31 * hv + (int) (bits ^ (bits >>> 32));
            }
            h = 31 * (31 * dim + hi) + hv;
            hash = h;
        }
        return h;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("SparseVector[dim=").append(dim).append(", {");
        for (int k = from; k < to; k++) {
            if (k > from) {
                sb.append(", ");
            }
            sb.append(indices[k]).append('=').append(values[k]);
        }
        return sb.append("}]").toString();
    }

    @Serial
    private Object writeReplace() {
        return new SerializedForm(dim, Arrays.copyOfRange(indices, from, to), Arrays.copyOfRange(values, from, to));
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
