package io.github.c0rwin.coreset.core;

import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Weighted points of one dimension in compressed sparse row (CSR) form: row offsets, column indices, values, weights
 * and cached row squared norms, five flat arrays in total. Row {@code i} holds the entries
 * {@code [rowStart(i), rowEnd(i))}, with strictly increasing indices in {@code [0, dim)} and non-zero finite values;
 * weights are finite and {@code >= 0}. Segments of a {@link PointSet} hold the points of a {@link Coreset}: the
 * algorithms stream the arrays through the row kernels below and never create per-point objects in their inner loops.
 *
 * <p>Immutable and safe to share between threads. {@link #vector(int)} and {@link #point(int)} materialise rows on
 * demand as views that share the arrays. Every empty block is {@link #EMPTY} (dimension 0).
 * Java serialization goes through a proxy rebuilt with {@link #wrap}, so a deserialized block is re-validated.
 */
public final class PointBlock implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final double[] NO_DOUBLES = new double[0];

    /** The block without points. */
    public static final PointBlock EMPTY = new PointBlock(0, 0, 0, new int[1], SparseVector.NO_INDICES,
            SparseVector.NO_VALUES, NO_DOUBLES, NO_DOUBLES);

    private final int dim;
    private final int size;
    private final int base; // rows are [base, base + size) of offsets, weights and norms (non-zero for slices)
    private final int[] offsets; // at least base + size + 1 entries, absolute positions in indices and values
    private final int[] indices;
    private final double[] values;
    private final double[] weights;
    private final double[] norms;
    private int hash;

    private PointBlock(int dim, int size, int base, int[] offsets, int[] indices, double[] values, double[] weights,
                       double[] norms) {
        this.dim = dim;
        this.size = size;
        this.base = base;
        this.offsets = offsets;
        this.indices = indices;
        this.values = values;
        this.weights = weights;
        this.norms = norms;
    }

    /**
     * Copies {@code points} into one block: two passes (sizes and dimension check, then the copy), five allocations.
     *
     * @throws IllegalArgumentException if the points differ in dimension or hold more than 2^31 - 9 entries in total
     * @throws NullPointerException if a point is null
     */
    public static PointBlock of(List<WeightedPoint> points) {
        final int n = points.size();
        if (n == 0) {
            return EMPTY;
        }
        final int dim = points.get(0).vector().dim;
        long entries = 0;
        for (int i = 0; i < n; i++) {
            SparseVector v = points.get(i).vector();
            if (v.dim != dim) {
                throw new IllegalArgumentException("point " + i + " has dim " + v.dim + ", expected " + dim);
            }
            entries += v.to - v.from;
        }
        Builder b = new Builder(dim, n, entries);
        for (int i = 0; i < n; i++) {
            WeightedPoint p = points.get(i);
            b.append(p.vector(), p.weight());
        }
        return b.build();
    }

    /**
     * Takes ownership of the arrays (the caller must not use them afterwards), validates them and canonicalises every
     * row in place as {@link SparseVector#of} would (sorting, summing duplicates, dropping zeros, compacting later rows
     * forward); rows that are already canonical are not touched. Meant for decoders, which can fill exactly sized
     * arrays and so build a block without any further copy. {@code indices} and {@code values} may be longer than
     * {@code rowOffsets[n]}.
     *
     * @throws IllegalArgumentException if {@code dim < 0}, {@code rowOffsets} is not {@code n + 1} non-decreasing
     *     offsets from 0 into both entry arrays for {@code n = weights.length}, a weight is negative or not finite,
     *     or an entry is invalid (see {@link SparseVector#of})
     */
    public static PointBlock wrap(int dim, int[] rowOffsets, int[] indices, double[] values, double[] weights) {
        final int n = weights.length;
        if (dim < 0) {
            throw new IllegalArgumentException("dim must be >= 0: " + dim);
        }
        if (rowOffsets.length != n + 1 || rowOffsets[0] != 0) {
            throw new IllegalArgumentException("need " + (n + 1) + " row offsets starting at 0, got "
                    + rowOffsets.length);
        }
        for (int i = 0; i < n; i++) {
            if (rowOffsets[i + 1] < rowOffsets[i]) {
                throw new IllegalArgumentException("row offsets decrease at row " + i);
            }
            double w = weights[i];
            if (!Double.isFinite(w) || w < 0.0) {
                throw new IllegalArgumentException("weight must be finite and >= 0: " + w);
            }
        }
        if (rowOffsets[n] > indices.length || rowOffsets[n] > values.length) {
            throw new IllegalArgumentException(rowOffsets[n] + " entries, arrays hold " + indices.length + " and "
                    + values.length);
        }
        if (n == 0) {
            return EMPTY;
        }
        double[] norms = new double[n];
        int w = 0;
        for (int i = 0; i < n; i++) {
            int from = rowOffsets[i];
            int to = rowOffsets[i + 1];
            rowOffsets[i] = w;
            int nnz = Entries.canonicalize(dim, indices, values, from, to, indices, values, w);
            norms[i] = Entries.squaredNorm(values, w, w + nnz);
            w += nnz;
        }
        rowOffsets[n] = w;
        return new PointBlock(dim, n, 0, rowOffsets, indices, values, weights, norms);
    }

    /**
     * Concatenates the blocks in order: one allocation per array, every entry copied (see {@link PointSet#concat} for
     * the copy-free form).
     *
     * @throws IllegalArgumentException if two non-empty blocks differ in dimension or the result would hold more than
     *     2^31 - 9 points or entries
     */
    public static PointBlock concat(List<PointBlock> parts) {
        int dim = -1;
        long rows = 0;
        long entries = 0;
        PointBlock only = null;
        int nonEmpty = 0;
        for (PointBlock p : parts) {
            if (p.size == 0) {
                continue;
            }
            if (dim < 0) {
                dim = p.dim;
            } else if (p.dim != dim) {
                throw new IllegalArgumentException("block has dim " + p.dim + ", expected " + dim);
            }
            rows += p.size;
            entries += p.storedEntries();
            only = p;
            nonEmpty++;
        }
        if (nonEmpty <= 1) {
            return only == null ? EMPTY : only;
        }
        Builder b = new Builder(dim, checkedSize(rows, "points"), entries);
        for (PointBlock p : parts) {
            b.appendAll(p);
        }
        return b.build();
    }

    /** Appends row {@code i} with {@code weight} to {@code b}: one {@code arraycopy} per entry array. */
    void appendRowTo(int i, double weight, Builder b) {
        b.appendRow(indices, values, offsets[base + i], offsets[base + i + 1], norms[base + i], weight);
    }

    /** Same entries (bitwise values) and weight as row {@code j} of {@code o}. */
    boolean rowEquals(int i, PointBlock o, int j) {
        int from = offsets[base + i];
        int to = offsets[base + i + 1];
        int oFrom = o.offsets[o.base + j];
        int oTo = o.offsets[o.base + j + 1];
        return Double.doubleToLongBits(weights[base + i]) == Double.doubleToLongBits(o.weights[o.base + j])
                && Arrays.equals(indices, from, to, o.indices, oFrom, oTo)
                && Arrays.equals(values, from, to, o.values, oFrom, oTo);
    }

    /** Hash of row {@code i}'s entries and weight, independent of where the row is stored. */
    int rowHash(int i) {
        int h = 1;
        for (int e = offsets[base + i], to = offsets[base + i + 1]; e < to; e++) {
            long bits = Double.doubleToLongBits(values[e]);
            h = 31 * (31 * h + indices[e]) + (int) (bits ^ (bits >>> 32));
        }
        long bits = Double.doubleToLongBits(weights[base + i]);
        return 31 * h + (int) (bits ^ (bits >>> 32));
    }

    /** Builder for a block of dimension {@code dim}; the expected sizes only presize its arrays. */
    public static Builder builder(int dim, int expectedRows, long expectedEntries) {
        if (dim < 0) {
            throw new IllegalArgumentException("dim must be >= 0: " + dim);
        }
        return new Builder(dim, Math.max(0, expectedRows), Math.max(0, expectedEntries));
    }

    public int dim() {
        return dim;
    }

    /** Number of points (rows). */
    public int size() {
        return size;
    }

    public boolean isEmpty() {
        return size == 0;
    }

    /** Total stored entries over all rows. */
    public long storedEntries() {
        return offsets[base + size] - offsets[base];
    }

    /**
     * Rows {@code [from, to)} as a block sharing all of this block's arrays: O(1), nothing copied. The slice keeps
     * the arrays reachable, so {@link #estimatedBytes()} counts them in full.
     *
     * @throws IndexOutOfBoundsException unless {@code 0 <= from <= to <= size()}
     */
    public PointBlock slice(int from, int to) {
        Objects.checkFromToIndex(from, to, size);
        if (from == 0 && to == size) {
            return this;
        }
        if (from == to) {
            return EMPTY;
        }
        return new PointBlock(dim, to - from, base + from, offsets, indices, values, weights, norms);
    }

    public double weight(int i) {
        return weights[base + i];
    }

    /** {@code ||row i||^2}, cached. */
    public double squaredNorm(int i) {
        return norms[base + i];
    }

    /** First entry of row {@code i}; its entries are {@code [rowStart(i), rowEnd(i))}. */
    public int rowStart(int i) {
        return offsets[base + i];
    }

    public int rowEnd(int i) {
        return offsets[base + i + 1];
    }

    /** Stored entries of row {@code i}. */
    public int nnz(int i) {
        return offsets[base + i + 1] - offsets[base + i];
    }

    /** Coordinate of entry {@code e} (an entry position, not a row). */
    public int index(int e) {
        return indices[e];
    }

    /** Value of entry {@code e}. */
    public double value(int e) {
        return values[e];
    }

    /** Sum of the weights in row order. */
    public double totalWeight() {
        double s = 0.0;
        for (int i = 0; i < size; i++) {
            s += weights[base + i];
        }
        return s;
    }

    /** Row {@code i} as a vector view sharing this block's arrays (one small object, no copy). */
    public SparseVector vector(int i) {
        return new SparseVector(dim, indices, values, offsets[base + i], offsets[base + i + 1], norms[base + i]);
    }

    public WeightedPoint point(int i) {
        return new WeightedPoint(vector(i), weights[base + i]);
    }

    /** {@code ||row i - c||^2} by merging supports: bit-identical to {@code vector(i).squaredDistance(c)}. */
    public double squaredDistance(int i, SparseVector c) {
        checkDim(c.dim);
        return Kernels.merge(indices, values, offsets[base + i], offsets[base + i + 1], c.indices, c.values, c.from,
                c.to);
    }

    /** {@code ||row i - row j of o||^2} by merging supports ({@code o} may be this block). */
    public double squaredDistance(int i, PointBlock o, int j) {
        checkDim(o.dim);
        return Kernels.merge(indices, values, offsets[base + i], offsets[base + i + 1], o.indices, o.values,
                o.offsets[o.base + j], o.offsets[o.base + j + 1]);
    }

    /** Bit-identical to {@code vector(i).squaredDistance(dense, denseSquaredNorm)}. */
    public double squaredDistance(int i, double[] dense, double denseSquaredNorm) {
        checkDim(dense.length);
        return Kernels.dense(indices, values, offsets[base + i], offsets[base + i + 1], dense, denseSquaredNorm);
    }

    /**
     * {@code ||row i - c||^2} for a center {@code c} given by its sorted entries {@code (cIdx, cVal)[cFrom, cTo)} and
     * squared norm, which the caller scattered into {@code scratch}: O(nnz(row)) lookups, bit-identical to
     * {@link #squaredDistance(int, double[], double)} on the scattered center (see
     * {@link SparseVector#squaredDistanceScattered}).
     */
    public double squaredDistanceScattered(int i, double[] scratch, int[] cIdx, double[] cVal, int cFrom, int cTo,
                                           double cNorm) {
        checkDim(scratch.length);
        return Kernels.scattered(indices, values, offsets[base + i], offsets[base + i + 1], scratch, cIdx, cVal, cFrom,
                cTo, cNorm);
    }

    /**
     * {@link #squaredDistanceScattered(int, double[], int[], double[], int, int, double)} without a scratch: the
     * center's non-zero values are looked up in its sorted coordinates, bit-identically, in O(nnz(row) log nnz(c)).
     */
    public double squaredDistanceSorted(int i, int[] cIdx, double[] cVal, int cFrom, int cTo, double cNorm) {
        return Kernels.lookup(indices, values, offsets[base + i], offsets[base + i + 1], cIdx, cVal, cFrom, cTo, cNorm);
    }

    /** {@link #squaredDistanceScattered(int, double[], int[], double[], int, int, double)} to a scattered vector. */
    public double squaredDistanceScattered(int i, double[] scratch, SparseVector c) {
        checkDim(c.dim);
        return squaredDistanceScattered(i, scratch, c.indices, c.values, c.from, c.to, c.squaredNorm());
    }

    /**
     * {@link #squaredDistanceScattered(int, double[], int[], double[], int, int, double)} to row j of {@code o}, which
     * the caller scattered into {@code scratch} ({@code o} may be this block).
     */
    public double squaredDistanceScatteredToRow(int i, double[] scratch, PointBlock o, int j) {
        checkDim(scratch.length);
        return Kernels.scattered(indices, values, offsets[base + i], offsets[base + i + 1], scratch, o.indices,
                o.values, o.offsets[o.base + j], o.offsets[o.base + j + 1], o.norms[o.base + j]);
    }

    /** {@code sum of c^2} over the entries of row {@code j} outside the support of row {@code i} of {@code p}. */
    public double offSupport(int j, PointBlock p, int i) {
        return Kernels.offSupportSparse(p.indices, p.offsets[p.base + i], p.offsets[p.base + i + 1], indices, values,
                offsets[base + j], offsets[base + j + 1]);
    }

    /** {@code scratch[c] = value} for every entry of row {@code j}; {@link #clearRow} undoes it. */
    public void scatterRow(int j, double[] scratch) {
        checkDim(scratch.length);
        Kernels.scatter(indices, values, offsets[base + j], offsets[base + j + 1], scratch);
    }

    public void clearRow(int j, double[] scratch) {
        checkDim(scratch.length);
        Kernels.clear(indices, offsets[base + j], offsets[base + j + 1], scratch);
    }

    /** {@code acc += scale * row i}. */
    public void addTo(int i, double[] acc, double scale) {
        checkDim(acc.length);
        Kernels.addTo(indices, values, offsets[base + i], offsets[base + i + 1], acc, scale);
    }

    /**
     * Approximate retained heap size in bytes under the {@link SparseVector#estimatedBytes()} model: this object and
     * its five arrays at their allocated lengths (shared in full with any block it was sliced from).
     */
    public long estimatedBytes() {
        long self = SparseVector.align(SparseVector.OBJECT_HEADER_BYTES + 4L * 4 + 5L * SparseVector.REFERENCE_BYTES);
        return self + intArray(offsets.length) + intArray(indices.length) + doubleArray(values.length)
                + doubleArray(weights.length) + doubleArray(norms.length);
    }

    private static long intArray(int n) {
        return SparseVector.align(SparseVector.ARRAY_HEADER_BYTES + 4L * n);
    }

    private static long doubleArray(int n) {
        return SparseVector.align(SparseVector.ARRAY_HEADER_BYTES + 8L * n);
    }

    private void checkDim(int otherDim) {
        if (otherDim != dim) {
            throw new IllegalArgumentException("dimension mismatch: " + dim + " vs " + otherDim);
        }
    }

    private static int checkedSize(long n, String what) {
        if (n > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("more than 2^31 " + what + ": " + n);
        }
        return (int) n;
    }

    /** Value-based: same dimension, and the same points (entries and weights) in the same order. */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof PointBlock b) || dim != b.dim || size != b.size) {
            return false;
        }
        for (int i = 0; i < size; i++) {
            if (!rowEquals(i, b, i)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        int h = hash;
        if (h == 0) {
            h = dim;
            for (int i = 0; i < size; i++) {
                h = 31 * h + rowHash(i);
            }
            hash = h;
        }
        return h;
    }

    @Override
    public String toString() {
        return "PointBlock[dim=" + dim + ", points=" + size + ", entries=" + storedEntries() + "]";
    }

    @Serial
    private Object writeReplace() {
        int first = offsets[base];
        int[] rows = new int[size + 1];
        for (int i = 1; i <= size; i++) {
            rows[i] = offsets[base + i] - first;
        }
        int last = offsets[base + size];
        return new SerializedForm(dim, rows, Arrays.copyOfRange(indices, first, last),
                Arrays.copyOfRange(values, first, last), Arrays.copyOfRange(weights, base, base + size));
    }

    @Serial
    private void readObject(ObjectInputStream in) throws InvalidObjectException {
        throw new InvalidObjectException("PointBlock is deserialized only through its serialized form");
    }

    private record SerializedForm(int dim, int[] offsets, int[] indices, double[] values, double[] weights)
            implements Serializable {

        @Serial
        private Object readResolve() throws InvalidObjectException {
            try {
                return wrap(dim, offsets, indices, values, weights);
            } catch (IllegalArgumentException | NullPointerException | ArrayIndexOutOfBoundsException e) {
                InvalidObjectException ex = new InvalidObjectException("invalid PointBlock: " + e.getMessage());
                ex.initCause(e);
                throw ex;
            }
        }
    }

    /**
     * Appends points to growing arrays (doubling) and trims them once in {@link #build}; with exact expected sizes
     * nothing is copied twice. Not thread-safe; build at most once.
     */
    public static final class Builder {
        private final int dim;
        private int size;
        private int entries;
        private int[] offsets;
        private int[] indices;
        private double[] values;
        private double[] weights;
        private double[] norms;

        private Builder(int dim, int expectedRows, long expectedEntries) {
            this.dim = dim;
            offsets = new int[checkedSize((long) expectedRows + 1, "points")];
            indices = new int[checkedSize(expectedEntries, "entries")];
            values = new double[indices.length];
            weights = new double[expectedRows];
            norms = new double[expectedRows];
        }

        public int dim() {
            return dim;
        }

        /** Number of points added so far. */
        public int size() {
            return size;
        }

        /**
         * Appends {@code v} with {@code weight}.
         *
         * @throws IllegalArgumentException if the dimension differs or the weight is negative or not finite
         */
        public Builder add(SparseVector v, double weight) {
            if (v.dim != dim) {
                throw new IllegalArgumentException("point " + size + " has dim " + v.dim + ", expected " + dim);
            }
            checkWeight(weight);
            append(v, weight);
            return this;
        }

        public Builder add(WeightedPoint p) {
            return add(p.vector(), p.weight());
        }

        /**
         * Appends the entries {@code (idx, val)[from, to)}, validated and canonicalised as {@link SparseVector#of}
         * does (they may be unsorted), with {@code weight}.
         *
         * @throws IllegalArgumentException if an entry or the weight is invalid
         */
        public Builder addRow(int[] idx, double[] val, int from, int to, double weight) {
            checkWeight(weight);
            ensure(1, to - from);
            int nnz = Entries.canonicalize(dim, idx, val, from, to, indices, values, entries);
            push(nnz, Entries.squaredNorm(values, entries, entries + nnz), weight);
            return this;
        }

        /** Appends every point of {@code b}, which must have this builder's dimension unless it is empty. */
        public Builder addAll(PointBlock b) {
            if (b.size > 0 && b.dim != dim) {
                throw new IllegalArgumentException("block has dim " + b.dim + ", expected " + dim);
            }
            appendAll(b);
            return this;
        }

        public PointBlock build() {
            if (offsets == null) {
                throw new IllegalStateException("already built");
            }
            PointBlock out = size == 0 ? EMPTY : new PointBlock(dim, size, 0, trim(offsets, size + 1),
                    trim(indices, entries), trim(values, entries), trim(weights, size), trim(norms, size));
            offsets = null;
            return out;
        }

        void append(SparseVector v, double weight) {
            appendRow(v.indices, v.values, v.from, v.to, v.squaredNorm(), weight);
        }

        void appendRow(int[] idx, double[] val, int from, int to, double norm, double weight) {
            checkWeight(weight);
            int nnz = to - from;
            ensure(1, nnz);
            System.arraycopy(idx, from, indices, entries, nnz);
            System.arraycopy(val, from, values, entries, nnz);
            push(nnz, norm, weight);
        }

        void appendAll(PointBlock b) {
            int n = b.size;
            int first = b.offsets[b.base];
            int nnz = b.offsets[b.base + n] - first;
            ensure(n, nnz);
            System.arraycopy(b.indices, first, indices, entries, nnz);
            System.arraycopy(b.values, first, values, entries, nnz);
            System.arraycopy(b.weights, b.base, weights, size, n);
            System.arraycopy(b.norms, b.base, norms, size, n);
            for (int i = 1; i <= n; i++) {
                offsets[size + i] = entries + b.offsets[b.base + i] - first;
            }
            size += n;
            entries += nnz;
        }

        private void push(int nnz, double norm, double weight) {
            weights[size] = weight;
            norms[size] = norm;
            entries += nnz;
            offsets[++size] = entries;
        }

        private void ensure(int rows, int nnz) {
            if (offsets == null) {
                throw new IllegalStateException("already built");
            }
            if (weights.length - size < rows) {
                int cap = checkedSize(Math.max((long) size + rows, 2L * weights.length), "points");
                weights = Arrays.copyOf(weights, cap);
                norms = Arrays.copyOf(norms, cap);
                offsets = Arrays.copyOf(offsets, cap + 1);
            }
            if (indices.length - entries < nnz) {
                int cap = checkedSize(Math.max((long) entries + nnz, 2L * indices.length), "entries");
                indices = Arrays.copyOf(indices, cap);
                values = Arrays.copyOf(values, cap);
            }
        }

        private static void checkWeight(double w) {
            if (!Double.isFinite(w) || w < 0.0) {
                throw new IllegalArgumentException("weight must be finite and >= 0: " + w);
            }
        }

        private static int[] trim(int[] a, int n) {
            return a.length == n ? a : Arrays.copyOf(a, n);
        }

        private static double[] trim(double[] a, int n) {
            return a.length == n ? a : Arrays.copyOf(a, n);
        }
    }
}
