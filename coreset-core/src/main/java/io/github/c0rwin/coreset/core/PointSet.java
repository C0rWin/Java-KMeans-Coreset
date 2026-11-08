package io.github.c0rwin.coreset.core;

import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.Serial;
import java.io.Serializable;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.RandomAccess;

/**
 * The weighted points of a {@link Coreset}: an immutable sequence of non-empty {@link PointBlock} segments of one
 * dimension. Global row {@code r} is row {@code r - start(s)} of segment {@code s = segmentOf(r)}. Concatenation
 * ({@link #concat}) shares the segments, so unions cost O(segments) and copy no entries; the algorithms stream the
 * segments in order, so their results do not depend on how the rows are segmented.
 *
 * <p>Safe to share between threads. Equality is value-based over the rows (entries and weights) in order, whatever the
 * segmentation. Java serialization stores the segments, each re-validated on the way in. Every empty set is
 * {@link #EMPTY} (dimension 0).
 */
public final class PointSet implements Serializable {

    @Serial
    private static final long serialVersionUID = 1L;

    private static final PointBlock[] NO_SEGMENTS = new PointBlock[0];

    /** The set without points. */
    public static final PointSet EMPTY = new PointSet(NO_SEGMENTS, new int[1], 0, 0L);

    private final PointBlock[] segments; // non-empty, all of dimension dim
    private final int[] starts; // segments.length + 1 entries; starts[s] = first global row of segment s
    private final int dim;
    private final long entries;
    private int hash;

    private PointSet(PointBlock[] segments, int[] starts, int dim, long entries) {
        this.segments = segments;
        this.starts = starts;
        this.dim = dim;
        this.entries = entries;
    }

    /** The block as a one-segment set (no copy); an empty block gives {@link #EMPTY}. */
    public static PointSet of(PointBlock block) {
        if (block.isEmpty()) {
            return EMPTY;
        }
        return new PointSet(new PointBlock[] {block}, new int[] {0, block.size()}, block.dim(), block.storedEntries());
    }

    /**
     * The rows of all parts in order, sharing their segments: O(total segments), no entry or row copied. A single
     * non-empty part is returned as is.
     *
     * @throws IllegalArgumentException if two non-empty parts differ in dimension or the result would hold more than
     *     2^31 - 9 points
     */
    public static PointSet concat(List<PointSet> parts) {
        int count = 0;
        int dim = -1;
        long rows = 0;
        long entries = 0;
        PointSet only = null;
        int nonEmpty = 0;
        for (PointSet p : parts) {
            if (p.segments.length == 0) {
                continue;
            }
            if (dim < 0) {
                dim = p.dim;
            } else if (p.dim != dim) {
                throw new IllegalArgumentException("points have dim " + p.dim + ", expected " + dim);
            }
            count += p.segments.length;
            rows += p.size();
            entries += p.entries;
            only = p;
            nonEmpty++;
        }
        if (nonEmpty <= 1) {
            return only == null ? EMPTY : only;
        }
        if (rows > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("more than 2^31 points: " + rows);
        }
        PointBlock[] segs = new PointBlock[count];
        int[] starts = new int[count + 1];
        int s = 0;
        for (PointSet p : parts) {
            for (int t = 0; t < p.segments.length; t++, s++) {
                segs[s] = p.segments[t];
                starts[s + 1] = starts[s] + segs[s].size();
            }
        }
        return new PointSet(segs, starts, dim, entries);
    }

    public int dim() {
        return dim;
    }

    /** Number of points (rows). */
    public int size() {
        return starts[segments.length];
    }

    public boolean isEmpty() {
        return segments.length == 0;
    }

    /** Total stored entries over all rows. */
    public long storedEntries() {
        return entries;
    }

    public int segmentCount() {
        return segments.length;
    }

    public PointBlock segment(int s) {
        return segments[s];
    }

    /** Global row of the first row of segment {@code s}; {@code start(segmentCount()) == size()}. */
    public int start(int s) {
        return starts[s];
    }

    /** Segment holding global row {@code row}: O(log segments). */
    public int segmentOf(int row) {
        if (row < 0 || row >= size()) {
            throw new IndexOutOfBoundsException("row " + row + " of " + size());
        }
        int lo = 0;
        int hi = segments.length - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (starts[mid] <= row) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /** Weight of global row {@code row}; O(log segments), so hot loops should stream the segments instead. */
    public double weight(int row) {
        int s = segmentOf(row);
        return segments[s].weight(row - starts[s]);
    }

    /** Global row {@code row} as a vector view sharing its segment's arrays. */
    public SparseVector vector(int row) {
        int s = segmentOf(row);
        return segments[s].vector(row - starts[s]);
    }

    public WeightedPoint point(int row) {
        int s = segmentOf(row);
        return segments[s].point(row - starts[s]);
    }

    /** Sum of the weights in row order (the same additions as on the compacted block). */
    public double totalWeight() {
        double t = 0.0;
        for (PointBlock b : segments) {
            for (int i = 0, n = b.size(); i < n; i++) {
                t += b.weight(i);
            }
        }
        return t;
    }

    /**
     * Read-only list view of the points; every {@code get} materialises a {@link WeightedPoint} over a vector view
     * (no entry copy). Hot loops should stream the segments instead.
     */
    public List<WeightedPoint> asList() {
        return new Rows(this);
    }

    /**
     * Global rows {@code [from, to)} sharing the segments' arrays ({@link PointBlock#slice}): O(log segments + covered
     * segments), no row or entry copied.
     *
     * @throws IndexOutOfBoundsException unless {@code 0 <= from <= to <= size()}
     */
    public PointSet slice(int from, int to) {
        Objects.checkFromToIndex(from, to, size());
        if (from == to) {
            return EMPTY;
        }
        if (from == 0 && to == size()) {
            return this;
        }
        int first = segmentOf(from);
        int last = segmentOf(to - 1);
        PointBlock[] segs = new PointBlock[last - first + 1];
        int[] st = new int[segs.length + 1];
        long total = 0;
        for (int s = first; s <= last; s++) {
            int lo = Math.max(from, starts[s]) - starts[s];
            int hi = Math.min(to, starts[s + 1]) - starts[s];
            PointBlock b = segments[s].slice(lo, hi);
            segs[s - first] = b;
            st[s - first + 1] = st[s - first] + b.size();
            total += b.storedEntries();
        }
        return new PointSet(segs, st, dim, total);
    }

    /**
     * The rows as one block: the only segment as is, otherwise one copy of every row ({@link PointBlock#concat}). For
     * consumers that need a single block; the algorithms in this library never call it.
     */
    public PointBlock compact() {
        return switch (segments.length) {
            case 0 -> PointBlock.EMPTY;
            case 1 -> segments[0];
            default -> PointBlock.concat(Arrays.asList(segments));
        };
    }

    /**
     * The global rows {@code rows[0, count)}, in that order, with weights {@code weights[0, count)}, as a new block:
     * one allocation per array, one {@code arraycopy} per row and array.
     *
     * @throws IllegalArgumentException if a weight is negative or not finite
     */
    public PointBlock select(int[] rows, double[] weights, int count) {
        if (count == 0) {
            return PointBlock.EMPTY;
        }
        long total = 0;
        for (int t = 0; t < count; t++) {
            int s = segmentOf(rows[t]);
            total += segments[s].nnz(rows[t] - starts[s]);
        }
        PointBlock.Builder b = PointBlock.builder(dim, count, total);
        for (int t = 0; t < count; t++) {
            int s = segmentOf(rows[t]);
            segments[s].appendRowTo(rows[t] - starts[s], weights[t], b);
        }
        return b.build();
    }

    /**
     * Approximate retained heap size in bytes under the {@link SparseVector#estimatedBytes()} model: this object, its
     * two arrays and every segment (segments shared with other sets are counted in full).
     */
    public long estimatedBytes() {
        long self = SparseVector.OBJECT_HEADER_BYTES + 2L * 4 + 8L + 2L * SparseVector.REFERENCE_BYTES;
        long bytes = SparseVector.align(self)
                + SparseVector.align(SparseVector.ARRAY_HEADER_BYTES
                        + (long) SparseVector.REFERENCE_BYTES * segments.length)
                + SparseVector.align(SparseVector.ARRAY_HEADER_BYTES + 4L * starts.length);
        for (PointBlock b : segments) {
            bytes += b.estimatedBytes();
        }
        return bytes;
    }

    /** Value-based: same dimension and the same rows (entries and weights) in the same order. */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof PointSet p) || dim != p.dim || size() != p.size() || entries != p.entries) {
            return false;
        }
        int s = 0;
        int i = 0;
        int t = 0;
        int j = 0;
        for (int r = 0, n = size(); r < n; r++) {
            PointBlock a = segments[s];
            PointBlock b = p.segments[t];
            if (!a.rowEquals(i, b, j)) {
                return false;
            }
            if (++i == a.size()) {
                s++;
                i = 0;
            }
            if (++j == b.size()) {
                t++;
                j = 0;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        int h = hash;
        if (h == 0) {
            h = dim;
            for (PointBlock b : segments) {
                for (int i = 0, n = b.size(); i < n; i++) {
                    h = 31 * h + b.rowHash(i);
                }
            }
            hash = h;
        }
        return h;
    }

    @Override
    public String toString() {
        return "PointSet[dim=" + dim + ", points=" + size() + ", entries=" + entries + ", segments=" + segments.length
                + "]";
    }

    @Serial
    private Object writeReplace() {
        return new SerializedForm(segments.clone());
    }

    @Serial
    private void readObject(ObjectInputStream in) throws InvalidObjectException {
        throw new InvalidObjectException("PointSet is deserialized only through its serialized form");
    }

    private record SerializedForm(PointBlock[] segments) implements Serializable {

        @Serial
        private Object readResolve() throws InvalidObjectException {
            try {
                List<PointSet> parts = new ArrayList<>(segments.length);
                for (PointBlock b : segments) {
                    parts.add(of(b));
                }
                return concat(parts);
            } catch (IllegalArgumentException | NullPointerException e) {
                InvalidObjectException ex = new InvalidObjectException("invalid PointSet: " + e.getMessage());
                ex.initCause(e);
                throw ex;
            }
        }
    }

    /** Read-only list over the rows; serialises as a plain list of materialised points. */
    private static final class Rows extends AbstractList<WeightedPoint> implements RandomAccess, Serializable {
        private final transient PointSet set;

        Rows(PointSet set) {
            this.set = set;
        }

        @Override
        public WeightedPoint get(int index) {
            return set.point(index);
        }

        @Override
        public int size() {
            return set.size();
        }

        @Serial
        private Object writeReplace() {
            return new ArrayList<>(this);
        }
    }
}
