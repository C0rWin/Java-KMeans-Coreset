package io.github.c0rwin.coreset.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Weighted point set summarising a larger input for k-means: for any centers {@code C},
 * {@code cost(input, C) ~ sum_p w_p * min_c ||p - c||^2 + offset}. The points are a {@link PointSet}, a sequence of
 * {@link PointBlock} segments that the algorithms stream directly; {@link #union} shares the parts' segments instead of
 * copying their points. Java deserialization re-validates the segments and re-runs the canonical constructor's checks.
 *
 * @param pointSet the weighted points; all share one dimension
 * @param offset   finite, non-negative additive cost constant that does not depend on the centers
 */
public record Coreset(PointSet pointSet, double offset) implements Serializable {

    public Coreset {
        Objects.requireNonNull(pointSet, "pointSet");
        if (!Double.isFinite(offset) || offset < 0.0) {
            throw new IllegalArgumentException("offset must be finite and >= 0: " + offset);
        }
    }

    /** The block as the only segment (no copy). */
    public Coreset(PointBlock block, double offset) {
        this(PointSet.of(block), offset);
    }

    /**
     * Copies every point's entries into one new block ({@link PointBlock#of}): O(total entries). Code on a hot path
     * should build a {@link PointBlock} instead (its {@link PointBlock.Builder}, or {@link PointBlock#wrap} in a
     * decoder).
     *
     * @throws IllegalArgumentException if the points differ in dimension or the offset is invalid
     */
    public Coreset(List<WeightedPoint> points, double offset) {
        this(PointBlock.of(points), offset);
    }

    /** {@link #Coreset(List, double)} with offset 0: copies the entries. */
    public static Coreset of(List<WeightedPoint> points) {
        return new Coreset(points, 0.0);
    }

    public static Coreset of(PointBlock block) {
        return new Coreset(block, 0.0);
    }

    public static Coreset of(PointSet points) {
        return new Coreset(points, 0.0);
    }

    /** Points of {@code a} then {@code b}, sharing their segments, with the offsets added: O(segments). */
    public static Coreset union(Coreset a, Coreset b) {
        return new Coreset(PointSet.concat(List.of(a.pointSet, b.pointSet)), a.offset + b.offset);
    }

    /**
     * The points of all coresets in order, sharing their segments, with the offsets added: O(parts + segments), no
     * point or entry copied and no per-point check beyond one dimension comparison per part.
     *
     * @throws IllegalArgumentException if two non-empty coresets differ in dimension
     */
    public static Coreset union(List<Coreset> parts) {
        List<PointSet> sets = new ArrayList<>(parts.size());
        double offset = 0.0;
        for (Coreset c : parts) {
            sets.add(c.pointSet);
            offset += c.offset;
        }
        return new Coreset(PointSet.concat(sets), offset);
    }

    /** Builder that collects points one by one; the expected size only presizes it. */
    public static Builder builder(int expectedSize) {
        if (expectedSize < 0) {
            throw new IllegalArgumentException("expectedSize must be >= 0: " + expectedSize);
        }
        return new Builder(expectedSize);
    }

    /**
     * The points as a read-only list view of {@link #pointSet()}: every {@code get} materialises a
     * {@link WeightedPoint} over a vector view (two small objects, no entry copy). Hot loops should stream the
     * segments instead.
     */
    public List<WeightedPoint> points() {
        return pointSet.asList();
    }

    public double totalWeight() {
        return pointSet.totalWeight();
    }

    public int size() {
        return pointSet.size();
    }

    public boolean isEmpty() {
        return pointSet.isEmpty();
    }

    /**
     * Approximate retained heap size in bytes (same model as {@link SparseVector#estimatedBytes()}): this record and
     * its {@link PointSet#estimatedBytes() point set}.
     */
    public long estimatedBytes() {
        return SparseVector.align(SparseVector.OBJECT_HEADER_BYTES + SparseVector.REFERENCE_BYTES + 8L)
                + pointSet.estimatedBytes();
    }

    /** Collects points into one block, checking each one's dimension as it is added. Not thread-safe; build once. */
    public static final class Builder {
        private final int expectedSize;
        private PointBlock.Builder points; // created by the first point, which fixes the dimension
        private boolean built;

        private Builder(int expectedSize) {
            this.expectedSize = expectedSize;
        }

        /** @throws IllegalArgumentException if {@code p}'s dimension differs from the points added before */
        public Builder add(WeightedPoint p) {
            return add(p.vector(), p.weight());
        }

        /** @throws IllegalArgumentException if the dimension differs or the weight is negative or not finite */
        public Builder add(SparseVector v, double weight) {
            target(v.dim(), v.nnz()).add(v, weight);
            return this;
        }

        /** Appends every point of {@code c}: one dimension check and one bulk copy per segment. */
        public Builder addAll(Coreset c) {
            PointSet s = c.pointSet;
            if (!s.isEmpty()) {
                PointBlock.Builder b = target(s.dim(), (s.storedEntries() + s.size() - 1) / s.size());
                for (int t = 0; t < s.segmentCount(); t++) {
                    b.addAll(s.segment(t));
                }
            }
            return this;
        }

        public Coreset build(double offset) {
            if (built) {
                throw new IllegalStateException("already built");
            }
            built = true;
            return new Coreset(points == null ? PointBlock.EMPTY : points.build(), offset);
        }

        /** The first point fixes the dimension; its entries per point times the expected size presize the arrays. */
        private PointBlock.Builder target(int dim, long entriesPerPoint) {
            if (built) {
                throw new IllegalStateException("already built");
            }
            if (points == null) {
                points = PointBlock.builder(dim, expectedSize, entriesPerPoint * Math.max(1, expectedSize));
            }
            return points;
        }
    }
}
