package io.github.c0rwin.coreset.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Weighted point set summarising a larger input for k-means: for any centers {@code C},
 * {@code cost(input, C) ~ sum_p w_p * min_c ||p - c||^2 + offset}. The points are stored as one {@link PointBlock}
 * (compressed sparse rows), which the algorithms read directly. Java deserialization re-validates the block and
 * re-runs the canonical constructor's checks.
 *
 * @param block  the weighted points; all share one dimension
 * @param offset finite, non-negative additive cost constant that does not depend on the centers
 */
public record Coreset(PointBlock block, double offset) implements Serializable {

    public Coreset {
        Objects.requireNonNull(block, "block");
        if (!Double.isFinite(offset) || offset < 0.0) {
            throw new IllegalArgumentException("offset must be finite and >= 0: " + offset);
        }
    }

    /**
     * Copies {@code points} into a new block ({@link PointBlock#of}).
     *
     * @throws IllegalArgumentException if the points differ in dimension or the offset is invalid
     */
    public Coreset(List<WeightedPoint> points, double offset) {
        this(PointBlock.of(points), offset);
    }

    public static Coreset of(List<WeightedPoint> points) {
        return new Coreset(points, 0.0);
    }

    public static Coreset of(PointBlock block) {
        return new Coreset(block, 0.0);
    }

    /** Concatenates points (a's first) and adds offsets; one allocation per block array. */
    public static Coreset union(Coreset a, Coreset b) {
        return union(List.of(a, b));
    }

    /**
     * Concatenates the points of all coresets in order and adds their offsets: one allocation per block array, and
     * no per-point checks beyond one dimension comparison per coreset.
     *
     * @throws IllegalArgumentException if two non-empty coresets differ in dimension
     */
    public static Coreset union(List<Coreset> parts) {
        List<PointBlock> blocks = new ArrayList<>(parts.size());
        double offset = 0.0;
        for (Coreset c : parts) {
            blocks.add(c.block);
            offset += c.offset;
        }
        return new Coreset(PointBlock.concat(blocks), offset);
    }

    /** Builder that collects points one by one; the expected size only presizes it. */
    public static Builder builder(int expectedSize) {
        if (expectedSize < 0) {
            throw new IllegalArgumentException("expectedSize must be >= 0: " + expectedSize);
        }
        return new Builder(expectedSize);
    }

    /**
     * The points as a read-only list view of {@link #block()}: every {@code get} materialises a {@link WeightedPoint}
     * over a vector view (two small objects, no entry copy). Hot loops should read the block instead.
     */
    public List<WeightedPoint> points() {
        return block.asList();
    }

    public double totalWeight() {
        return block.totalWeight();
    }

    public int size() {
        return block.size();
    }

    public boolean isEmpty() {
        return block.isEmpty();
    }

    /**
     * Approximate retained heap size in bytes (same model as {@link SparseVector#estimatedBytes()}): this record and
     * its {@link PointBlock#estimatedBytes() block}.
     */
    public long estimatedBytes() {
        return SparseVector.align(SparseVector.OBJECT_HEADER_BYTES + SparseVector.REFERENCE_BYTES + 8L)
                + block.estimatedBytes();
    }

    /** Collects points, checking each one's dimension as it is added. Not thread-safe; build at most once. */
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

        /** Appends every point of {@code c}: one dimension check and one bulk copy. */
        public Builder addAll(Coreset c) {
            if (!c.isEmpty()) {
                target(c.block.dim(), c.block.storedEntries()).addAll(c.block);
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

        private PointBlock.Builder target(int dim, long entries) {
            if (built) {
                throw new IllegalStateException("already built");
            }
            if (points == null) {
                points = PointBlock.builder(dim, expectedSize, entries * Math.max(1, expectedSize));
            }
            return points;
        }
    }
}
