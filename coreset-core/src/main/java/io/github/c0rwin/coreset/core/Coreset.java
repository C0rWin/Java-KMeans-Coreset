package io.github.c0rwin.coreset.core;

import java.io.Serial;
import java.io.Serializable;
import java.util.AbstractList;
import java.util.Arrays;
import java.util.List;
import java.util.RandomAccess;

/**
 * Weighted point set summarising a larger input for k-means: for any centers {@code C},
 * {@code cost(input, C) ~ sum_p w_p * min_c ||p - c||^2 + offset}. Java deserialization re-runs the canonical
 * constructor's copy and checks.
 *
 * @param points immutable copy of the weighted points; all vectors share one dimension
 * @param offset finite, non-negative additive cost constant that does not depend on the centers
 */
public record Coreset(List<WeightedPoint> points, double offset) implements Serializable {

    public Coreset {
        if (!(points instanceof PointArray)) { // built by Builder, which already checked every point
            points = List.copyOf(points);
            checkDimensions(points);
        }
        if (!Double.isFinite(offset) || offset < 0.0) {
            throw new IllegalArgumentException("offset must be finite and >= 0: " + offset);
        }
    }

    public static Coreset of(List<WeightedPoint> points) {
        return new Coreset(points, 0.0);
    }

    /** Concatenates points (a's first) and adds offsets; one allocation for the result's points. */
    public static Coreset union(Coreset a, Coreset b) {
        return union(List.of(a, b));
    }

    /**
     * Concatenates the points of all coresets in order and adds their offsets; one allocation for the result's
     * points, and no per-point checks beyond one dimension comparison per coreset.
     *
     * @throws IllegalArgumentException if two non-empty coresets differ in dimension
     */
    public static Coreset union(List<Coreset> parts) {
        int total = 0;
        double offset = 0.0;
        for (Coreset c : parts) {
            total = Math.addExact(total, c.size());
            offset += c.offset;
        }
        Builder b = new Builder(total);
        for (Coreset c : parts) {
            b.addAll(c);
        }
        return b.build(offset);
    }

    /** Builder that collects points into one exactly sized array (when the expected size is right). */
    public static Builder builder(int expectedSize) {
        return new Builder(expectedSize);
    }

    public double totalWeight() {
        double s = 0.0;
        for (WeightedPoint p : points) {
            s += p.weight();
        }
        return s;
    }

    public int size() {
        return points.size();
    }

    public boolean isEmpty() {
        return points.isEmpty();
    }

    /**
     * Approximate retained heap size in bytes (same model as {@link SparseVector#estimatedBytes()}): this record, the
     * immutable list and its reference array, and every point. Vectors shared between points are counted per point.
     */
    public long estimatedBytes() {
        long self = SparseVector.align(SparseVector.OBJECT_HEADER_BYTES + SparseVector.REFERENCE_BYTES + 8L);
        long list = SparseVector.align(SparseVector.OBJECT_HEADER_BYTES + SparseVector.REFERENCE_BYTES)
                + SparseVector.align(SparseVector.ARRAY_HEADER_BYTES + (long) SparseVector.REFERENCE_BYTES * points.size());
        long total = self + list;
        for (WeightedPoint p : points) {
            total += p.estimatedBytes();
        }
        return total;
    }

    private static void checkDimensions(List<WeightedPoint> points) {
        if (!points.isEmpty()) {
            int dim = points.get(0).vector().dim();
            for (int i = 1; i < points.size(); i++) {
                if (points.get(i).vector().dim() != dim) {
                    throw new IllegalArgumentException(
                            "point " + i + " has dim " + points.get(i).vector().dim() + ", expected " + dim);
                }
            }
        }
    }

    /** Collects points, checking each one's dimension as it is added. Not thread-safe; build at most once. */
    public static final class Builder {
        private WeightedPoint[] points;
        private int size;
        private int dim = -1;

        private Builder(int expectedSize) {
            if (expectedSize < 0) {
                throw new IllegalArgumentException("expectedSize must be >= 0: " + expectedSize);
            }
            points = new WeightedPoint[expectedSize];
        }

        /** @throws IllegalArgumentException if {@code p}'s dimension differs from the points added before */
        public Builder add(WeightedPoint p) {
            int d = p.vector().dim();
            if (dim < 0) {
                dim = d;
            } else if (d != dim) {
                throw new IllegalArgumentException("point " + size + " has dim " + d + ", expected " + dim);
            }
            ensure(1);
            points[size++] = p;
            return this;
        }

        public Builder add(SparseVector v, double weight) {
            return add(new WeightedPoint(v, weight));
        }

        /** Appends every point of {@code c}, which is consistent already: one dimension check in total. */
        public Builder addAll(Coreset c) {
            List<WeightedPoint> src = c.points;
            int n = src.size();
            if (n == 0) {
                return this;
            }
            int d = src.get(0).vector().dim();
            if (dim < 0) {
                dim = d;
            } else if (d != dim) {
                throw new IllegalArgumentException("coreset has dim " + d + ", expected " + dim);
            }
            ensure(n);
            if (src instanceof PointArray a) {
                System.arraycopy(a.points, 0, points, size, n);
            } else {
                for (int i = 0; i < n; i++) {
                    points[size + i] = src.get(i);
                }
            }
            size += n;
            return this;
        }

        public Coreset build(double offset) {
            WeightedPoint[] out = size == points.length ? points : Arrays.copyOf(points, size);
            points = null;
            return new Coreset(new PointArray(out), offset);
        }

        private void ensure(int extra) {
            if (points == null) {
                throw new IllegalStateException("already built");
            }
            if (points.length - size < extra) {
                points = Arrays.copyOf(points, Math.max(Math.addExact(size, extra), 2 * points.length));
            }
        }
    }

    /** Immutable list over an array no one else references. Serialised as a plain list, so it is re-checked. */
    private static final class PointArray extends AbstractList<WeightedPoint> implements RandomAccess, Serializable {
        private final WeightedPoint[] points;

        PointArray(WeightedPoint[] points) {
            this.points = points;
        }

        @Override
        public WeightedPoint get(int index) {
            return points[index];
        }

        @Override
        public int size() {
            return points.length;
        }

        @Serial
        private Object writeReplace() {
            return Arrays.asList(points.clone());
        }
    }
}
