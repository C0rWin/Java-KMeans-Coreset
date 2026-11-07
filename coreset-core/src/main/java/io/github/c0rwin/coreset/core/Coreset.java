package io.github.c0rwin.coreset.core;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

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
        points = List.copyOf(points);
        if (!Double.isFinite(offset) || offset < 0.0) {
            throw new IllegalArgumentException("offset must be finite and >= 0: " + offset);
        }
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

    public static Coreset of(List<WeightedPoint> points) {
        return new Coreset(points, 0.0);
    }

    /** Concatenates points (a's first) and adds offsets. */
    public static Coreset union(Coreset a, Coreset b) {
        List<WeightedPoint> all = new ArrayList<>(a.points.size() + b.points.size());
        all.addAll(a.points);
        all.addAll(b.points);
        return new Coreset(all, a.offset + b.offset);
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
}
