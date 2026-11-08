package io.github.c0rwin.coreset.core;

import java.io.Serializable;
import java.util.Objects;

/**
 * A point with a non-negative, finite multiplicity used by weighted k-means and coresets. Serializable so engines can
 * ship it with their default serializer; deserialization re-runs the canonical constructor's checks.
 */
public record WeightedPoint(SparseVector vector, double weight) implements Serializable {

    public WeightedPoint {
        Objects.requireNonNull(vector, "vector");
        if (!Double.isFinite(weight) || weight < 0.0) {
            throw new IllegalArgumentException("weight must be finite and >= 0: " + weight);
        }
    }

    /** Approximate retained heap size in bytes, including the vector; see {@link SparseVector#estimatedBytes()}. */
    public long estimatedBytes() {
        return SparseVector.align(SparseVector.OBJECT_HEADER_BYTES + SparseVector.REFERENCE_BYTES + 8L)
                + vector.estimatedBytes();
    }
}
