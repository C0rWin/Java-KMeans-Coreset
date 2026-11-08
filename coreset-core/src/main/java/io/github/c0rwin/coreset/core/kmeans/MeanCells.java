package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.PointBlock;
import java.util.Objects;

/**
 * A mean coreset with its offset kept per cell (the clustering-feature view of BIRCH): row {@code g} of {@code means}
 * is the weighted mean of cell {@code g} with the cell's weight, and {@code offsets[g]} is the cell's
 * {@code sum_p w_p ||p - mean||^2}. For any center {@code x}, {@code sum_{p in cell} w_p ||p - x||^2 =
 * w_g ||mean_g - x||^2 + offsets[g]}, so any subset of cells is the exact mean coreset of the points it covers.
 *
 * @param means   one row per cell
 * @param offsets per-cell offsets; the array is shared, not copied
 */
public record MeanCells(PointBlock means, double[] offsets) {

    public MeanCells {
        Objects.requireNonNull(means, "means");
        Objects.requireNonNull(offsets, "offsets");
        if (offsets.length != means.size()) {
            throw new IllegalArgumentException(offsets.length + " offsets for " + means.size() + " cells");
        }
    }
}
