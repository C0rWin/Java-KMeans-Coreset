package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;

/**
 * One shard of a distributed cost evaluation.
 *
 * @param cost   {@code sum_p w_p * min_c ||p - c||^2}, plus the coreset offset when evaluated on a session
 * @param weight total weight of the points
 * @param points number of points
 */
public record PartialCost(double cost, double weight, long points) {

    /** Points between two polls of the cancellation flag. */
    static final int CANCEL_CHECK_EVERY = 4096;

    /**
     * One pass over {@code points}, no allocation: the weight sum is fused with the nearest-center scan, which matches
     * {@code Cost.cost} term for term (zero-weight points skipped, same summation order).
     *
     * @throws CancellationException once {@code cancelled} returns true (polled every {@value #CANCEL_CHECK_EVERY})
     */
    static PartialCost of(List<WeightedPoint> points, List<SparseVector> centers, double offset,
                          BooleanSupplier cancelled) {
        int n = points.size();
        int k = centers.size();
        if (n > 0 && k == 0) {
            throw new IllegalArgumentException("centers must not be empty");
        }
        double cost = 0.0;
        double weight = 0.0;
        for (int i = 0; i < n; i++) {
            if ((i & (CANCEL_CHECK_EVERY - 1)) == 0 && cancelled.getAsBoolean()) {
                throw new CancellationException("cost evaluation cancelled after " + i + " of " + n + " points");
            }
            WeightedPoint p = points.get(i);
            double w = p.weight();
            weight += w;
            if (w > 0.0) {
                SparseVector v = p.vector();
                double best = v.squaredDistance(centers.get(0));
                for (int j = 1; j < k; j++) {
                    double d = v.squaredDistance(centers.get(j));
                    if (d < best) {
                        best = d;
                    }
                }
                cost += w * best;
            }
        }
        return new PartialCost(cost + offset, weight, n);
    }
}
