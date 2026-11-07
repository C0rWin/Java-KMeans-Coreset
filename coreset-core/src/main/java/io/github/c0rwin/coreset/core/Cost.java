package io.github.c0rwin.coreset.core;

import java.util.List;

/** Weighted k-means cost using exact (merge-based) squared distances. */
public final class Cost {

    private Cost() {
    }

    /** {@code sum_p w_p * min_c ||p - c||^2 + offset}. */
    public static double cost(Coreset c, List<SparseVector> centers) {
        return cost(c.block(), centers) + c.offset();
    }

    /**
     * {@code sum_p w_p * min_c ||p - c||^2} over the rows of {@code points}, streamed from the block arrays;
     * bit-identical to the list form on the same points. Zero-weight points are skipped.
     *
     * @throws IllegalArgumentException if {@code points} is non-empty and {@code centers} is empty
     */
    public static double cost(PointBlock points, List<SparseVector> centers) {
        if (points.isEmpty()) {
            return 0.0;
        }
        requireCenters(centers);
        SparseVector[] cs = centers.toArray(new SparseVector[0]);
        double s = 0.0;
        for (int i = 0, n = points.size(); i < n; i++) {
            double w = points.weight(i);
            if (w > 0.0) {
                double best = points.squaredDistance(i, cs[0]);
                for (int j = 1; j < cs.length; j++) {
                    double d = points.squaredDistance(i, cs[j]);
                    if (d < best) {
                        best = d;
                    }
                }
                s += w * best;
            }
        }
        return s;
    }

    /**
     * {@code sum_p w_p * min_c ||p - c||^2}; zero-weight points are skipped.
     *
     * @throws IllegalArgumentException if {@code points} is non-empty and {@code centers} is empty
     */
    public static double cost(List<WeightedPoint> points, List<SparseVector> centers) {
        if (points.isEmpty()) {
            return 0.0;
        }
        requireCenters(centers);
        double s = 0.0;
        for (int i = 0, n = points.size(); i < n; i++) {
            WeightedPoint p = points.get(i);
            double w = p.weight();
            if (w > 0.0) {
                s += w * minSquaredDistance(p.vector(), centers);
            }
        }
        return s;
    }

    /** Index of the closest center; ties go to the lowest index. */
    public static int nearest(SparseVector p, List<SparseVector> centers) {
        requireCenters(centers);
        int best = 0;
        double bestD = p.squaredDistance(centers.get(0));
        for (int j = 1, k = centers.size(); j < k; j++) {
            double d = p.squaredDistance(centers.get(j));
            if (d < bestD) {
                bestD = d;
                best = j;
            }
        }
        return best;
    }

    private static double minSquaredDistance(SparseVector p, List<SparseVector> centers) {
        double best = p.squaredDistance(centers.get(0));
        for (int j = 1, k = centers.size(); j < k; j++) {
            double d = p.squaredDistance(centers.get(j));
            if (d < best) {
                best = d;
            }
        }
        return best;
    }

    private static void requireCenters(List<SparseVector> centers) {
        if (centers.isEmpty()) {
            throw new IllegalArgumentException("centers must not be empty");
        }
    }
}
