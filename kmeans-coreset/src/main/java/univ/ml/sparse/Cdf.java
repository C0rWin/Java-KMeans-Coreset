package univ.ml.sparse;

import java.util.List;

/**
 * Cumulative distribution over the (possibly unnormalized) sampling probabilities of a dataset.
 */
final class Cdf {

    private Cdf() {
    }

    /** Running sums of the point probabilities; the last entry is their total. */
    static double[] of(final List<SparseWeightableVector> dataset) {
        final double[] cdf = new double[dataset.size()];
        double sum = 0;
        for (int i = 0; i < cdf.length; i++) {
            sum += dataset.get(i).getProbability();
            cdf[i] = sum;
        }
        return cdf;
    }

    /**
     * Index of the point whose interval [cdf[i-1], cdf[i]) contains {@code u * total}, for u uniform in [0, 1).
     * That is the first index with cdf[i] > u * total, which is never a zero-probability point.
     */
    static int draw(final double[] cdf, final double u) {
        final double r = u * cdf[cdf.length - 1];
        // hi starts at the last index, so a degenerate total (zero or NaN) still yields a valid index.
        int lo = 0;
        int hi = cdf.length - 1;
        while (lo < hi) {
            final int mid = (lo + hi) >>> 1;
            if (cdf[mid] > r) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }
}
