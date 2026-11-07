package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.SparseVector;
import java.util.Arrays;

/**
 * Weighted means of the clusters of a partition of sparse points, without any O(dim) buffer.
 *
 * <p>{@link #group} buckets the positive-weight points by cluster (counting sort, one pass). {@link #mean} then gathers
 * one cluster's stored entries, sorts them by (coordinate, point order) and sums each coordinate in point order, which
 * are exactly the additions a dense accumulator performs, so the means are bit-identical to the dense computation.
 * Memory is O(n + clusters + entries of the largest cluster); time per {@link #mean} is O(e log e) for its e entries.
 */
final class ClusterSums {

    private final SparseVector[] vectors;
    private final double[] weights;
    private final int[] members;
    private int[] start = new int[0];
    private int[] next = new int[0];
    private long[] entries = new long[0];
    private double[] clusterWeight = new double[0];
    private long[] keys = new long[0]; // (coordinate << 32) | entry, sorted per cluster
    private double[] values = new double[0];
    private double[] entryWeights = new double[0];

    ClusterSums(SparseVector[] vectors, double[] weights) {
        this.vectors = vectors;
        this.weights = weights;
        this.members = new int[vectors.length];
    }

    /** Number of points. */
    int size() {
        return vectors.length;
    }

    /**
     * Buckets the positive-weight points by {@code assignment} into {@code clusters} groups and sums their weights in
     * point order; returns the per-cluster weights (owned by this object, overwritten by the next call).
     */
    double[] group(int[] assignment, int clusters) {
        if (start.length != clusters + 1) {
            start = new int[clusters + 1];
            next = new int[clusters];
            entries = new long[clusters];
            clusterWeight = new double[clusters];
        } else {
            Arrays.fill(start, 0);
            Arrays.fill(entries, 0L);
            Arrays.fill(clusterWeight, 0.0);
        }
        for (int i = 0; i < vectors.length; i++) {
            double w = weights[i];
            if (w > 0.0) {
                int j = assignment[i];
                start[j + 1]++;
                clusterWeight[j] += w;
                entries[j] += vectors[i].nnz();
            }
        }
        long maxEntries = 0;
        for (int j = 0; j < clusters; j++) {
            start[j + 1] += start[j];
            next[j] = start[j];
            maxEntries = Math.max(maxEntries, entries[j]);
        }
        for (int i = 0; i < vectors.length; i++) {
            if (weights[i] > 0.0) {
                members[next[assignment[i]]++] = i;
            }
        }
        if (maxEntries > keys.length) {
            if (maxEntries > Integer.MAX_VALUE - 8) {
                throw new IllegalArgumentException("a cluster holds more than 2^31 stored entries: " + maxEntries);
            }
            keys = new long[(int) maxEntries];
            values = new double[(int) maxEntries];
            entryWeights = new double[(int) maxEntries];
        }
        return clusterWeight;
    }

    /**
     * Weighted mean of cluster {@code j} (which must have positive weight) after {@link #group}; with {@code withCost}
     * also {@code sum_p w_p ||p - mean||^2} over its members, computed as a sum of non-negative terms per coordinate:
     * {@code sum_{p storing c} w_p (p_c - mean_c)^2 + (W - sum_{p storing c} w_p) mean_c^2}, so it does not cancel.
     */
    Mean mean(int j, boolean withCost) {
        int e = 0;
        for (int t = start[j], end = start[j + 1]; t < end; t++) {
            int i = members[t];
            SparseVector x = vectors[i];
            double w = weights[i];
            for (int k = 0, nnz = x.nnz(); k < nnz; k++, e++) {
                keys[e] = ((long) x.indexAt(k) << 32) | e;
                values[e] = x.valueAt(k);
                entryWeights[e] = w;
            }
        }
        Arrays.sort(keys, 0, e);
        int distinct = 0;
        for (int r = 0; r < e; r++) {
            if (r == 0 || (keys[r] >>> 32) != (keys[r - 1] >>> 32)) {
                distinct++;
            }
        }
        final double total = clusterWeight[j];
        final double inv = 1.0 / total;
        int[] idx = new int[distinct];
        double[] val = new double[distinct];
        double norm = 0.0;
        double cost = 0.0;
        for (int r = 0, d = 0; r < e; d++) {
            final long coordinate = keys[r] >>> 32;
            int end = r;
            double sum = 0.0;
            while (end < e && (keys[end] >>> 32) == coordinate) {
                int p = (int) keys[end];
                sum += entryWeights[p] * values[p];
                end++;
            }
            double mu = sum * inv;
            idx[d] = (int) coordinate;
            val[d] = mu;
            norm += mu * mu;
            if (withCost) {
                double covered = 0.0;
                for (int q = r; q < end; q++) {
                    int p = (int) keys[q];
                    double diff = values[p] - mu;
                    cost += entryWeights[p] * diff * diff;
                    covered += entryWeights[p];
                }
                cost += Math.max(0.0, total - covered) * mu * mu;
            }
            r = end;
        }
        return new Mean(idx, val, norm, cost);
    }

    /** Sorted coordinates and values of a mean (zeros possible after cancellation), its squared norm and cost. */
    static final class Mean {
        final int[] indices;
        final double[] values;
        final double squaredNorm;
        final double cost;

        Mean(int[] indices, double[] values, double squaredNorm, double cost) {
            this.indices = indices;
            this.values = values;
            this.squaredNorm = squaredNorm;
            this.cost = cost;
        }
    }
}
