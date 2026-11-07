package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.SparseVector;
import java.util.Arrays;
import java.util.List;

/**
 * The centers of one Lloyd run. {@link Dense} keeps m x dim arrays; {@link Sparse} keeps every center as its sorted
 * stored coordinates plus an open-addressing index, so memory and update time follow the clusters' stored entries
 * instead of m * dim. Both perform the same floating-point operations in the same order and return bit-identical
 * results; {@link #layoutFor} picks one. Not thread-safe: one instance per run.
 */
abstract class Centroids {

    /** Dense arrays are always allowed up to this many center cells (8 MiB per copy, two copies). */
    static final long DENSE_CELLS = 1L << 20;

    enum Layout { DENSE, SPARSE }

    final int dim;
    final double[] norms;
    private final int[] nearest;
    /** Weighted cost of the last {@link #assign}. */
    double cost;

    Centroids(int m, int dim, int n) {
        this.dim = dim;
        this.norms = new double[m];
        this.nearest = new int[n];
    }

    /**
     * Dense when {@code m * dim <= max(DENSE_CELLS, 2 * totalNnz)}: the two dense copies then stay within a small
     * multiple of the input size and their O(m * dim) sweep per update costs no more than a pass over the input.
     */
    static Layout layoutFor(int m, int dim, long totalNnz) {
        return (long) m * dim <= Math.max(DENSE_CELLS, 2 * totalNnz) ? Layout.DENSE : Layout.SPARSE;
    }

    static Centroids create(Layout layout, List<SparseVector> initial, int dim, SparseVector[] vectors,
                            double[] weights) {
        Centroids c = layout == Layout.DENSE
                ? new Dense(initial.size(), dim, vectors.length)
                : new Sparse(initial.size(), dim, new ClusterSums(vectors, weights));
        for (int j = 0; j < initial.size(); j++) {
            c.set(j, initial.get(j));
        }
        return c;
    }

    /**
     * Assigns every point to its nearest center (ties: lowest index), stores the squared distances in {@code dist} and
     * the weighted cost in {@link #cost}; returns whether any assignment changed.
     */
    final boolean assign(SparseVector[] vectors, double[] weights, int[] assignment, double[] dist) {
        findNearest(vectors, nearest, dist);
        boolean changed = false;
        double total = 0.0;
        for (int i = 0; i < vectors.length; i++) {
            if (assignment[i] != nearest[i]) {
                assignment[i] = nearest[i];
                changed = true;
            }
            if (weights[i] > 0.0) { // a zero weight times an overflowed distance would be NaN
                total += weights[i] * dist[i];
            }
        }
        cost = total;
        return changed;
    }

    /** For every point, the index of its nearest center (ties: lowest index) and the squared distance to it. */
    abstract void findNearest(SparseVector[] vectors, int[] best, double[] dist);

    /**
     * Moves every center whose cluster has positive weight to the cluster's weighted mean and leaves the others in
     * place; fills {@code clusterWeight}, summed in point order. One pass over the points (two for {@link Sparse}).
     */
    abstract void moveToMeans(SparseVector[] vectors, double[] weights, int[] assignment, double[] clusterWeight);

    /** Places center {@code j} at {@code v}. */
    abstract void set(int j, SparseVector v);

    abstract SparseVector center(int j);

    /** m x dim arrays; the spare copy receives the next update's sums. */
    static final class Dense extends Centroids {
        private final double[][] centers;
        private final double[][] spare;

        Dense(int m, int dim, int n) {
            super(m, dim, n);
            centers = new double[m][dim];
            spare = new double[m][dim];
        }

        /** Point-major: a point is read once while all centers, small by {@link #layoutFor}, stay cached. */
        @Override
        void findNearest(SparseVector[] vectors, int[] best, double[] dist) {
            for (int i = 0; i < vectors.length; i++) {
                SparseVector x = vectors[i];
                int b = 0;
                double bestD = x.squaredDistance(centers[0], norms[0]);
                for (int j = 1; j < centers.length; j++) {
                    double d = x.squaredDistance(centers[j], norms[j]);
                    if (d < bestD) {
                        bestD = d;
                        b = j;
                    }
                }
                best[i] = b;
                dist[i] = bestD;
            }
        }

        @Override
        void moveToMeans(SparseVector[] vectors, double[] weights, int[] assignment, double[] clusterWeight) {
            for (double[] row : spare) {
                Arrays.fill(row, 0.0);
            }
            Arrays.fill(clusterWeight, 0.0);
            for (int i = 0; i < vectors.length; i++) {
                double w = weights[i];
                if (w > 0.0) {
                    int j = assignment[i];
                    vectors[i].addTo(spare[j], w);
                    clusterWeight[j] += w;
                }
            }
            for (int j = 0; j < centers.length; j++) {
                if (clusterWeight[j] > 0.0) {
                    double[] acc = spare[j];
                    double inv = 1.0 / clusterWeight[j];
                    double norm = 0.0;
                    for (int d = 0; d < acc.length; d++) {
                        double v = acc[d] * inv;
                        acc[d] = v;
                        norm += v * v;
                    }
                    spare[j] = centers[j];
                    centers[j] = acc;
                    norms[j] = norm;
                }
            }
        }

        @Override
        void set(int j, SparseVector v) {
            double[] row = centers[j];
            Arrays.fill(row, 0.0);
            v.addTo(row, 1.0);
            norms[j] = v.squaredNorm();
        }

        @Override
        SparseVector center(int j) {
            return SparseVector.dense(centers[j]);
        }
    }

    /**
     * Per center: sorted stored coordinates and values, and an open-addressing table (load at most 1/2) from
     * coordinate to value for O(1) lookups during assignment. A table slot keeps key and value next to each other, so a
     * lookup usually touches one cache line. About 44-76 bytes per stored center coordinate.
     */
    static final class Sparse extends Centroids {
        private final ClusterSums sums;
        private final int[][] indices;
        private final double[][] values;
        private final double[][] tables; // slot s: [2s] = coordinate + 1 (0 = empty), [2s + 1] = value

        Sparse(int m, int dim, ClusterSums sums) {
            super(m, dim, sums.size());
            this.sums = sums;
            indices = new int[m][];
            values = new double[m][];
            tables = new double[m][];
        }

        /**
         * Center-major: each center's table stays cached while the points stream past it, instead of every lookup
         * missing into a different table. Visiting centers in index order with a strict comparison keeps the lowest
         * index on ties, as the point-major order does.
         */
        @Override
        void findNearest(SparseVector[] vectors, int[] best, double[] dist) {
            for (int i = 0; i < vectors.length; i++) {
                dist[i] = squaredDistance(0, vectors[i]);
                best[i] = 0;
            }
            for (int j = 1; j < norms.length; j++) {
                for (int i = 0; i < vectors.length; i++) {
                    double d = squaredDistance(j, vectors[i]);
                    if (d < dist[i]) {
                        dist[i] = d;
                        best[i] = j;
                    }
                }
            }
        }

        /** Same arithmetic as {@link SparseVector#squaredDistance(double[], double)}, with table lookups. */
        private double squaredDistance(int j, SparseVector x) {
            final double[] table = tables[j];
            final int mask = (table.length >> 1) - 1;
            double onSupport = 0.0;
            double centerOnSupport = 0.0;
            for (int k = 0, nnz = x.nnz(); k < nnz; k++) {
                int index = x.indexAt(k);
                double key = index + 1.0;
                double c = 0.0;
                for (int s = slot(index, mask); ; s = (s + 1) & mask) {
                    double stored = table[2 * s];
                    if (stored == key) {
                        c = table[2 * s + 1];
                        break;
                    }
                    if (stored == 0.0) {
                        break;
                    }
                }
                double d = x.valueAt(k) - c;
                onSupport += d * d;
                centerOnSupport += c * c;
            }
            double norm = norms[j];
            double offSupport = norm - centerOnSupport;
            if (offSupport > SparseVector.CANCELLATION_GUARD * norm) {
                return onSupport + offSupport;
            }
            return onSupport + offSupportSquaredNorm(j, x);
        }

        /** Exact {@code sum of c_i^2} over the center's coordinates outside x's support, in index order. */
        private double offSupportSquaredNorm(int j, SparseVector x) {
            final int[] idx = indices[j];
            final double[] val = values[j];
            final int nnz = x.nnz();
            double s = 0.0;
            int k = 0;
            for (int t = 0; t < idx.length; t++) {
                int i = idx[t];
                while (k < nnz && x.indexAt(k) < i) {
                    k++;
                }
                if (k == nnz || x.indexAt(k) != i) {
                    s += val[t] * val[t];
                }
            }
            return s;
        }

        @Override
        void moveToMeans(SparseVector[] vectors, double[] weights, int[] assignment, double[] clusterWeight) {
            double[] w = sums.group(assignment, norms.length);
            System.arraycopy(w, 0, clusterWeight, 0, w.length);
            for (int j = 0; j < norms.length; j++) {
                if (clusterWeight[j] > 0.0) {
                    ClusterSums.Mean mean = sums.mean(j, false);
                    install(j, mean.indices, mean.values, mean.squaredNorm);
                }
            }
        }

        @Override
        void set(int j, SparseVector v) {
            int nnz = v.nnz();
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int k = 0; k < nnz; k++) {
                idx[k] = v.indexAt(k);
                val[k] = v.valueAt(k);
            }
            install(j, idx, val, v.squaredNorm());
        }

        private void install(int j, int[] idx, double[] val, double norm) {
            int capacity = 1;
            while (capacity < 2 * idx.length) {
                capacity <<= 1;
            }
            double[] table = new double[2 * capacity];
            int mask = capacity - 1;
            for (int t = 0; t < idx.length; t++) {
                int s = slot(idx[t], mask);
                while (table[2 * s] != 0.0) {
                    s = (s + 1) & mask;
                }
                table[2 * s] = idx[t] + 1.0;
                table[2 * s + 1] = val[t];
            }
            indices[j] = idx;
            values[j] = val;
            tables[j] = table;
            norms[j] = norm;
        }

        private static int slot(int index, int mask) {
            int h = index * 0x9E3779B9;
            return (h ^ (h >>> 16)) & mask;
        }

        @Override
        SparseVector center(int j) {
            return SparseVector.of(dim, indices[j], values[j]);
        }
    }
}
