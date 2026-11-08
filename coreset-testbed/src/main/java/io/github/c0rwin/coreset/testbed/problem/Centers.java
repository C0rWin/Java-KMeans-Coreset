package io.github.c0rwin.coreset.testbed.problem;

import io.github.c0rwin.coreset.core.SparseVector;
import java.util.List;

/**
 * A center set laid out for repeated nearest-center queries. Coordinates are stored dimension-major
 * ({@code values[t * k + j]} is coordinate {@code t} of center {@code j}), so each nonzero of a query point touches
 * one contiguous run of {@code k} doubles and all {@code k} dot products are accumulated in a single sweep; distances
 * follow from {@code ||p||^2 - 2 p.c + ||c||^2}, clamped at 0. One {@code k * d} array per set; queries allocate
 * nothing (callers pass a scratch array of at least {@link #size()} doubles).
 */
public final class Centers {

    private final List<SparseVector> vectors;
    private final int k;
    private final int dim;
    private final double[] values;
    private final double[] norms;

    private Centers(List<SparseVector> vectors) {
        if (vectors.isEmpty()) {
            throw new IllegalArgumentException("a center set needs at least one center");
        }
        this.vectors = List.copyOf(vectors);
        this.k = vectors.size();
        this.dim = vectors.get(0).dim();
        this.values = new double[Math.multiplyExact(k, dim)];
        this.norms = new double[k];
        for (int j = 0; j < k; j++) {
            SparseVector c = vectors.get(j);
            if (c.dim() != dim) {
                throw new IllegalArgumentException("center " + j + " has dim " + c.dim() + ", expected " + dim);
            }
            for (int t = 0, nnz = c.nnz(); t < nnz; t++) {
                values[c.indexAt(t) * k + j] = c.valueAt(t);
            }
            norms[j] = c.squaredNorm();
        }
    }

    public static Centers of(List<SparseVector> vectors) {
        return new Centers(vectors);
    }

    public List<SparseVector> vectors() {
        return vectors;
    }

    public int size() {
        return k;
    }

    public int dim() {
        return dim;
    }

    /** Fills {@code scratch[0..k)} with {@code ||p - c_j||^2}. */
    private void distances(SparseVector p, double[] scratch) {
        if (p.dim() != dim) {
            throw new IllegalArgumentException("point has dim " + p.dim() + ", centers have " + dim);
        }
        final int kk = k;
        final double[] v = values;
        for (int j = 0; j < kk; j++) {
            scratch[j] = 0.0;
        }
        double pn = 0.0;
        for (int t = 0, nnz = p.nnz(); t < nnz; t++) {
            double x = p.valueAt(t);
            pn += x * x;
            int base = p.indexAt(t) * kk;
            for (int j = 0; j < kk; j++) {
                scratch[j] += x * v[base + j];
            }
        }
        for (int j = 0; j < kk; j++) {
            double d = pn - 2.0 * scratch[j] + norms[j];
            scratch[j] = d > 0.0 ? d : 0.0;
        }
    }

    /** Squared Euclidean distance from {@code p} to its nearest center. */
    public double minSquaredDistance(SparseVector p, double[] scratch) {
        distances(p, scratch);
        double best = scratch[0];
        for (int j = 1; j < k; j++) {
            if (scratch[j] < best) {
                best = scratch[j];
            }
        }
        return best;
    }

    /** Index of the nearest center; ties go to the lowest index. */
    public int nearest(SparseVector p, double[] scratch) {
        distances(p, scratch);
        int best = 0;
        for (int j = 1; j < k; j++) {
            if (scratch[j] < scratch[best]) {
                best = j;
            }
        }
        return best;
    }
}
