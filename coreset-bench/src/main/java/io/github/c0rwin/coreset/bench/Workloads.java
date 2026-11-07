package io.github.c0rwin.coreset.bench;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.SplittableRandom;

/** Seeded synthetic inputs shared by the benchmarks and harnesses; identical seeds give identical data. */
public final class Workloads {

    private Workloads() {
    }

    /**
     * {@code n} unit-weight points from {@code clusters} isotropic Gaussians (sigma 1) whose means are uniform in
     * {@code [-50, 50]^d}; every coordinate is stored, so {@code nnz = d}.
     */
    public static List<WeightedPoint> gaussianMixture(int n, int d, int clusters, long seed) {
        SplittableRandom rng = new SplittableRandom(seed);
        double[][] means = new double[clusters][d];
        for (double[] m : means) {
            for (int j = 0; j < d; j++) {
                m[j] = rng.nextDouble(-50.0, 50.0);
            }
        }
        List<WeightedPoint> out = new ArrayList<>(n);
        double[] x = new double[d];
        for (int i = 0; i < n; i++) {
            double[] m = means[rng.nextInt(clusters)];
            for (int j = 0; j < d; j++) {
                x[j] = m[j] + rng.nextGaussian();
            }
            out.add(new WeightedPoint(SparseVector.dense(x), 1.0));
        }
        return out;
    }

    /** Sparse vector of dimension {@code dim} with exactly {@code nnz} distinct uniformly placed non-zeros. */
    public static SparseVector sparse(int dim, int nnz, SplittableRandom rng) {
        if (nnz > dim) {
            throw new IllegalArgumentException("nnz " + nnz + " > dim " + dim);
        }
        int[] idx = new int[nnz];
        double[] val = new double[nnz];
        HashSet<Integer> chosen = new HashSet<>(nnz * 2);
        int w = 0;
        for (int j = dim - nnz; j < dim; j++) {
            int t = rng.nextInt(j + 1);
            int pick = chosen.contains(t) ? j : t;
            chosen.add(pick);
            idx[w] = pick;
            val[w] = 1.0 + rng.nextDouble();
            w++;
        }
        return SparseVector.of(dim, idx, val);
    }

    /** Splits {@code points} into consecutive immutable batches of {@code batchSize} (the last may be shorter). */
    public static List<List<WeightedPoint>> batches(List<WeightedPoint> points, int batchSize) {
        List<List<WeightedPoint>> out = new ArrayList<>((points.size() + batchSize - 1) / batchSize);
        for (int from = 0; from < points.size(); from += batchSize) {
            out.add(List.copyOf(points.subList(from, Math.min(points.size(), from + batchSize))));
        }
        return out;
    }
}

