package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * The hard benchmark instance of Schwiegelshohn and Sheikh-Omar, "An Empirical Evaluation of k-Means Coresets"
 * (ESA 2022, Section 3): {@code n = k^alpha} points in {@code d = alpha * k} dimensions. Write row {@code r} in base
 * {@code k} with digits {@code r_0 .. r_(alpha-1)}; column block {@code a} of row {@code r} is {@code v_(r_a)}, where
 * {@code v_j} has {@code (k-1)/k} at position {@code j} and {@code -1/k} elsewhere. This is the paper's Kronecker
 * construction with rows in lexicographic order.
 *
 * <p>Clustering {@code C^a} groups rows by digit {@code a}. All {@code alpha} clusterings have {@code k} clusters of
 * {@code k^(alpha-1)} points, equal cost {@code n (alpha-1)(k-1)/k}, and any two clusters of different clusterings
 * share exactly {@code k^(alpha-2)} points, so a coreset must preserve every one of them. {@link #deltaSweep} is the
 * paper's distortion lower bound.
 */
public final class Esa22Benchmark {

    /** Largest instance generated in memory. */
    public static final long MAX_POINTS = 1L << 24;

    private final int k;
    private final int alpha;
    private final int n;

    private Esa22Benchmark(int k, int alpha, int n) {
        this.k = k;
        this.alpha = alpha;
        this.n = n;
    }

    /** @throws IllegalArgumentException unless {@code k >= 2}, {@code alpha >= 2} and {@code k^alpha <= 2^24} */
    public static Esa22Benchmark of(int k, int alpha) {
        if (k < 2 || alpha < 2) {
            throw new IllegalArgumentException("esa22 needs k >= 2 and alpha >= 2: k=" + k + ", alpha=" + alpha);
        }
        long n = 1;
        for (int a = 0; a < alpha; a++) {
            n *= k;
            if (n > MAX_POINTS) {
                throw new IllegalArgumentException("esa22 k^alpha exceeds " + MAX_POINTS + ": k=" + k + ", alpha="
                        + alpha);
            }
        }
        return new Esa22Benchmark(k, alpha, (int) n);
    }

    public int k() {
        return k;
    }

    public int alpha() {
        return alpha;
    }

    public int n() {
        return n;
    }

    public int dim() {
        return alpha * k;
    }

    /** Unit-weight rows in index order. */
    public List<WeightedPoint> points() {
        final int d = dim();
        final double on = (k - 1.0) / k;
        final double off = -1.0 / k;
        int[] indices = new int[d];
        for (int t = 0; t < d; t++) {
            indices[t] = t;
        }
        List<WeightedPoint> out = new ArrayList<>(n);
        double[] values = new double[d];
        for (int r = 0; r < n; r++) {
            int rest = r;
            for (int a = 0; a < alpha; a++) {
                int digit = rest % k;
                rest /= k;
                for (int b = 0; b < k; b++) {
                    values[a * k + b] = b == digit ? on : off;
                }
            }
            out.add(new WeightedPoint(SparseVector.of(d, indices, values), 1.0));
        }
        return out;
    }

    /** Cluster of row {@code r} in clustering {@code C^a}: digit {@code a} of {@code r} in base {@code k}. */
    public int clusterOf(int a, int r) {
        int rest = r;
        for (int i = 0; i < a; i++) {
            rest /= k;
        }
        return rest % k;
    }

    /** Mean of cluster {@code j} of {@code C^a}: {@code v_j} in block {@code a}, zero elsewhere. */
    public SparseVector center(int a, int j) {
        int[] idx = new int[k];
        double[] val = new double[k];
        for (int b = 0; b < k; b++) {
            idx[b] = a * k + b;
            val[b] = b == j ? (k - 1.0) / k : -1.0 / k;
        }
        return SparseVector.of(dim(), idx, val);
    }

    public int clusterSize() {
        return n / k;
    }

    /** Squared distance of every row to the mean of its own cluster, in any clustering: {@code (alpha-1)(k-1)/k}. */
    public double pointCostToOwnCenter() {
        return (alpha - 1.0) * (k - 1.0) / k;
    }

    /**
     * Exact cost on the full instance of the centers of the non-deficient clusters of one clustering: a row in a
     * deficient cluster pays {@code ||v_j - v_h||^2 = 2} on top of its own-center cost.
     */
    public double fullCost(int deficientClusters) {
        double c0 = pointCostToOwnCenter();
        return (double) clusterSize() * ((k - deficientClusters) * c0 + deficientClusters * (c0 + 2.0));
    }

    /**
     * Result of the delta sweep.
     *
     * @param delta             deficiency threshold reaching the largest distortion
     * @param clustering        index {@code a} of the clustering used at that threshold
     * @param deficientClusters clusters {@code C_i^a} whose coreset mass is below {@code |C_i^a| (1 - delta)}
     */
    public record DeltaSweep(double delta, int clustering, int deficientClusters, double costFull, double costCoreset,
                             double distortion) {
    }

    /** {@code steps} thresholds {@code 0, 1/steps, ..., (steps-1)/steps}. */
    public static double[] deltaGrid(int steps) {
        double[] out = new double[steps];
        for (int i = 0; i < steps; i++) {
            out[i] = (double) i / steps;
        }
        return out;
    }

    /**
     * For each threshold, takes the clustering with the most deficient clusters (keeping at least one center), places
     * centers at its non-deficient cluster means and compares full and coreset k-means cost; returns the threshold
     * with the largest distortion. Coreset points are assigned to the nearest cluster mean, which for input rows is
     * their own cluster. Empty when every clustering is entirely deficient at every threshold.
     */
    public Optional<DeltaSweep> deltaSweep(Coreset coreset, double[] deltas) {
        double[][] mass = new double[alpha][k];
        double[] scratch = new double[k];
        List<Centers> clusterings = new ArrayList<>(alpha);
        for (int a = 0; a < alpha; a++) {
            List<SparseVector> centers = new ArrayList<>(k);
            for (int j = 0; j < k; j++) {
                centers.add(center(a, j));
            }
            Centers c = Centers.of(centers);
            clusterings.add(c);
            for (WeightedPoint p : coreset.points()) {
                mass[a][c.nearest(p.vector(), scratch)] += p.weight();
            }
        }
        Problem kMeans = Problem.kMeans();
        DeltaSweep best = null;
        for (double delta : deltas) {
            double need = clusterSize() * (1.0 - delta);
            int bestA = -1;
            int bestCount = -1;
            for (int a = 0; a < alpha; a++) {
                int count = 0;
                for (int j = 0; j < k; j++) {
                    if (mass[a][j] < need) {
                        count++;
                    }
                }
                if (count < k && count > bestCount) {
                    bestCount = count;
                    bestA = a;
                }
            }
            if (bestA < 0) {
                continue;
            }
            List<SparseVector> kept = new ArrayList<>(k - bestCount);
            for (int j = 0; j < k; j++) {
                if (mass[bestA][j] >= need) {
                    kept.add(clusterings.get(bestA).vectors().get(j));
                }
            }
            double full = fullCost(bestCount);
            double onCoreset = kMeans.cost(coreset, Centers.of(kept));
            double ratio = onCoreset / full;
            double distortion = ratio >= 1.0 ? ratio : (ratio > 0.0 ? 1.0 / ratio : Double.POSITIVE_INFINITY);
            if (best == null || distortion > best.distortion()) {
                best = new DeltaSweep(delta, bestA, bestCount, full, onCoreset, distortion);
            }
        }
        return Optional.ofNullable(best);
    }
}
