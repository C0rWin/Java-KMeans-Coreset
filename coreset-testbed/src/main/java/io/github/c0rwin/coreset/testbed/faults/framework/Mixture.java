package io.github.c0rwin.coreset.testbed.faults.framework;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A dense Gaussian mixture with component labels (the record class), D^z seeding, the two-term sensitivity proposal
 * and cost tables through the testbed's {@link Problem} objective.
 */
final class Mixture {

    final int n;
    final int d;
    final int components;
    final double[] x;
    final int[] label;
    final double[][] means;

    private Mixture(int n, int d, int components, double[] x, int[] label, double[][] means) {
        this.n = n;
        this.d = d;
        this.components = components;
        this.x = x;
        this.label = label;
        this.means = means;
    }

    /** Means uniform in {@code [0, box]^d}, isotropic {@code sigma}, component sizes proportional to {@code 1/(t+1)^imbalance}. */
    static Mixture gaussian(int n, int d, int components, double sigma, double box, double imbalance, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] means = new double[components][d];
        for (double[] m : means) {
            for (int t = 0; t < d; t++) {
                m[t] = box * rnd.nextDouble();
            }
        }
        int[] sizes = zipfSizes(n, components, imbalance);
        double[] x = new double[n * d];
        int[] label = new int[n];
        for (int c = 0, i = 0; c < components; c++) {
            for (int s = 0; s < sizes[c]; s++, i++) {
                label[i] = c;
                for (int t = 0; t < d; t++) {
                    x[i * d + t] = means[c][t] + sigma * gaussian(rnd);
                }
            }
        }
        return new Mixture(n, d, components, x, label, means);
    }

    static int[] zipfSizes(int n, int k, double imbalance) {
        double[] w = new double[k];
        double total = 0.0;
        for (int c = 0; c < k; c++) {
            w[c] = Math.pow(c + 1.0, -imbalance);
            total += w[c];
        }
        int[] sizes = new int[k];
        int used = 0;
        for (int c = 0; c < k; c++) {
            sizes[c] = Math.max(1, (int) Math.floor(n * w[c] / total));
            used += sizes[c];
        }
        sizes[0] += n - used;
        return sizes;
    }

    private static double gaussian(SplittableRandom rnd) {
        double u = 1.0 - rnd.nextDouble();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * rnd.nextDouble());
    }

    /** {@code ||x_row - c_j||^z} with {@code c_j} at offset {@code j d} of {@code centers}. */
    double distZ(int row, double[] centers, int j, double z) {
        double s = 0.0;
        int a = row * d;
        int b = j * d;
        for (int t = 0; t < d; t++) {
            double v = x[a + t] - centers[b + t];
            s += v * v;
        }
        return z == 2.0 ? s : z == 1.0 ? Math.sqrt(s) : Math.pow(s, z / 2.0);
    }

    /** {@code D^z} seeding of the weighted rows: k centers, each drawn with probability proportional to {@code w D^z}. */
    double[] seed(int[] rows, double[] w, int k, double z, RandomGenerator rng) {
        int size = rows.length;
        double[] centers = new double[k * d];
        double[] best = new double[size];
        Arrays.fill(best, Double.POSITIVE_INFINITY);
        for (int c = 0; c < k; c++) {
            double total = 0.0;
            for (int t = 0; t < size; t++) {
                total += c == 0 ? w[t] : w[t] * best[t];
            }
            boolean byWeight = c == 0 || !(total > 0.0);
            if (byWeight && c > 0) {
                total = 0.0;
                for (int t = 0; t < size; t++) {
                    total += w[t];
                }
            }
            double u = rng.nextDouble() * total;
            int pick = size - 1;
            double acc = 0.0;
            for (int t = 0; t < size; t++) {
                acc += byWeight ? w[t] : w[t] * best[t];
                if (acc > u) {
                    pick = t;
                    break;
                }
            }
            System.arraycopy(x, rows[pick] * d, centers, c * d, d);
            for (int t = 0; t < size; t++) {
                best[t] = Math.min(best[t], distZ(rows[t], centers, c, z));
            }
        }
        return centers;
    }

    /**
     * The usual two-term sensitivity proposal around a {@code D^z} seeding {@code B}:
     * {@code s_t = 2 w_t dist(x_t, B)^z / cost_z(B) + w_t / W(cluster of t)}.
     */
    double[] proposal(int[] rows, double[] w, int k, double z, RandomGenerator rng) {
        double[] centers = seed(rows, w, k, z, rng);
        int size = rows.length;
        int[] near = new int[size];
        double[] dz = new double[size];
        double[] clusterWeight = new double[k];
        double cost = 0.0;
        for (int t = 0; t < size; t++) {
            double b = Double.POSITIVE_INFINITY;
            for (int c = 0; c < k; c++) {
                double v = distZ(rows[t], centers, c, z);
                if (v < b) {
                    b = v;
                    near[t] = c;
                }
            }
            dz[t] = b;
            cost += w[t] * b;
            clusterWeight[near[t]] += w[t];
        }
        double[] s = new double[size];
        for (int t = 0; t < size; t++) {
            s[t] = (cost > 0.0 ? 2.0 * w[t] * dz[t] / cost : 0.0) + w[t] / clusterWeight[near[t]];
        }
        return s;
    }

    /** {@code table[row * Q + j] = problem.pointCost(x_row, Q_j)} for every row. */
    double[] table(List<double[]> centerSets, int k, Problem problem) {
        int qs = centerSets.size();
        List<Centers> cs = new ArrayList<>(qs);
        for (double[] c : centerSets) {
            List<SparseVector> v = new ArrayList<>(k);
            for (int j = 0; j < k; j++) {
                v.add(SparseVector.dense(Arrays.copyOfRange(c, j * d, (j + 1) * d)));
            }
            cs.add(Centers.of(v));
        }
        double[] table = new double[n * qs];
        double[] scratch = new double[k];
        double[] row = new double[d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x, i * d, row, 0, d);
            SparseVector p = SparseVector.dense(row);
            for (int j = 0; j < qs; j++) {
                table[i * qs + j] = problem.pointCost(p, cs.get(j), scratch);
            }
        }
        return table;
    }
}
