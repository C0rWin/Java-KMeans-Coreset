package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** Seeded synthetic inputs shared by the algorithm and validation tests. */
public final class TestData {

    private TestData() {
    }

    /** Gaussian mixture centers: k points uniform in {@code [-spread, spread]^d}. */
    public static double[][] mixtureCenters(int k, int d, double spread, SplittableRandom rnd) {
        double[][] centers = new double[k][d];
        for (double[] c : centers) {
            for (int j = 0; j < d; j++) {
                c[j] = (2 * rnd.nextDouble() - 1) * spread;
            }
        }
        return centers;
    }

    /**
     * n points from an unbalanced mixture around {@code centers} (component j drawn with probability proportional to
     * {@code j + 1}), unit-variance noise; weights 1, or uniform in [0.5, 2) when {@code weighted}.
     */
    public static List<WeightedPoint> mixture(double[][] centers, int n, boolean weighted, SplittableRandom rnd) {
        int k = centers.length;
        int d = centers[0].length;
        double total = k * (k + 1) / 2.0;
        List<WeightedPoint> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            double u = rnd.nextDouble() * total;
            int c = 0;
            for (double acc = 1; acc <= u && c < k - 1; acc += c + 1) {
                c++;
            }
            double[] v = new double[d];
            for (int j = 0; j < d; j++) {
                v[j] = centers[c][j] + rnd.nextGaussian();
            }
            out.add(new WeightedPoint(SparseVector.dense(v), weighted ? 0.5 + 1.5 * rnd.nextDouble() : 1.0));
        }
        return out;
    }

    public static Coreset mixture(int n, int k, int d, boolean weighted, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        return Coreset.of(mixture(mixtureCenters(k, d, 10.0, rnd), n, weighted, rnd));
    }

    /** k centers uniform in {@code [-scale, scale]^d}. */
    public static List<SparseVector> randomCenters(int k, int d, double scale, SplittableRandom rnd) {
        List<SparseVector> out = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            double[] v = new double[d];
            for (int j = 0; j < d; j++) {
                v[j] = (2 * rnd.nextDouble() - 1) * scale;
            }
            out.add(SparseVector.dense(v));
        }
        return out;
    }

    public static WeightedPoint point(double weight, double... coords) {
        return new WeightedPoint(SparseVector.dense(coords), weight);
    }
}
