package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.stream.IntStream;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

/** Seeded, unbalanced Gaussian mixtures (component j drawn with probability proportional to j + 1, unit variance). */
final class Mixture {

    private Mixture() {
    }

    static double[][] centers(int k, int d, double spread, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] c = new double[k][d];
        for (double[] row : c) {
            for (int j = 0; j < d; j++) {
                row[j] = (2 * rnd.nextDouble() - 1) * spread;
            }
        }
        return c;
    }

    /** Weights 1, or uniform in [0.5, 2) when {@code weighted}. */
    static List<WeightedPoint> points(double[][] centers, int n, long seed, boolean weighted) {
        SplittableRandom rnd = new SplittableRandom(seed);
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

    /** {@code partitions * perPartition} points, generated inside each partition from {@code (seed, partition)}. */
    static JavaRDD<WeightedPoint> rdd(JavaSparkContext jsc, double[][] centers, int partitions, int perPartition,
                                      long seed, boolean weighted) {
        return jsc.parallelize(IntStream.range(0, partitions).boxed().toList(), partitions)
                .flatMap(i -> points(centers, perPartition, seed * 1_000_003L + i, weighted).iterator());
    }
}
