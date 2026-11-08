package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Sanity references for a point set: per-feature scale and the cost and cluster balance of a k-means++ seeding. They
 * describe the data (docs/datasets.md); they are not optimal costs.
 *
 * @param totalVariance  weighted mean squared distance to the mean, i.e. the 1-means cost per unit weight
 * @param minStd         smallest per-feature (population) standard deviation; the median and largest follow
 * @param seeding        one entry per k
 */
public record DatasetStats(long n, int d, long nnz, double weightSum, double totalVariance, double minStd,
                           double medianStd, double maxStd, long seed, List<Seeding> seeding) {

    /**
     * Greedy k-means++ ({@link KMeansPlusPlus}, default local trials) with {@code SplittableRandom(seed)}, every point
     * assigned to its nearest center.
     *
     * @param cost    {@code sum w * min_c ||p - c||^2}
     * @param balance largest over smallest cluster weight
     */
    public record Seeding(int k, double cost, double minClusterWeight, double maxClusterWeight, double balance,
                          long millis) {
    }

    public DatasetStats {
        seeding = List.copyOf(seeding);
    }

    public static DatasetStats of(PointSet points, int[] ks, long seed) {
        final int d = points.dim();
        double[] mean = new double[d];
        double[] colWeight = new double[d]; // weight of the rows with an entry in the column
        double w = 0.0;
        for (int s = 0; s < points.segmentCount(); s++) {
            PointBlock b = points.segment(s);
            for (int i = 0; i < b.size(); i++) {
                double wi = b.weight(i);
                w += wi;
                for (int e = b.rowStart(i); e < b.rowEnd(i); e++) {
                    mean[b.index(e)] += wi * b.value(e);
                    colWeight[b.index(e)] += wi;
                }
            }
        }
        for (int c = 0; c < d; c++) {
            mean[c] /= w;
        }
        // Squared deviations of the stored entries, plus mean^2 for every implicit zero: no cancellation.
        double[] var = new double[d];
        for (int s = 0; s < points.segmentCount(); s++) {
            PointBlock b = points.segment(s);
            for (int i = 0; i < b.size(); i++) {
                double wi = b.weight(i);
                for (int e = b.rowStart(i); e < b.rowEnd(i); e++) {
                    double x = b.value(e) - mean[b.index(e)];
                    var[b.index(e)] += wi * x * x;
                }
            }
        }
        double total = 0.0;
        double[] std = new double[d];
        for (int c = 0; c < d; c++) {
            var[c] = (var[c] + (w - colWeight[c]) * mean[c] * mean[c]) / w;
            total += var[c];
            std[c] = Math.sqrt(var[c]);
        }
        Arrays.sort(std);
        double median = d % 2 == 1 ? std[d / 2] : 0.5 * (std[d / 2 - 1] + std[d / 2]);
        List<Seeding> seeding = new ArrayList<>(ks.length);
        for (int k : ks) {
            seeding.add(seed(points, k, seed));
        }
        return new DatasetStats(points.size(), d, points.storedEntries(), w, total, std[0], median, std[d - 1], seed,
                seeding);
    }

    static Seeding seed(PointSet points, int k, long seed) {
        long t0 = System.nanoTime();
        List<SparseVector> centers = KMeansPlusPlus.seed(points, k, new SplittableRandom(seed), MetricsSink.NOOP);
        KMeansResult r = WeightedLloyd.refine(points, centers, 0, MetricsSink.NOOP);
        long millis = (System.nanoTime() - t0) / 1_000_000;
        double[] size = new double[centers.size()];
        int[] assignment = r.assignment();
        for (int row = 0; row < assignment.length; row++) {
            size[assignment[row]] += points.weight(row);
        }
        double min = Double.POSITIVE_INFINITY;
        double max = 0.0;
        for (double s : size) {
            min = Math.min(min, s);
            max = Math.max(max, s);
        }
        return new Seeding(centers.size(), r.cost(), min, max, max / min, millis);
    }
}
