package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Seeded, unbalanced Gaussian mixture addressed by record index, so a parallel source can generate any record
 * independently: component j is drawn with probability proportional to j + 1, coordinates have unit variance, and
 * weights are 1 or uniform in [0.5, 2).
 */
final class Mixture implements Serializable {

    private static final long serialVersionUID = 1L;

    private final double[][] centers;
    private final long seed;
    private final boolean weighted;

    Mixture(int k, int d, double spread, long seed, boolean weighted) {
        SplittableRandom rnd = new SplittableRandom(seed);
        this.centers = new double[k][d];
        for (double[] row : centers) {
            for (int j = 0; j < d; j++) {
                row[j] = (2 * rnd.nextDouble() - 1) * spread;
            }
        }
        this.seed = seed;
        this.weighted = weighted;
    }

    WeightedPoint point(long index) {
        SplittableRandom rnd = new SplittableRandom(MergeRandomness.derive(seed, 1, 0, index));
        int k = centers.length;
        double u = rnd.nextDouble() * (k * (k + 1) / 2.0);
        int c = 0;
        for (double acc = 1; acc <= u && c < k - 1; acc += c + 1) {
            c++;
        }
        double[] v = new double[centers[c].length];
        for (int j = 0; j < v.length; j++) {
            v[j] = centers[c][j] + rnd.nextGaussian();
        }
        return new WeightedPoint(SparseVector.dense(v), weighted ? 0.5 + 1.5 * rnd.nextDouble() : 1.0);
    }

    /** Records {@code [from, to)} as one coreset. */
    Coreset coreset(long from, long to) {
        List<WeightedPoint> out = new ArrayList<>((int) (to - from));
        for (long i = from; i < to; i++) {
            out.add(point(i));
        }
        return Coreset.of(out);
    }
}
