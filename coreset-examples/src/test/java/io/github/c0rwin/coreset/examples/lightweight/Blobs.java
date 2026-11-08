package io.github.c0rwin.coreset.examples.lightweight;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** Unbalanced weighted Gaussian blobs, and the distortion of a coreset of them. */
final class Blobs {

    static final int K = 8;
    static final int D = 6;

    private Blobs() {
    }

    /** {@code n} points; blob {@code j} gets a share proportional to {@code j + 1}, weights uniform in [0.5, 1.5). */
    static List<WeightedPoint> points(int n, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] centers = new double[K][D];
        for (double[] c : centers) {
            for (int j = 0; j < D; j++) {
                c[j] = 20.0 * rnd.nextDouble();
            }
        }
        List<WeightedPoint> out = new ArrayList<>(n);
        int shares = K * (K + 1) / 2;
        for (int i = 0; i < n; i++) {
            int pick = rnd.nextInt(shares);
            int blob = 0;
            while (pick >= blob + 1) {
                pick -= blob + 1;
                blob++;
            }
            double[] x = new double[D];
            for (int j = 0; j < D; j++) {
                x[j] = centers[blob][j] + rnd.nextGaussian();
            }
            out.add(new WeightedPoint(SparseVector.dense(x), 0.5 + rnd.nextDouble()));
        }
        return out;
    }

    /** Largest relative cost error of {@code coreset} over the standard candidate center sets for {@code K}. */
    static double distortion(Coreset full, Coreset coreset) {
        List<Candidate> candidates = CandidateSolutions.standard(full, coreset, K, 10, 3, new SplittableRandom(99));
        return Distortion.evaluate(full, coreset, candidates).max();
    }
}
