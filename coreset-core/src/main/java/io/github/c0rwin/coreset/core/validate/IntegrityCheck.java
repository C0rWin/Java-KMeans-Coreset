package io.github.c0rwin.coreset.core.validate;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import java.util.List;

/**
 * One-pass sanity checks of a coreset against its input. Total weight, first moment {@code sum w p} and second moment
 * {@code offset + sum w ||p||^2} (the cost for Q = {0}) are preserved exactly by the mean coreset and in expectation
 * by the sampling coresets; the mean coreset also never underestimates any cost.
 */
public final class IntegrityCheck {

    private IntegrityCheck() {
    }

    /**
     * Relative errors are {@code |coreset - input| / |input|} for the total weight and the second moment; when the
     * input value is 0 the absolute error is reported instead.
     *
     * <p>The first moment is normalised by its natural scale {@code sqrt(W * E)} instead of its own norm, where W is
     * the input weight and {@code E = offset + sum w ||p||^2} its second moment. By Cauchy-Schwarz
     * {@code |sum w p| <= sqrt(W * E)}, so the value is at most the plain relative error, and it stays meaningful for
     * data centered at the origin, where {@code sum w p} is near 0. It reads as the error of the weighted mean in
     * units of the input's RMS norm.
     *
     * @param negativeWeights coreset points with negative weight; 0 while {@code WeightedPoint} rejects them, kept so
     *                        the report matches the validity-report schema
     * @param nonFinite       coreset points whose weight or {@code w * ||p||^2} is not finite
     */
    public record Report(double weightRelErr, double firstMomentRelErr, double secondMomentRelErr,
                         int negativeWeights, int nonFinite) {
    }

    /** @throws IllegalArgumentException if both are non-empty and their dimensions differ */
    public static Report check(Coreset input, Coreset coreset) {
        int dim = dimension(input, coreset);
        double[] inFirst = new double[dim];
        double[] outFirst = new double[dim];
        Moments in = moments(input.pointSet(), inFirst);
        Moments out = moments(coreset.pointSet(), outFirst);
        double diff = 0.0;
        for (int d = 0; d < dim; d++) {
            double e = outFirst[d] - inFirst[d];
            diff += e * e;
        }
        double inSecond = input.offset() + in.second;
        double scale = Math.sqrt(in.weight) * Math.sqrt(inSecond); // two roots: the product may overflow
        return new Report(
                relErr(out.weight, in.weight),
                scale > 0.0 ? Math.sqrt(diff) / scale : Math.sqrt(diff),
                relErr(coreset.offset() + out.second, inSecond),
                out.negative, out.nonFinite);
    }

    /**
     * Number of center sets Q with {@code cost(coreset, Q) < (1 - relTolerance) * cost(input, Q)}. Must be 0 for the
     * mean coreset (Barger-Feldman Lemma 1); any violation means its additive offset is missing or wrong.
     */
    public static int oneSidedViolations(Coreset input, Coreset coreset, List<List<SparseVector>> queries,
                                         double relTolerance) {
        int violations = 0;
        for (List<SparseVector> q : queries) {
            if (Cost.cost(coreset, q) < (1.0 - relTolerance) * Cost.cost(input, q)) {
                violations++;
            }
        }
        return violations;
    }

    private static int dimension(Coreset a, Coreset b) {
        int da = a.isEmpty() ? -1 : a.pointSet().dim();
        int db = b.isEmpty() ? -1 : b.pointSet().dim();
        if (da >= 0 && db >= 0 && da != db) {
            throw new IllegalArgumentException("dimension mismatch: input " + da + ", coreset " + db);
        }
        return Math.max(Math.max(da, db), 0);
    }

    private record Moments(double weight, double second, int negative, int nonFinite) {
    }

    private static Moments moments(PointSet points, double[] first) {
        double weight = 0.0;
        double second = 0.0;
        int negative = 0;
        int nonFinite = 0;
        for (int t = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0, n = b.size(); i < n; i++) {
                double w = b.weight(i);
                double s = w * b.squaredNorm(i);
                if (w < 0.0) {
                    negative++;
                }
                if (!Double.isFinite(w) || !Double.isFinite(s)) {
                    nonFinite++;
                }
                weight += w;
                second += s;
                b.addTo(i, first, w);
            }
        }
        return new Moments(weight, second, negative, nonFinite);
    }

    private static double relErr(double value, double reference) {
        double e = Math.abs(value - reference);
        return reference != 0.0 ? e / Math.abs(reference) : e;
    }
}
