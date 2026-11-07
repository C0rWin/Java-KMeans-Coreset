package io.github.c0rwin.coreset.core.validate;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointBlock;
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
     * Relative errors are {@code |coreset - input| / |input|} (vector norm for the first moment); when the input value
     * is 0 the absolute error is reported instead.
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
        Moments in = moments(input.block(), inFirst);
        Moments out = moments(coreset.block(), outFirst);
        double diff = 0.0;
        double ref = 0.0;
        for (int d = 0; d < dim; d++) {
            double e = outFirst[d] - inFirst[d];
            diff += e * e;
            ref += inFirst[d] * inFirst[d];
        }
        return new Report(
                relErr(out.weight, in.weight),
                ref > 0.0 ? Math.sqrt(diff / ref) : Math.sqrt(diff),
                relErr(coreset.offset() + out.second, input.offset() + in.second),
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
        int da = a.isEmpty() ? -1 : a.block().dim();
        int db = b.isEmpty() ? -1 : b.block().dim();
        if (da >= 0 && db >= 0 && da != db) {
            throw new IllegalArgumentException("dimension mismatch: input " + da + ", coreset " + db);
        }
        return Math.max(Math.max(da, db), 0);
    }

    private record Moments(double weight, double second, int negative, int nonFinite) {
    }

    private static Moments moments(PointBlock points, double[] first) {
        double weight = 0.0;
        double second = 0.0;
        int negative = 0;
        int nonFinite = 0;
        for (int i = 0, n = points.size(); i < n; i++) {
            double w = points.weight(i);
            double s = w * points.squaredNorm(i);
            if (w < 0.0) {
                negative++;
            }
            if (!Double.isFinite(w) || !Double.isFinite(s)) {
                nonFinite++;
            }
            weight += w;
            second += s;
            points.addTo(i, first, w);
        }
        return new Moments(weight, second, negative, nonFinite);
    }

    private static double relErr(double value, double reference) {
        double e = Math.abs(value - reference);
        return reference != 0.0 ? e / Math.abs(reference) : e;
    }
}
