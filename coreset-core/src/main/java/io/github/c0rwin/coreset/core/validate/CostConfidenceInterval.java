package io.github.c0rwin.coreset.core.validate;

import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.SampledCoreset;
import java.util.List;

/**
 * Confidence intervals for {@code cost(P, Q)} computed from a sampled coreset alone, without a pass over P.
 *
 * <p>With m i.i.d. draws, draw j of point p contributes the Horvitz-Thompson term {@code T_j = w_p d(p, Q)^2 / q_p},
 * and {@code mean(T) = cost(C, Q) - offset} is unbiased for {@code cost(P, Q) - offset}.
 *
 * <p><b>Caveat:</b> both intervals hold only for a Q chosen independently of the sample. A Q optimised on the coreset
 * (e.g. Lloyd on C) is selected using the same draws and the coverage guarantee is lost; evaluate such a Q on an
 * independent second sample or on P.
 *
 * <p><b>Scope:</b> P is the exact input of a single sampling step, as returned by
 * {@code SensitivityCoreset.compressWithProvenance}. Coresets from {@code MergeReduceTree}, the node or a merged
 * snapshot went through several random reductions and carry no per-level provenance, so these intervals do not apply
 * to them: the variance of every level would have to be added, which is not tracked.
 */
public final class CostConfidenceInterval {

    private CostConfidenceInterval() {
    }

    /** Bounds and point estimate of {@code cost(P, Q)}, offsets included; {@code upper} may be +infinity. */
    public record Interval(double lower, double estimate, double upper) {

        public boolean contains(double value) {
            return lower <= value && value <= upper;
        }
    }

    /**
     * CLT interval {@code mean(T) +- z * s / sqrt(m)}, with s the sample standard deviation of the m terms; the lower
     * end is clamped at the offset. Exact coresets give a zero-width interval.
     *
     * @throws IllegalArgumentException unless {@code 0 < confidence < 1}, or if the coreset has exactly one draw
     */
    public static Interval normal(SampledCoreset sample, List<SparseVector> q, double confidence) {
        if (!(confidence > 0.0 && confidence < 1.0)) {
            throw new IllegalArgumentException("confidence must be in (0, 1): " + confidence);
        }
        if (sample.exact()) {
            return exact(sample, q);
        }
        double offset = sample.coreset().offset();
        final int m = sample.draws();
        if (m < 2) {
            throw new IllegalArgumentException("the sample variance needs at least 2 draws");
        }
        double[] terms = terms(sample, q);
        int[] counts = sample.drawCounts();
        double mean = 0.0;
        for (int j = 0; j < terms.length; j++) {
            mean += counts[j] * terms[j];
        }
        mean /= m;
        double ss = 0.0;
        for (int j = 0; j < terms.length; j++) {
            double e = terms[j] - mean;
            ss += counts[j] * e * e;
        }
        double halfWidth = normalQuantile(0.5 + confidence / 2.0) * Math.sqrt(ss / (m - 1) / m);
        return new Interval(Math.max(mean - halfWidth, 0.0) + offset, mean + offset, mean + halfWidth + offset);
    }

    /**
     * Distribution-free interval from Hoeffding's inequality (Bachem, Lucic, Krause 2017, eq. 2.8): if every
     * {@code s_p} bounds the true sensitivity, each term lies in {@code [0, S * cost]} with S the sample's
     * {@link SampledCoreset#totalSensitivity()} (the bound's actual total), so with probability at least
     * {@code 1 - delta} the relative error is at most {@code eps = S * sqrt(ln(2 / delta) / (2 m))}. Very wide in
     * practice (upper end +infinity once {@code eps >= 1}).
     *
     * <p>S is the total of whichever bound and alpha produced the sample. The guarantee holds only for
     * {@code SensitivityBound.standard(alpha)} when the seeding's cost is within {@code alpha} of the optimum: with
     * {@code alpha = kMeansPlusPlusAlpha(k)} that holds with probability at least 1/2 per seeding. With the default
     * {@code DEFAULT_ALPHA = 2} or the legacy bound it is not ensured, so the interval is then heuristic.
     *
     * @throws IllegalArgumentException unless {@code 0 < delta < 1}
     */
    public static Interval hoeffding(SampledCoreset sample, List<SparseVector> q, double delta) {
        if (!(delta > 0.0 && delta < 1.0)) {
            throw new IllegalArgumentException("delta must be in (0, 1): " + delta);
        }
        if (sample.exact()) {
            return exact(sample, q);
        }
        double offset = sample.coreset().offset();
        double estimate = Cost.cost(sample.coreset().points(), q);
        double eps = sample.totalSensitivity() * Math.sqrt(Math.log(2.0 / delta) / (2.0 * sample.draws()));
        double upper = eps < 1.0 ? estimate / (1.0 - eps) + offset : Double.POSITIVE_INFINITY;
        return new Interval(estimate / (1.0 + eps) + offset, estimate + offset, upper);
    }

    private static Interval exact(SampledCoreset sample, List<SparseVector> q) {
        double c = Cost.cost(sample.coreset(), q);
        return new Interval(c, c, c);
    }

    /** Per coreset point, the term {@code w_p f(p) / q_p = u_p * m * f(p) / c_p} of each of its draws. */
    private static double[] terms(SampledCoreset sample, List<SparseVector> q) {
        List<WeightedPoint> points = sample.coreset().points();
        int[] counts = sample.drawCounts();
        double[] terms = new double[points.size()];
        for (int j = 0; j < terms.length; j++) {
            WeightedPoint p = points.get(j);
            double f = p.vector().squaredDistance(q.get(Cost.nearest(p.vector(), q)));
            terms[j] = p.weight() * sample.draws() / counts[j] * f;
        }
        return terms;
    }

    /** Standard normal quantile (Acklam's rational approximation, relative error below 1.2e-9). */
    static double normalQuantile(double p) {
        if (!(p > 0.0 && p < 1.0)) {
            throw new IllegalArgumentException("p must be in (0, 1): " + p);
        }
        final double low = 0.02425;
        if (p < low) {
            double t = Math.sqrt(-2.0 * Math.log(p));
            return tail(t);
        }
        if (p > 1.0 - low) {
            double t = Math.sqrt(-2.0 * Math.log1p(-p));
            return -tail(t);
        }
        double t = p - 0.5;
        double r = t * t;
        return (((((-3.969683028665376e+01 * r + 2.209460984245205e+02) * r - 2.759285104469687e+02) * r
                + 1.383577518672690e+02) * r - 3.066479806614716e+01) * r + 2.506628277459239e+00) * t
                / (((((-5.447609879822406e+01 * r + 1.615858368580409e+02) * r - 1.556989798598866e+02) * r
                + 6.680131188771972e+01) * r - 1.328068155288572e+01) * r + 1.0);
    }

    private static double tail(double t) {
        return (((((-7.784894002430293e-03 * t - 3.223964580411365e-01) * t - 2.400758277161838e+00) * t
                - 2.549732539343734e+00) * t + 4.374664141464968e+00) * t + 2.938163982698783e+00)
                / ((((7.784695709041462e-03 * t + 3.224671290700398e-01) * t + 2.445134137142996e+00) * t
                + 3.754408661907416e+00) * t + 1.0);
    }
}
