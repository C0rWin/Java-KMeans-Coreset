package io.github.c0rwin.coreset.testbed.faults;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/** Summary statistics and least-squares fits used by the verdicts. */
final class Stats {

    private Stats() {
    }

    static double mean(double[] x) {
        double s = 0.0;
        for (double v : x) {
            s += v;
        }
        return x.length == 0 ? Double.NaN : s / x.length;
    }

    static double sd(double[] x) {
        if (x.length < 2) {
            return Double.NaN;
        }
        double m = mean(x);
        double s = 0.0;
        for (double v : x) {
            s += (v - m) * (v - m);
        }
        return Math.sqrt(s / (x.length - 1));
    }

    static double rms(double[] x) {
        double s = 0.0;
        for (double v : x) {
            s += v * v;
        }
        return x.length == 0 ? Double.NaN : Math.sqrt(s / x.length);
    }

    /** Linear-interpolated quantile of unsorted data. */
    static double quantile(double[] x, double p) {
        if (x.length == 0) {
            return Double.NaN;
        }
        double[] s = x.clone();
        Arrays.sort(s);
        double h = p * (s.length - 1);
        int lo = (int) Math.floor(h);
        int hi = Math.min(s.length - 1, lo + 1);
        return s[lo] + (h - lo) * (s[hi] - s[lo]);
    }

    static double min(double[] x) {
        double m = Double.POSITIVE_INFINITY;
        for (double v : x) {
            m = Math.min(m, v);
        }
        return m;
    }

    static double sum(double[] x) {
        double s = 0.0;
        for (double v : x) {
            s += v;
        }
        return s;
    }

    static double max(double[] x) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            m = Math.max(m, v);
        }
        return m;
    }

    static double fraction(double[] x, double threshold) {
        int c = 0;
        for (double v : x) {
            c += v > threshold ? 1 : 0;
        }
        return x.length == 0 ? Double.NaN : (double) c / x.length;
    }

    /** {@code y = a + b x} with standard errors; {@code ci95} is the half-width of the slope's 95% interval. */
    record Fit(double a, double b, double seA, double seB, double r2, int n) {
        double ci95() {
            return t975(n - 2) * seB;
        }
    }

    static Fit ols(double[] x, double[] y) {
        int n = x.length;
        double mx = mean(x);
        double my = mean(y);
        double sxx = 0.0;
        double sxy = 0.0;
        double syy = 0.0;
        for (int i = 0; i < n; i++) {
            sxx += (x[i] - mx) * (x[i] - mx);
            sxy += (x[i] - mx) * (y[i] - my);
            syy += (y[i] - my) * (y[i] - my);
        }
        double b = sxy / sxx;
        double a = my - b * mx;
        double sse = 0.0;
        for (int i = 0; i < n; i++) {
            double e = y[i] - a - b * x[i];
            sse += e * e;
        }
        double s2 = n > 2 ? sse / (n - 2) : Double.NaN;
        double seB = Math.sqrt(s2 / sxx);
        double seA = Math.sqrt(s2 * (1.0 / n + mx * mx / sxx));
        return new Fit(a, b, seA, seB, syy > 0.0 ? 1.0 - sse / syy : 1.0, n);
    }

    /** {@code y = c x}: returns {c, R^2 about zero}. */
    static double[] throughOrigin(double[] x, double[] y) {
        double sxy = 0.0;
        double sxx = 0.0;
        double syy = 0.0;
        for (int i = 0; i < x.length; i++) {
            sxy += x[i] * y[i];
            sxx += x[i] * x[i];
            syy += y[i] * y[i];
        }
        double c = sxy / sxx;
        double sse = 0.0;
        for (int i = 0; i < x.length; i++) {
            sse += (y[i] - c * x[i]) * (y[i] - c * x[i]);
        }
        return new double[] {c, 1.0 - sse / syy};
    }

    /** Two-sided 95% Student t quantile. */
    static double t975(int df) {
        double[] table = {Double.NaN, 12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201,
            2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
            2.052, 2.048, 2.045, 2.042};
        if (df < 1) {
            return Double.NaN;
        }
        return df < table.length ? table[df] : 1.96 + 2.4 / df;
    }

    /**
     * Percentile bootstrap over {@code groups} independent units (seeds): {@code stat} gets the resampled unit indices
     * and returns the statistic; result {lo, hi} at 2.5% and 97.5%.
     */
    static double[] bootstrap(int groups, ToDoubleFunction<int[]> stat, int reps, long seed) {
        SplittableRandom rng = new SplittableRandom(seed);
        double[] out = new double[reps];
        int[] idx = new int[groups];
        for (int b = 0; b < reps; b++) {
            for (int i = 0; i < groups; i++) {
                idx[i] = rng.nextInt(groups);
            }
            out[b] = stat.applyAsDouble(idx);
        }
        return new double[] {quantile(out, 0.025), quantile(out, 0.975)};
    }
}
