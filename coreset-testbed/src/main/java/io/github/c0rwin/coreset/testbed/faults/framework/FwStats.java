package io.github.c0rwin.coreset.testbed.faults.framework;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/** Summary statistics, confidence intervals and two-sample tests used by the framework simulations. */
final class FwStats {

    static final double Z95 = 1.959963984540054;

    private FwStats() {
    }

    /** A point estimate with a two-sided 95% interval. */
    record Ci(double value, double lo, double hi) {

        boolean contains(double x) {
            return lo <= x && x <= hi;
        }
    }

    record Fit(double slope, double intercept, double seSlope) {
    }

    record Test(double stat, double p) {
    }

    static double mean(double[] x) {
        double s = 0.0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    static double sd(double[] x) {
        if (x.length < 2) {
            return 0.0;
        }
        double m = mean(x);
        double s = 0.0;
        for (double v : x) {
            s += (v - m) * (v - m);
        }
        return Math.sqrt(s / (x.length - 1));
    }

    static Ci meanCi(double[] x) {
        double m = mean(x);
        double h = Z95 * sd(x) / Math.sqrt(x.length);
        return new Ci(m, m - h, m + h);
    }

    /** {@code |mean(x) - target| / se(mean)}. */
    static double z(double[] x, double target) {
        return Math.abs(mean(x) - target) / (sd(x) / Math.sqrt(x.length));
    }

    static double rms(double[] x) {
        double s = 0.0;
        for (double v : x) {
            s += v * v;
        }
        return Math.sqrt(s / x.length);
    }

    /** RMS with the normal interval of the mean square, mapped through the square root. */
    static Ci rmsCi(double[] x) {
        double[] sq = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            sq[i] = x[i] * x[i];
        }
        Ci ms = meanCi(sq);
        return new Ci(Math.sqrt(ms.value()), Math.sqrt(Math.max(0.0, ms.lo())), Math.sqrt(ms.hi()));
    }

    static double[] abs(double[] x) {
        double[] a = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            a[i] = Math.abs(x[i]);
        }
        return a;
    }

    /** Type-7 quantile of an ascending array. */
    static double quantileSorted(double[] s, double p) {
        double h = p * (s.length - 1);
        int lo = (int) Math.floor(h);
        int hi = Math.min(s.length - 1, lo + 1);
        return s[lo] + (h - lo) * (s[hi] - s[lo]);
    }

    static double quantile(double[] x, double p) {
        double[] s = x.clone();
        Arrays.sort(s);
        return quantileSorted(s, p);
    }

    /** Distribution-free interval for the p-quantile from the order statistics at ranks {@code np +- z sqrt(np(1-p))}. */
    static Ci quantileCi(double[] x, double p) {
        double[] s = x.clone();
        Arrays.sort(s);
        int[] r = ranks(s.length, p);
        return new Ci(quantileSorted(s, p), s[r[0]], s[r[1]]);
    }

    /** 0-based ranks of the order statistics bounding the 95% interval of the p-quantile of n values. */
    static int[] ranks(long n, double p) {
        double c = n * p;
        double h = Z95 * Math.sqrt(n * p * (1.0 - p));
        int lo = (int) Math.max(0, Math.floor(c - h) - 1);
        int hi = (int) Math.min(n - 1, Math.ceil(c + h));
        return new int[] {lo, hi};
    }

    /** Wilson interval for k successes in n trials. */
    static Ci wilson(long k, long n, double z) {
        double p = (double) k / n;
        double z2 = z * z;
        double den = 1.0 + z2 / n;
        double centre = (p + z2 / (2.0 * n)) / den;
        double h = z * Math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / den;
        return new Ci(p, Math.max(0.0, centre - h), Math.min(1.0, centre + h));
    }

    static Fit ols(double[] x, double[] y) {
        int n = x.length;
        double mx = mean(x);
        double my = mean(y);
        double sxx = 0.0;
        double sxy = 0.0;
        for (int i = 0; i < n; i++) {
            sxx += (x[i] - mx) * (x[i] - mx);
            sxy += (x[i] - mx) * (y[i] - my);
        }
        double b = sxy / sxx;
        double a = my - b * mx;
        double sse = 0.0;
        for (int i = 0; i < n; i++) {
            double e = y[i] - a - b * x[i];
            sse += e * e;
        }
        double se = n > 2 ? Math.sqrt(sse / (n - 2) / sxx) : Double.NaN;
        return new Fit(b, a, se);
    }

    /**
     * Slope of {@code log stat(cell)} against {@code x}, with a percentile bootstrap interval: the replicates of each
     * cell are resampled with replacement, independently across cells.
     */
    static Ci bootstrapLogSlope(double[] x, double[][] cells, ToDoubleFunction<double[]> stat, int resamples,
                                long seed) {
        double[] y = new double[cells.length];
        for (int c = 0; c < cells.length; c++) {
            y[c] = Math.log(stat.applyAsDouble(cells[c]));
        }
        double slope = ols(x, y).slope();
        SplittableRandom rng = new SplittableRandom(seed);
        double[] slopes = new double[resamples];
        double[][] buf = new double[cells.length][];
        for (int c = 0; c < cells.length; c++) {
            buf[c] = new double[cells[c].length];
        }
        for (int b = 0; b < resamples; b++) {
            for (int c = 0; c < cells.length; c++) {
                double[] src = cells[c];
                for (int i = 0; i < src.length; i++) {
                    buf[c][i] = src[rng.nextInt(src.length)];
                }
                y[c] = Math.log(stat.applyAsDouble(buf[c]));
            }
            slopes[b] = ols(x, y).slope();
        }
        Arrays.sort(slopes);
        return new Ci(slope, quantileSorted(slopes, 0.025), quantileSorted(slopes, 0.975));
    }

    /** Two-sample Kolmogorov-Smirnov test, asymptotic p-value with Stephens' small-sample correction. */
    static Test ks(double[] a, double[] b) {
        double[] x = a.clone();
        double[] y = b.clone();
        Arrays.sort(x);
        Arrays.sort(y);
        int i = 0;
        int j = 0;
        double d = 0.0;
        while (i < x.length && j < y.length) {
            double v = Math.min(x[i], y[j]);
            while (i < x.length && x[i] == v) {
                i++;
            }
            while (j < y.length && y[j] == v) {
                j++;
            }
            d = Math.max(d, Math.abs((double) i / x.length - (double) j / y.length));
        }
        double en = Math.sqrt((double) x.length * y.length / (x.length + y.length));
        return new Test(d, kolmogorovQ((en + 0.12 + 0.11 / en) * d));
    }

    /** {@code P[K > lambda]} for the Kolmogorov distribution. */
    static double kolmogorovQ(double lambda) {
        if (lambda < 0.2) {
            return 1.0;
        }
        double sum = 0.0;
        double sign = 1.0;
        for (int k = 1; k <= 200; k++) {
            double term = sign * 2.0 * Math.exp(-2.0 * k * k * lambda * lambda);
            sum += term;
            if (Math.abs(term) < 1e-16) {
                break;
            }
            sign = -sign;
        }
        return Math.min(1.0, Math.max(0.0, sum));
    }

    /**
     * Two-sample Anderson-Darling test: the midrank statistic {@code A2_akN} of Scholz and Stephens (1987), which
     * handles ties, with a permutation p-value {@code (1 + #{A2* >= A2}) / (1 + perms)}.
     */
    static Test ad(double[] a, double[] b, int perms, long seed) {
        int n = a.length + b.length;
        double[] all = new double[n];
        System.arraycopy(a, 0, all, 0, a.length);
        System.arraycopy(b, 0, all, a.length, b.length);
        int[] order = sortIndex(all);
        // Groups of tied values in sorted order: start offsets and sizes.
        int[] groupOf = new int[n];
        int groups = 0;
        for (int r = 0; r < n; r++) {
            if (r > 0 && all[order[r]] != all[order[r - 1]]) {
                groups++;
            }
            groupOf[r] = groups;
        }
        groups++;
        int[] l = new int[groups];
        for (int r = 0; r < n; r++) {
            l[groupOf[r]]++;
        }
        boolean[] first = new boolean[n];
        for (int r = 0; r < n; r++) {
            first[r] = order[r] < a.length;
        }
        int[] f1 = new int[groups];
        double obs = adMidrank(first, groupOf, l, f1, a.length, b.length);
        SplittableRandom rng = new SplittableRandom(seed);
        int ge = 0;
        for (int p = 0; p < perms; p++) {
            for (int r = n - 1; r > 0; r--) {
                int s = rng.nextInt(r + 1);
                boolean t = first[r];
                first[r] = first[s];
                first[s] = t;
            }
            ge += adMidrank(first, groupOf, l, f1, a.length, b.length) >= obs - 1e-12 * Math.abs(obs) ? 1 : 0;
        }
        return new Test(obs, (1.0 + ge) / (1.0 + perms));
    }

    private static double adMidrank(boolean[] first, int[] groupOf, int[] l, int[] f1, int n1, int n2) {
        Arrays.fill(f1, 0);
        for (int r = 0; r < first.length; r++) {
            if (first[r]) {
                f1[groupOf[r]]++;
            }
        }
        double nn = n1 + n2;
        double s1 = 0.0;
        double s2 = 0.0;
        double b = 0.0;
        double m1 = 0.0;
        double m2 = 0.0;
        for (int g = 0; g < l.length; g++) {
            double lj = l[g];
            double fa = f1[g];
            double fb = lj - fa;
            b += lj;
            m1 += fa;
            m2 += fb;
            double ba = b - lj / 2.0;
            double den = ba * (nn - ba) - nn * lj / 4.0;
            if (den <= 0.0) {
                continue;
            }
            double ma1 = m1 - fa / 2.0;
            double ma2 = m2 - fb / 2.0;
            s1 += lj * Math.pow(nn * ma1 - n1 * ba, 2) / den;
            s2 += lj * Math.pow(nn * ma2 - n2 * ba, 2) / den;
        }
        return (nn - 1.0) / (nn * nn) * (s1 / n1 + s2 / n2);
    }

    private static int[] sortIndex(double[] v) {
        Integer[] idx = new Integer[v.length];
        for (int i = 0; i < v.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (x, y) -> Double.compare(v[x], v[y]));
        int[] out = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    /** Inverse of the standard normal distribution function (Acklam's rational approximation, rel. error 1.2e-9). */
    static double invNormal(double p) {
        if (!(p > 0.0 && p < 1.0)) {
            throw new IllegalArgumentException("p must be in (0, 1): " + p);
        }
        final double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
            1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        final double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
            6.680131188771972e+01, -1.328068155288572e+01};
        final double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
            -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        final double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
            3.754408661907416e+00};
        final double low = 0.02425;
        if (p < low) {
            double q = Math.sqrt(-2.0 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
        }
        if (p > 1.0 - low) {
            return -invNormal(1.0 - p);
        }
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
}
