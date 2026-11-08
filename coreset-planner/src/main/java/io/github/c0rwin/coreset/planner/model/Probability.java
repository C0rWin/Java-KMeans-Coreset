package io.github.c0rwin.coreset.planner.model;

/** Distribution helpers used by the resilience model; all exact or one-sided conservative as documented. */
public final class Probability {

    private Probability() {
    }

    /** Standard normal quantile, Acklam's rational approximation (relative error below 1.2e-9). */
    public static double normalQuantile(double p) {
        if (!(p > 0.0 && p < 1.0)) {
            throw new IllegalArgumentException("p must be in (0,1): " + p);
        }
        final double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00};
        final double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01};
        final double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
                -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00};
        final double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
                3.754408661907416e+00};
        double q;
        if (p < 0.02425) {
            q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        if (p > 1 - 0.02425) {
            return -normalQuantile(1 - p);
        }
        q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }

    /** Smallest l with {@code P[Bin(n, p) > l] <= tail}. */
    public static int binomialUpperQuantile(int n, double p, double tail) {
        if (p <= 0.0 || n == 0) {
            return 0;
        }
        if (p >= 1.0) {
            return n;
        }
        double logP = Math.log(p);
        double logQ = Math.log1p(-p);
        double cdf = 0.0;
        double logCoef = 0.0; // log C(n, l)
        for (int l = 0; l <= n; l++) {
            if (l > 0) {
                logCoef += Math.log(n - l + 1) - Math.log(l);
            }
            cdf += Math.exp(logCoef + l * logP + (n - l) * logQ);
            if (1.0 - cdf <= tail) {
                return l;
            }
        }
        return n;
    }

    /** Largest l with {@code P[Bin(n, p) < l] <= tail}. */
    public static int binomialLowerQuantile(int n, double p, double tail) {
        return n - binomialUpperQuantile(n, 1.0 - p, tail);
    }

    /**
     * Lower {@code eta}-quantile of the number of delivered units among {@code units} sent over a stationary
     * two-state chain with loss rate {@code loss} and second eigenvalue {@code lambda}: cor:loss-eps(b) (Chernoff)
     * when {@code lambda == 0}, cor:loss-eps(d) (Chebyshev with the Gilbert-Elliott variance bound) otherwise.
     * Returns 0 when the bound is vacuous.
     */
    public static long deliveredLowerBound(long units, double loss, double lambda, double eta) {
        if (loss <= 0.0) {
            return units;
        }
        double mu = (1.0 - loss) * units;
        double dev;
        if (lambda == 0.0) {
            dev = Math.sqrt(2.0 * mu * Math.log(1.0 / eta));
        } else {
            double l = Math.abs(lambda);
            double rt = Math.min(loss, 0.5); // thm:P-sound: rho~ = min(rho, 1/2) keeps the margin monotone in rho
            double v = units * rt * (1.0 - rt) * (1.0 + l) / (1.0 - l);
            dev = Math.sqrt(v * (1.0 / eta - 1.0));
        }
        return Math.max(0L, (long) Math.ceil(mu - dev));
    }

    /**
     * Probability that a block of {@code data + parity} packets loses more than {@code parity} of them on a
     * stationary Gilbert channel (loss probability 1 in the bad state, 0 in the good one), by dynamic programming over
     * (losses so far, state). Memoryless erasures are the case {@code pGB = rho, pBG = 1 - rho}.
     */
    public static double blockFailure(int data, int parity, double pGoodToBad, double pBadToGood) {
        int n = data + parity;
        double s = pGoodToBad + pBadToGood;
        if (pGoodToBad <= 0.0) {
            return 0.0;
        }
        double piBad = pGoodToBad / s;
        // good[l], bad[l]: probability of l losses so far with the last packet in that state
        double[] good = new double[n + 2];
        double[] bad = new double[n + 2];
        good[0] = 1.0 - piBad;
        bad[1] = piBad;
        for (int t = 1; t < n; t++) {
            double[] g2 = new double[n + 2];
            double[] b2 = new double[n + 2];
            for (int l = 0; l <= t; l++) {
                g2[l] += good[l] * (1.0 - pGoodToBad) + bad[l] * pBadToGood;
                b2[l + 1] += good[l] * pGoodToBad + bad[l] * (1.0 - pBadToGood);
            }
            good = g2;
            bad = b2;
        }
        double fail = 0.0;
        for (int l = parity + 1; l <= n; l++) {
            fail += good[l] + bad[l];
        }
        return Math.min(1.0, Math.max(0.0, fail));
    }

    /** {@code E[K^2 / ceil(K/beta) - K]} for {@code K ~ Poisson(lambda)}; equals lambda^2 when beta is infinite. */
    public static double splitExcess(double lambda, int beta) {
        if (lambda <= 0.0) {
            return 0.0;
        }
        if (lambda < 1e-4) {
            return beta >= 2 ? lambda * lambda : 0.0; // K = 2 dominates: P = lambda^2/2, excess 2 per such record
        }
        if (lambda > 500.0) {
            return (beta - 1.0) * lambda; // K >> beta: K^2/ceil(K/beta) ~ beta K
        }
        int max = (int) Math.ceil(lambda + 12.0 * Math.sqrt(lambda) + 30.0);
        double p = Math.exp(-lambda);
        double acc = 0.0;
        for (int kk = 1; kk <= max; kk++) {
            p *= lambda / kk;
            int j = (kk + beta - 1) / beta;
            acc += p * ((double) kk * kk / j - kk);
        }
        return acc;
    }
}
