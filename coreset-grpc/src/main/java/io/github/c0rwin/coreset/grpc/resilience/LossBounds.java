package io.github.c0rwin.coreset.grpc.resilience;

/**
 * Predictions of cor:loss-eps for per-draw messages: with probability at least {@code 1 - delta - eta} the decoded
 * output is as accurate as a loss-free sample of {@code m_eta} draws, so its accuracy is the loss-free one times
 * {@link #inflation} = {@code sqrt(m / m_eta)}.
 */
public final class LossBounds {

    private LossBounds() {
    }

    /**
     * cor:loss-eps (b, c): memoryless channel with loss {@code rho} and {@code b} records per packet ({@code b} divides
     * {@code m}): {@code m_eta = b * ceil(mu_b - sqrt(2 mu_b ln(1/eta)))}, {@code mu_b = (1 - rho) m / b}. Below 1
     * the bound is vacuous and 0 is returned.
     */
    public static long memorylessDraws(long m, double rho, double eta, int b) {
        check(m, rho, eta);
        if (b < 1) {
            throw new IllegalArgumentException("b must be >= 1: " + b);
        }
        double mu = (1.0 - rho) * m / b;
        double packets = Math.ceil(mu - Math.sqrt(2.0 * mu * Math.log(1.0 / eta)));
        return packets < 1 ? 0 : (long) packets * b;
    }

    /**
     * cor:loss-eps (d): stationary Gilbert channel with loss {@code rho} and second eigenvalue {@code lambda}, one
     * record per packet: {@code m_eta = ceil((1 - rho) m - sqrt(V (1/eta - 1)))} with
     * {@code V = m rho (1 - rho) (1 + |lambda|) / (1 - |lambda|)}.
     */
    public static long gilbertDraws(long m, double rho, double lambda, double eta) {
        check(m, rho, eta);
        if (!(lambda > -1.0 && lambda < 1.0)) {
            throw new IllegalArgumentException("lambda must be in (-1, 1): " + lambda);
        }
        double a = Math.abs(lambda);
        double v = m * rho * (1.0 - rho) * (1.0 + a) / (1.0 - a);
        double draws = Math.ceil((1.0 - rho) * m - Math.sqrt(v * (1.0 / eta - 1.0)));
        return draws < 1 ? 0 : (long) draws;
    }

    /** {@code sqrt(m / mEta)}: the factor on the loss-free accuracy; infinite when {@code mEta} is 0. */
    public static double inflation(long m, long mEta) {
        return mEta <= 0 ? Double.POSITIVE_INFINITY : Math.sqrt((double) m / mEta);
    }

    private static void check(long m, double rho, double eta) {
        if (m < 1 || !(rho >= 0.0 && rho < 1.0) || !(eta > 0.0 && eta < 1.0)) {
            throw new IllegalArgumentException("need m >= 1, 0 <= rho < 1, 0 < eta < 1");
        }
    }
}
