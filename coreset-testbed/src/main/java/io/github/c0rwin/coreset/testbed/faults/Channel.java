package io.github.c0rwin.coreset.testbed.faults;

import java.util.Locale;
import java.util.random.RandomGenerator;

/**
 * Packet erasure channels. Both draw exactly one uniform per packet (plus one for the Gilbert-Elliott start state), so
 * Bernoulli patterns at different rates from the same generator state are nested: a packet lost at rate rho is lost
 * at every higher rate.
 */
abstract class Channel {

    final double rho;

    private Channel(double rho) {
        if (!(rho >= 0.0 && rho < 1.0)) {
            throw new IllegalArgumentException("loss rate must be in [0, 1): " + rho);
        }
        this.rho = rho;
    }

    abstract String id();

    /** Writes the loss pattern of {@code len} packets into {@code lost[0..len)}. */
    abstract void erase(boolean[] lost, int len, RandomGenerator rng);

    static Channel bernoulli(double rho) {
        return new Channel(rho) {
            @Override
            String id() {
                return "bernoulli";
            }

            @Override
            void erase(boolean[] lost, int len, RandomGenerator rng) {
                for (int i = 0; i < len; i++) {
                    lost[i] = rng.nextDouble() < rho;
                }
            }
        };
    }

    /**
     * Two-state Gilbert channel (every packet lost in the bad state, none in the good one) with stationary loss rate
     * {@code rho} and mean burst length {@code meanBurst}: {@code p_BG = 1/meanBurst},
     * {@code p_GB = rho p_BG / (1 - rho)}. Needs {@code p_GB <= 1}.
     */
    static Channel gilbertElliott(double rho, double meanBurst) {
        if (!(meanBurst >= 1.0)) {
            throw new IllegalArgumentException("mean burst must be >= 1: " + meanBurst);
        }
        final double pBG = 1.0 / meanBurst;
        final double pGB = rho * pBG / (1.0 - rho);
        if (pGB > 1.0) {
            throw new IllegalArgumentException("rho " + rho + " unreachable with mean burst " + meanBurst);
        }
        return new Channel(rho) {
            @Override
            String id() {
                return String.format(Locale.ROOT, "ge-b%s", trim(meanBurst));
            }

            @Override
            void erase(boolean[] lost, int len, RandomGenerator rng) {
                boolean bad = rng.nextDouble() < rho;
                for (int i = 0; i < len; i++) {
                    lost[i] = bad;
                    double u = rng.nextDouble();
                    bad = bad ? u >= pBG : u < pGB;
                }
            }
        };
    }

    static String trim(double x) {
        return x == Math.rint(x) ? Long.toString((long) x) : Double.toString(x);
    }
}
