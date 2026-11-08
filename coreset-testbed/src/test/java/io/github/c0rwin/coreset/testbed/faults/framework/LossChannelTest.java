package io.github.c0rwin.coreset.testbed.faults.framework;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class LossChannelTest {

    private static int countDelivered(boolean[] lost, int n) {
        int c = 0;
        for (int i = 0; i < n; i++) {
            c += lost[i] ? 0 : 1;
        }
        return c;
    }

    /** Mean and variance of the delivered count from erase() and from delivered(), checked against the given law. */
    private static void checkCountLaw(LossChannel ch, int n, int trials, double mean, double var) {
        SplittableRandom rng = new SplittableRandom(7);
        boolean[] lost = new boolean[n];
        double[] fromErase = new double[trials];
        double[] fromCount = new double[trials];
        for (int t = 0; t < trials; t++) {
            int ret = ch.erase(lost, n, rng);
            assertEquals(countDelivered(lost, n), ret, "erase() returns the delivered count");
            fromErase[t] = ret;
            fromCount[t] = ch.delivered(n, rng);
        }
        for (double[] x : new double[][] {fromErase, fromCount}) {
            double se = Math.sqrt(var / trials);
            assertEquals(mean, FwStats.mean(x), 5 * se, ch.id() + " mean");
            double sd = FwStats.sd(x);
            assertEquals(Math.sqrt(var), sd, 0.05 * Math.sqrt(var), ch.id() + " sd");
        }
    }

    @Test
    void bernoulliCountIsBinomialOnBothSidesOfOneHalf() {
        for (double rho : new double[] {0.1, 0.3, 0.7, 0.9}) {
            int n = 1000;
            checkCountLaw(LossChannel.bernoulli(rho), n, 20000, n * (1 - rho), n * rho * (1 - rho));
        }
    }

    @Test
    void bernoulliLossesAreIndependentAcrossPositions() {
        LossChannel ch = LossChannel.bernoulli(0.3);
        SplittableRandom rng = new SplittableRandom(1);
        int n = 50;
        int trials = 200000;
        boolean[] lost = new boolean[n];
        long first = 0;
        long both = 0;
        long last = 0;
        for (int t = 0; t < trials; t++) {
            ch.erase(lost, n, rng);
            first += lost[0] ? 1 : 0;
            last += lost[n - 1] ? 1 : 0;
            both += lost[10] && lost[11] ? 1 : 0;
        }
        assertEquals(0.3, (double) first / trials, 0.005);
        assertEquals(0.3, (double) last / trials, 0.005);
        assertEquals(0.09, (double) both / trials, 0.004);
    }

    @Test
    void zeroLossDeliversEverything() {
        LossChannel ch = LossChannel.bernoulli(0.0);
        boolean[] lost = new boolean[100];
        lost[5] = true;
        assertEquals(100, ch.erase(lost, 100, new SplittableRandom(3)));
        assertEquals(0, 100 - countDelivered(lost, 100));
        assertEquals(100, ch.delivered(100, new SplittableRandom(3)));
    }

    @Test
    void gilbertMatchesTheStationaryChain() {
        double alpha = 0.01;
        double beta = 0.09;
        LossChannel ch = LossChannel.gilbert(alpha, beta);
        double pi = alpha / (alpha + beta);
        double lambda = 1 - alpha - beta;
        assertEquals(0.1, ch.rate(), 1e-15);
        int n = 2000;
        checkCountLaw(ch, n, 20000, n * (1 - pi), ExpC.exactVariance(pi, lambda, n));
        SplittableRandom rng = new SplittableRandom(11);
        boolean[] lost = new boolean[201];
        int trials = 400000;
        long adjacent = 0;
        long lag1 = 0;
        long at200 = 0;
        for (int t = 0; t < trials; t++) {
            ch.erase(lost, 201, rng);
            adjacent += lost[0] && lost[1] && lost[2] ? 1 : 0;
            lag1 += lost[0] && lost[1] ? 1 : 0;
            at200 += lost[200] ? 1 : 0;
        }
        // P[lost_j = lost_{j+1} = 1] = pi (1 - beta); stationarity at the end of the stream.
        assertEquals(pi * (1 - beta), (double) lag1 / trials, 0.002);
        assertEquals(pi * (1 - beta) * (1 - beta), (double) adjacent / trials, 0.002);
        assertEquals(pi, (double) at200 / trials, 0.002);
    }

    @Test
    void blockLosesTheMiddleDeterministically() {
        LossChannel ch = LossChannel.block(0.1);
        boolean[] a = new boolean[100];
        boolean[] b = new boolean[100];
        assertEquals(90, ch.erase(a, 100, new SplittableRandom(1)));
        assertEquals(90, ch.erase(b, 100, new SplittableRandom(2)));
        assertArrayEquals(a, b);
        for (int i = 0; i < 100; i++) {
            assertEquals(i >= 45 && i < 55, a[i], "packet " + i);
        }
        assertEquals(90, ch.delivered(100, new SplittableRandom(5)));
    }

    @Test
    void patternsAreReproducibleAndValidated() {
        LossChannel ch = LossChannel.gilbert(0.2, 0.3);
        boolean[] a = new boolean[500];
        boolean[] b = new boolean[500];
        ch.erase(a, 500, new SplittableRandom(9));
        ch.erase(b, 500, new SplittableRandom(9));
        assertArrayEquals(a, b);
        assertThrows(IllegalArgumentException.class, () -> LossChannel.bernoulli(1.0));
        assertThrows(IllegalArgumentException.class, () -> LossChannel.gilbert(0.0, 0.5));
        assertThrows(IllegalArgumentException.class, () -> LossChannel.block(-0.1));
        assertEquals(1L, LossChannel.geometric(Math.log1p(-1.0), new SplittableRandom(1)));
    }
}
