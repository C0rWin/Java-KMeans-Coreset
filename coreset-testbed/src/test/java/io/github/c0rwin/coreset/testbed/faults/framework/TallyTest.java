package io.github.c0rwin.coreset.testbed.faults.framework;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class TallyTest {

    /** Rows: f = (1, 2), (3, 0), (0, 5); w = (1, 2, 1); s = (1, 1, 2), so q = (1/4, 1/4, 1/2); classes 0, 0, 1. */
    private static CostSpace tiny() {
        double[] table = {1, 2, 3, 0, 0, 5};
        return new CostSpace(CostSpace.identity(3), new double[] {1, 2, 1}, new double[] {1, 1, 2}, table, 2,
                new int[] {0, 0, 1}, 2);
    }

    @Test
    void spaceStoresScaleFreeCosts() {
        CostSpace sp = tiny();
        assertEquals(7.0, sp.cost[0], 1e-12);
        assertEquals(7.0, sp.cost[1], 1e-12);
        assertEquals(0.5, sp.q[2], 1e-12);
        // g = (w / q) f
        assertEquals(24.0, sp.g[2], 1e-12);
        assertEquals(10.0, sp.g[5], 1e-12);
    }

    @Test
    void decodersMatchTheirClosedForms() {
        CostSpace sp = tiny();
        Tally t = new Tally(sp);
        int[] draws = {0, 1, 2, 2, 1};
        boolean[] ok = {true, false, true, false, true};
        for (int s = 0; s < draws.length; s++) {
            t.add(draws[s], ok[s]);
        }
        assertEquals(5, t.sent());
        assertEquals(3, t.received());
        assertEquals(2, t.received(0));
        // Received sums: class 0 = g0 + g1 = (28, 8), class 1 = g2 = (0, 10).
        assertEquals(28.0 / 3, t.rc(0), 1e-12);
        assertEquals(6.0, t.rc(1), 1e-12);
        assertEquals(28.0 / 2.5, t.ht(0, 0.5), 1e-12);
        assertEquals(7.0, t.known(0, new double[] {0.8, 0.4}), 1e-12);
        assertEquals(7.0, t.known(1, new double[] {0.8, 0.4}), 1e-12);
        // Per class: n = (3, 2), M' = (2, 1).
        assertEquals(28.0 * 1.5 / 5, t.perClass(0), 1e-12);
        assertEquals((8.0 * 1.5 + 10.0 * 2) / 5, t.perClass(1), 1e-12);
        assertEquals(1.0 / 3, t.rcMaxAbs(), 1e-12);
    }

    @Test
    void emptyAndLossFreeMessages() {
        CostSpace sp = tiny();
        Tally t = new Tally(sp);
        t.add(0, false);
        assertEquals(0.0, t.rc(0));
        assertEquals(0.0, t.perClass(1));
        t.reset();
        int[] draws = {0, 2, 1, 1};
        for (int d : draws) {
            t.add(d, true);
        }
        double plain = (4.0 + 0.0 + 24.0 + 24.0) / 4;
        assertEquals(plain, t.rc(0), 1e-12);
        assertEquals(plain, t.ht(0, 1.0), 1e-12);
        assertEquals(plain, t.perClass(0), 1e-12);
    }

    @Test
    void contentCorrelatedLossBiasesRcButNotPerClass() {
        // RC-worst instance: E[RC] = 0.9 / (60.9/101) cost; per-class and known-p decoders are unbiased.
        CostSpace sp = ExpB.twoPoint(100, 1);
        double[] p = {0.6, 0.9};
        int trials = 4000;
        double[] rc = new double[trials];
        double[] per = new double[trials];
        double[] known = new double[trials];
        for (int r = 0; r < trials; r++) {
            double[] v = ExpB.trial(sp, p, 2000, new SplittableRandom(r), new SplittableRandom(1_000_000L + r));
            rc[r] = v[0];
            known[r] = v[2];
            per[r] = v[3];
        }
        double pred = 0.9 * 101 / 60.9 - 1;
        assertEquals(pred, FwStats.mean(rc), 4 * FwStats.sd(rc) / Math.sqrt(trials));
        assertEquals(0.0, FwStats.mean(known), 4 * FwStats.sd(known) / Math.sqrt(trials));
        assertEquals(0.0, FwStats.mean(per), 4 * FwStats.sd(per) / Math.sqrt(trials));
    }

    @Test
    void packingHelpers() {
        CostSpace sp = tiny();
        int[] a = {2, 0, 1, 2, 0};
        Buffers.get().sortByClass(a, 5, sp);
        assertEquals("[0, 0, 1, 2, 2]", java.util.Arrays.toString(a));
        int[] b = {1, 2, 3, 4, 5, 6};
        Buffers.shuffle(b, 6, new SplittableRandom(4));
        assertEquals(21, java.util.Arrays.stream(b).sum());
    }
}
