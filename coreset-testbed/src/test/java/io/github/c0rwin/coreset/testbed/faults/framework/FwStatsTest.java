package io.github.c0rwin.coreset.testbed.faults.framework;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class FwStatsTest {

    private static double[] normals(int n, double shift, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = shift + Math.sqrt(-2 * Math.log(1 - r.nextDouble())) * Math.cos(2 * Math.PI * r.nextDouble());
        }
        return x;
    }

    @Test
    void inverseNormal() {
        assertEquals(0.0, FwStats.invNormal(0.5), 1e-9);
        assertEquals(1.959963985, FwStats.invNormal(0.975), 1e-8);
        assertEquals(2.326347874, FwStats.invNormal(0.99), 1e-8);
        assertEquals(3.719016485, FwStats.invNormal(0.9999), 1e-7);
        assertEquals(-3.719016485, FwStats.invNormal(1e-4), 1e-7);
    }

    @Test
    void kolmogorovSmirnov() {
        double[] a = {1, 2, 3, 4};
        double[] b = {1, 2, 3, 4};
        assertEquals(0.0, FwStats.ks(a, b).stat());
        assertEquals(1.0, FwStats.ks(a, b).p());
        // Disjoint supports: D = 1.
        assertEquals(1.0, FwStats.ks(new double[] {1, 2}, new double[] {3, 4, 5}).stat(), 1e-15);
        // Ties across samples are stepped over together: F_a(2) = 1, F_b(2) = 2/3.
        assertEquals(1.0 / 3, FwStats.ks(new double[] {1, 2}, new double[] {1, 2, 3}).stat(), 1e-15);
        // Q_KS(1.36) is about 0.0505 (5% critical value 1.358).
        assertEquals(0.05, FwStats.kolmogorovQ(1.3581), 5e-4);
        double[] x = normals(2000, 0, 1);
        double[] y = normals(2000, 0, 2);
        double[] z = normals(2000, 0.3, 3);
        assertTrue(FwStats.ks(x, y).p() > 0.01);
        assertTrue(FwStats.ks(x, z).p() < 1e-10);
    }

    @Test
    void andersonDarlingPermutation() {
        double[] x = normals(500, 0, 4);
        double[] y = normals(500, 0, 5);
        double[] z = normals(500, 0.4, 6);
        FwStats.Test same = FwStats.ad(x, y, 199, 1);
        FwStats.Test diff = FwStats.ad(x, z, 199, 1);
        assertTrue(same.p() > 0.01, "same law: p = " + same.p());
        assertEquals(1.0 / 200, diff.p(), 1e-15);
        assertTrue(diff.stat() > same.stat());
        // Ties: both samples constant and equal, so every permutation gives the same statistic.
        double[] c = {2, 2, 2};
        assertEquals(1.0, FwStats.ad(c, c.clone(), 49, 2).p());
        // Reproducible for a fixed seed.
        assertEquals(same.p(), FwStats.ad(x, y, 199, 1).p());
    }

    @Test
    void intervals() {
        FwStats.Ci w = FwStats.wilson(0, 100, FwStats.Z95);
        assertEquals(0.0, w.lo(), 1e-12);
        assertTrue(w.hi() > 0.03 && w.hi() < 0.04);
        FwStats.Ci m = FwStats.meanCi(new double[] {1, 2, 3, 4, 5});
        assertEquals(3.0, m.value());
        assertTrue(m.contains(3.0) && !m.contains(5.0));
        int[] r = FwStats.ranks(1_000_000, 0.01);
        assertTrue(r[0] < 10_000 && r[1] > 10_000 && r[1] - r[0] < 500);
        assertEquals(2.5, FwStats.quantile(new double[] {4, 1, 3, 2}, 0.5), 1e-15);
        assertEquals(Math.sqrt(2.5), FwStats.rms(new double[] {1, -2}), 1e-15);
        assertEquals(0.0, FwStats.z(new double[] {1, 2, 3}, 2.0), 1e-15);
    }

    @Test
    void bootstrapSlopeRecoversAPowerLaw() {
        double[] x = new double[4];
        double[][] cells = new double[4][];
        SplittableRandom r = new SplittableRandom(8);
        for (int c = 0; c < 4; c++) {
            double m = Math.pow(10, c + 2);
            x[c] = Math.log(m);
            cells[c] = new double[400];
            for (int i = 0; i < 400; i++) {
                cells[c][i] = (r.nextDouble() - 0.5) / Math.sqrt(m);
            }
        }
        FwStats.Ci s = FwStats.bootstrapLogSlope(x, cells, FwStats::rms, 500, 3);
        assertEquals(-0.5, s.value(), 0.03);
        assertTrue(s.lo() < s.value() && s.value() < s.hi());
        FwStats.Fit fit = FwStats.ols(new double[] {0, 1, 2}, new double[] {1, 3, 5});
        assertEquals(2.0, fit.slope(), 1e-12);
        assertEquals(1.0, fit.intercept(), 1e-12);
    }
}
