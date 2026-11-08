package io.github.c0rwin.coreset.testbed.faults.framework;

import static io.github.c0rwin.coreset.testbed.faults.framework.ExpA.ci;
import static io.github.c0rwin.coreset.testbed.faults.framework.Lab.f;

import java.io.IOException;
import java.io.Writer;
import java.util.List;

/**
 * E. The master theorem beyond k-means: RC on a Bernoulli channel for k-median (z = 1) and k-means (z = 2), with the
 * testbed's {@code dist^z} cost. At fixed m the error should scale as eps(0) / sqrt(1 - rho).
 */
final class ExpE {

    private ExpE() {
    }

    static void run(Lab lab) throws IOException {
        Instance inst = lab.instance();
        double[] zs = {1.0, 2.0};
        double[] rhos = lab.quick() ? new double[] {0.0, 0.5, 0.8} : new double[] {0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8,
            0.9};
        final int m = 2000;
        int reps = lab.pick(1000, 30);
        int cells = zs.length * rhos.length;
        CostSpace[] spaces = {inst.full(1.0), inst.full(2.0)};
        List<double[]> res = lab.parallel(cells * reps, idx -> {
            int cell = idx / reps;
            int zi = cell / rhos.length;
            int ri = cell % rhos.length;
            int r = idx % reps;
            CostSpace sp = spaces[zi];
            Buffers buf = Buffers.get();
            int[] draws = buf.ints(m);
            boolean[] lost = buf.flags(m);
            sp.draws(draws, m, Seeds.rng(lab.seed, "E-sample", cell, r));
            LossChannel.bernoulli(rhos[ri]).erase(lost, m, Seeds.rng(lab.seed, "E-channel", cell, r));
            Tally t = new Tally(sp);
            for (int s = 0; s < m; s++) {
                t.add(draws[s], !lost[s]);
            }
            return new double[] {sp.rel(t.rc(0), 0), t.rcMaxAbs(), t.received()};
        });
        try (Writer w = lab.raw("E-kmedian", "z,rho,rep,rc_rel_q0,rc_maxabs,received")) {
            for (int idx = 0; idx < res.size(); idx++) {
                double[] v = res.get(idx);
                int cell = idx / reps;
                w.write(f("%.0f,%.2f,%d,%.17g,%.17g,%d%n", zs[cell / rhos.length], rhos[cell % rhos.length],
                        idx % reps, v[0], v[1], (long) v[2]));
            }
        }
        Table tab = lab.table("E-kmedian", "z", "rho", "mean M'", "RMS err Q0 [95% CI]", "RMS max_Q err [95% CI]",
                "RMS Q0 x sqrt(1-rho) / RMS Q0 at rho=0", "RMS max_Q x sqrt(1-rho) / at rho=0");
        double[] x = new double[rhos.length];
        for (int ri = 0; ri < rhos.length; ri++) {
            x[ri] = Math.log(1.0 / (1.0 - rhos[ri]));
        }
        boolean ok = true;
        StringBuilder meas = new StringBuilder();
        for (int zi = 0; zi < zs.length; zi++) {
            double[][] c0 = new double[rhos.length][];
            double[][] cm = new double[rhos.length][];
            for (int ri = 0; ri < rhos.length; ri++) {
                int cell = zi * rhos.length + ri;
                c0[ri] = ExpA.col(res, cell, reps, 0);
                cm[ri] = ExpA.col(res, cell, reps, 1);
            }
            double base0 = FwStats.rms(c0[0]);
            double baseM = FwStats.rms(cm[0]);
            for (int ri = 0; ri < rhos.length; ri++) {
                double sq = Math.sqrt(1.0 - rhos[ri]);
                tab.row(f("%.0f", zs[zi]), f("%.1f", rhos[ri]),
                        f("%.1f", FwStats.mean(ExpA.col(res, zi * rhos.length + ri, reps, 2))),
                        ci(FwStats.rmsCi(c0[ri]), "%.4f"), ci(FwStats.rmsCi(cm[ri]), "%.4f"),
                        f("%.3f", FwStats.rms(c0[ri]) * sq / base0), f("%.3f", FwStats.rms(cm[ri]) * sq / baseM));
            }
            FwStats.Ci s0 = FwStats.bootstrapLogSlope(x, c0, FwStats::rms, 2000, Seeds.of(lab.seed, "E-boot", zi, 0));
            FwStats.Ci sm = FwStats.bootstrapLogSlope(x, cm, FwStats::rms, 2000, Seeds.of(lab.seed, "E-boot", zi, 1));
            ok &= Math.abs(s0.value() - 0.5) <= 0.05 && Math.abs(sm.value() - 0.5) <= 0.05;
            meas.append(f("z=%.0f: Q0 %s, max_Q %s; ", zs[zi], ci(s0, "%.3f"), ci(sm, "%.3f")));
        }
        lab.section("E. k-median (z = 1) and k-means (z = 2)",
                f("Mixture, two-term sensitivity proposal around a D^z seeding, m = %d draws, B = 1, Bernoulli loss, RC "
                        + "decoder, costs dist^z via the testbed's Problem objective, %d repetitions per cell. Slopes of "
                        + "log RMS on log 1/(1-rho) (predicted 0.5): %s", m, reps, meas.toString().trim()), tab);
        lab.verdict("E k-median scaling", "the master theorem holds for k-median: error ~ eps/sqrt(1-rho) "
                + "(log-log slope 0.5 +- 0.05), as for k-means", meas.toString().trim(), ok);
    }
}
