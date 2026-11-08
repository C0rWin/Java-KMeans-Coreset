package io.github.c0rwin.coreset.testbed.faults.framework;

import static io.github.c0rwin.coreset.testbed.faults.framework.ExpA.ci;
import static io.github.c0rwin.coreset.testbed.faults.framework.Lab.f;

import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.SplittableRandom;

/**
 * C. Bursty loss: the lower tail of M' on Gilbert's channel against the normal, Markov and Cantelli bounds, and the RC
 * error law given M' under Bernoulli, Gilbert and block loss with random and sorted packing.
 */
final class ExpC {

    static final double ALPHA = 0.01;
    static final double BETA = 0.09;
    static final int M = 10_000;
    static final int TARGET = 9_000;

    private ExpC() {
    }

    static void run(Lab lab) throws IOException {
        deficits(lab);
        conditionalLaw(lab);
    }

    static double thetaMarkov(double rho, double lambda, int b, int m, double eta) {
        double l0 = Math.max(0.0, lambda);
        double kappa = (1.0 + l0) / (1.0 - l0);
        return (1.0 + Math.sqrt(kappa)) / (1.0 - rho) * Math.sqrt(b * Math.log(2.0 / eta) / (2.0 * m));
    }

    /** Exact variance of the number of bad slots among n for the stationary two-state chain. */
    static double exactVariance(double pi, double lambda, int n) {
        return n * pi * (1.0 - pi) * ((1.0 + lambda) / (1.0 - lambda)
                - 2.0 * lambda * (1.0 - Math.pow(lambda, n)) / (n * (1.0 - lambda) * (1.0 - lambda)));
    }

    private static void deficits(Lab lab) throws IOException {
        long trials = lab.pick(10_000_000L, 40_000L);
        int chunks = 400;
        long per = trials / chunks;
        long n = per * chunks;
        LossChannel ge = LossChannel.gilbert(ALPHA, BETA);
        List<long[]> parts = lab.parallel(chunks, c -> {
            SplittableRandom rng = Seeds.rng(lab.seed, "C1-channel", c, 0);
            long[] h = new long[M + 1];
            for (long t = 0; t < per; t++) {
                h[ge.delivered(M, rng)]++;
            }
            return h;
        });
        long[] hist = new long[M + 1];
        for (long[] h : parts) {
            for (int v = 0; v <= M; v++) {
                hist[v] += h[v];
            }
        }
        try (Writer w = lab.raw("C1-delivered-histogram", "delivered,count")) {
            for (int v = 0; v <= M; v++) {
                if (hist[v] > 0) {
                    w.write(v + "," + hist[v] + "\n");
                }
            }
        }
        double rho = ge.rate();
        double lambda = 1.0 - ALPHA - BETA;
        double base = (1.0 - rho) * M;
        double sdAsym = Math.sqrt(M * rho * (1.0 - rho) * (1.0 + lambda) / (1.0 - lambda));
        double sdRel = sdAsym / base;
        double mean = 0.0;
        double sq = 0.0;
        for (int v = 0; v <= M; v++) {
            mean += (double) v * hist[v];
            sq += (double) v * v * hist[v];
        }
        mean /= n;
        double sdEmp = Math.sqrt(sq / n - mean * mean);
        double[] etas = lab.quick() ? new double[] {1e-1, 1e-2} : new double[] {1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
        Table tab = lab.table("C1-deficit-quantiles", "eta", "empirical deficit [95% CI]", "normal approx.",
                "empirical / normal", "Markov bound theta_MM", "empirical / theta_MM", "Cantelli bound",
                "samples beyond the quantile");
        double[][] emp = new double[etas.length][];
        double[] lx = new double[etas.length];
        double[] lyEmp = new double[etas.length];
        double[] lyNorm = new double[etas.length];
        for (int e = 0; e < etas.length; e++) {
            double eta = etas[e];
            int[] r = FwStats.ranks(n, eta);
            int point = valueAtRank(hist, (long) Math.ceil(eta * n) - 1);
            // Lower M' rank = larger deficit.
            double dPoint = (base - point) / base;
            double dHi = (base - valueAtRank(hist, r[0])) / base;
            double dLo = (base - valueAtRank(hist, r[1])) / base;
            emp[e] = new double[] {dPoint, dLo, dHi};
            double normal = FwStats.invNormal(1.0 - eta) * sdRel;
            double theta = thetaMarkov(rho, lambda, 1, M, eta);
            double cantelli = sdRel * Math.sqrt((1.0 - eta) / eta);
            tab.row(f("%.0e", eta), f("%.4f [%.4f, %.4f]", dPoint, dLo, dHi), f("%.4f", normal),
                    f("%.3f", dPoint / normal), f("%.4f", theta), f("%.3f", dPoint / theta), f("%.4f", cantelli),
                    (long) Math.ceil(eta * n));
            lx[e] = Math.log(Math.log(1.0 / eta));
            lyEmp[e] = Math.log(dPoint);
            lyNorm[e] = Math.log(normal);
        }
        double slopeEmp = FwStats.ols(lx, lyEmp).slope();
        double slopeNorm = FwStats.ols(lx, lyNorm).slope();
        lab.section("C1. Lower tail of M' on Gilbert's channel",
                f("alpha = %.2f, beta_GE = %.2f: rho = %.2f, lambda = %.2f, kappa = %.0f, mean burst %.1f packets; "
                        + "m = %d, B = 1, %d channel realizations from the stationary law. Deficit = ((1-rho)m - M') / "
                        + "((1-rho)m) at quantile 1-eta. Mean M' %.2f (predicted %.0f); sd %.2f (asymptotic %.2f, exact "
                        + "finite-m %.2f). Normal approximation z_{1-eta} sd/((1-rho)m); Cantelli sd sqrt((1-eta)/eta)"
                        + "/((1-rho)m).", ALPHA, BETA, rho, lambda, (1 + lambda) / (1 - lambda), 1.0 / BETA, M, n,
                        mean, base, sdEmp, sdAsym, Math.sqrt(exactVariance(rho, lambda, M)))
                + f(" Log-log slope of the deficit quantile on ln(1/eta) over eta = %.0e..%.0e: %.3f empirical, %.3f "
                        + "for the normal quantile (sqrt(ln 1/eta) is 0.5 only asymptotically).", etas[0],
                        etas[etas.length - 1], slopeEmp, slopeNorm), tab);
        int i2 = 1;
        int i4 = etas.length > 3 ? 3 : etas.length - 1;
        double n2 = FwStats.invNormal(1.0 - etas[i2]) * sdRel;
        double n4 = FwStats.invNormal(1.0 - etas[i4]) * sdRel;
        double dev2 = emp[i2][0] / n2 - 1.0;
        double dev4 = emp[i4][0] / n4 - 1.0;
        lab.verdict("C1 normal approximation", "deficit quantile ~0.034 (eta=1e-2) and ~0.054 (eta=1e-4), within 15%",
                f("%.4f [%.4f, %.4f] (%+.1f%%) and %.4f [%.4f, %.4f] (%+.1f%%) vs %.4f and %.4f", emp[i2][0],
                        emp[i2][1], emp[i2][2], 100 * dev2, emp[i4][0], emp[i4][1], emp[i4][2], 100 * dev4, n2, n4),
                Math.abs(dev2) <= 0.15 && Math.abs(dev4) <= 0.15);
        double t2 = thetaMarkov(rho, lambda, 1, M, etas[i2]);
        double t4 = thetaMarkov(rho, lambda, 1, M, etas[i4]);
        lab.verdict("C1 Markov bound", f("Markov bound %.3f (eta=1e-2) and %.3f (eta=1e-4) holds", t2, t4),
                f("upper CI %.4f <= %.4f and %.4f <= %.4f (slack x%.1f, x%.1f)", emp[i2][2], t2, emp[i4][2], t4,
                        t2 / emp[i2][0], t4 / emp[i4][0]), emp[i2][2] <= t2 && emp[i4][2] <= t4);
        double c2 = sdRel * Math.sqrt((1.0 - etas[i2]) / etas[i2]);
        double c4 = sdRel * Math.sqrt((1.0 - etas[i4]) / etas[i4]);
        lab.verdict("C1 Cantelli bound", f("Cantelli %.3f (eta=1e-2) and %.2f (eta=1e-4, vacuous)", c2, c4),
                f("%.4f <= %.3f; %.2f >= 1 is vacuous", emp[i2][2], c2, c4), emp[i2][2] <= c2 && c4 >= 1.0);
        double ratio = emp[i4][0] / emp[i2][0];
        double ratioLo = emp[i4][1] / emp[i2][2];
        double ratioHi = emp[i4][2] / emp[i2][1];
        double sqrtLog = Math.sqrt(Math.log(1.0 / etas[i4]) / Math.log(1.0 / etas[i2]));
        double gauss = FwStats.invNormal(1.0 - etas[i4]) / FwStats.invNormal(1.0 - etas[i2]);
        lab.verdict("C1 growth", f("quantiles grow like sqrt(ln 1/eta): ratio eta=1e-4 : 1e-2 in [1.2, 2.0] "
                        + "(sqrt-log %.2f, Gaussian %.2f; 1/sqrt(eta) shape %.1f)", sqrtLog, gauss,
                        Math.sqrt(etas[i2] / etas[i4])),
                f("ratio %.3f [%.3f, %.3f]", ratio, ratioLo, ratioHi), ratioLo >= 1.2 && ratioHi <= 2.0);
    }

    /** The M' value at 0-based rank r of the sorted sample summarized by {@code hist}. */
    static int valueAtRank(long[] hist, long r) {
        long cum = 0;
        for (int v = 0; v < hist.length; v++) {
            cum += hist[v];
            if (cum > r) {
                return v;
            }
        }
        return hist.length - 1;
    }

    private static void conditionalLaw(Lab lab) throws IOException {
        CostSpace sp = lab.instance().full(2.0);
        String[] arms = {"loss-free m'=9000", "Bernoulli / random", "Gilbert / random", "block / random",
            "Bernoulli / sorted", "Gilbert / sorted", "block / sorted"};
        LossChannel bern = LossChannel.bernoulli(0.1);
        LossChannel ge = LossChannel.gilbert(ALPHA, BETA);
        LossChannel block = LossChannel.block(0.1);
        LossChannel[] ch = {null, bern, ge, block, bern, ge, block};
        int samples = lab.pick(4000, 80);
        List<double[]> res = lab.parallel(arms.length * samples, idx -> {
            int arm = idx / samples;
            int i = idx % samples;
            Buffers buf = Buffers.get();
            int[] draws = buf.ints(M);
            boolean[] lost = buf.flags(M);
            Tally t = new Tally(sp);
            SplittableRandom sample = Seeds.rng(lab.seed, "C2-sample", arm, i);
            long attempts = 0;
            if (arm == 0) {
                sp.draws(draws, TARGET, sample);
                for (int s = 0; s < TARGET; s++) {
                    t.add(draws[s], true);
                }
            } else {
                // Rejection on the channel alone: exact conditioning on M' = TARGET.
                SplittableRandom chan = Seeds.rng(lab.seed, "C2-channel", arm, i);
                do {
                    attempts++;
                } while (ch[arm].erase(lost, M, chan) != TARGET);
                sp.draws(draws, M, sample);
                if (arm >= 4) {
                    buf.sortByClass(draws, M, sp);
                } else {
                    Buffers.shuffle(draws, M, Seeds.rng(lab.seed, "C2-pack", arm, i));
                }
                for (int s = 0; s < M; s++) {
                    t.add(draws[s], !lost[s]);
                }
            }
            return new double[] {sp.rel(t.rc(0), 0), t.rcMaxAbs(), attempts};
        });
        try (Writer w = lab.raw("C2-conditional-errors", "arm,sample,rc_rel_q0,rc_maxabs,attempts")) {
            for (int idx = 0; idx < res.size(); idx++) {
                double[] v = res.get(idx);
                w.write(f("%d,%d,%.17g,%.17g,%d%n", idx / samples, idx % samples, v[0], v[1], (long) v[2]));
            }
        }
        double[][] e0 = new double[arms.length][];
        double[][] emax = new double[arms.length][];
        for (int a = 0; a < arms.length; a++) {
            e0[a] = ExpA.col(res, a, samples, 0);
            emax[a] = ExpA.col(res, a, samples, 1);
        }
        int perms = lab.pick(999, 199);
        Table tab = lab.table("C2-conditional-law", "arm", "mean err Q0", "sd err Q0", "mean max_Q |err|",
                "KS p (Q0)", "AD p (Q0)", "KS p (max_Q)", "AD p (max_Q)");
        double minSame = 1.0;
        double[] minP = new double[arms.length];
        for (int a = 0; a < arms.length; a++) {
            String[] p = {"-", "-", "-", "-"};
            if (a > 0) {
                double[] ps = {FwStats.ks(e0[a], e0[0]).p(),
                    FwStats.ad(e0[a], e0[0], perms, Seeds.of(lab.seed, "C2-perm", a, 0)).p(),
                    FwStats.ks(emax[a], emax[0]).p(),
                    FwStats.ad(emax[a], emax[0], perms, Seeds.of(lab.seed, "C2-perm", a, 1)).p()};
                for (int k = 0; k < 4; k++) {
                    p[k] = f("%.4f", ps[k]);
                }
                double mn = Math.min(Math.min(ps[0], ps[1]), Math.min(ps[2], ps[3]));
                if (a <= 4) {
                    minSame = Math.min(minSame, mn);
                } else {
                    minP[a] = mn;
                }
            }
            tab.row(arms[a], f("%+.5f", FwStats.mean(e0[a])), f("%.5f", FwStats.sd(e0[a])),
                    f("%.5f", FwStats.mean(emax[a])), p[0], p[1], p[2], p[3]);
        }
        Table pair = lab.table("C2-conditional-pairs", "pair", "KS p (Q0)", "AD p (Q0)", "KS p (max_Q)",
                "AD p (max_Q)");
        int[][] pairs = {{1, 2}, {1, 3}, {2, 3}};
        for (int[] pr : pairs) {
            double[] ps = {FwStats.ks(e0[pr[0]], e0[pr[1]]).p(),
                FwStats.ad(e0[pr[0]], e0[pr[1]], perms, Seeds.of(lab.seed, "C2-pair", pr[0], pr[1])).p(),
                FwStats.ks(emax[pr[0]], emax[pr[1]]).p(),
                FwStats.ad(emax[pr[0]], emax[pr[1]], perms, Seeds.of(lab.seed, "C2-pair", pr[1], pr[0])).p()};
            pair.row(arms[pr[0]] + " vs " + arms[pr[1]], f("%.4f", ps[0]), f("%.4f", ps[1]), f("%.4f", ps[2]),
                    f("%.4f", ps[3]));
            minSame = Math.min(minSame, Math.min(Math.min(ps[0], ps[1]), Math.min(ps[2], ps[3])));
        }
        lab.section("C2. The RC error given M' = 9000",
                f("Sensitivity sample of the mixture, m = %d, B = 1, %d samples per arm, each conditioned exactly on "
                        + "M' = %d by rejection on the channel alone (the draws are made after acceptance). Random "
                        + "packing = uniform permutation of the draws; sorted packing = draws sorted by mixture "
                        + "component. Block = the middle 1000 packets lost every time. Tests against the loss-free "
                        + "sample of 9000 draws: two-sample KS (asymptotic p) and Anderson-Darling (midrank, %d "
                        + "permutations).", M, samples, TARGET, perms), tab);
        lab.section(pair.markdown());
        lab.verdict("C2 identical given M'", "given M', the error law is the same under Bernoulli, Gilbert-Elliott and "
                        + "deterministic loss with random packing (all p > 0.001; 28 tests)",
                f("min p over random-packing arms, Bernoulli/sorted and the 3 pairs = %.4f", minSame),
                minSame > 0.001);
        lab.verdict("C2 sorted + block", "sorted packing with deterministic block loss breaks the conditional law",
                f("min p of the 4 tests vs loss-free = %.2e; mean err Q0 %+.4f vs %+.4f", minP[6],
                        FwStats.mean(e0[6]), FwStats.mean(e0[0])), minP[6] < 0.001);
        lab.verdict("C2 sorted + bursts", "sorted packing with Gilbert bursts breaks the conditional law",
                f("min p of the 4 tests vs loss-free = %.2e; sd err Q0 %.5f vs %.5f", minP[5], FwStats.sd(e0[5]),
                        FwStats.sd(e0[0])), minP[5] < 0.001);
    }
}
