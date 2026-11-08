package io.github.c0rwin.coreset.testbed.faults.framework;

import static io.github.c0rwin.coreset.testbed.faults.framework.Lab.f;

import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.DoublePredicate;

/**
 * A. Unknown loss rate and lossy headers (Theorem "Unknown loss rate and lossy headers"): a misspecified HT decoder
 * has a floor while RC converges; header copies in separate packets on a Gilbert channel; a header in every packet.
 */
final class ExpA {

    static final double RHO = 0.3;
    static final double RHO_HAT = 0.2;
    static final double ALPHA = 0.01;
    static final double BETA = 0.09;

    private ExpA() {
    }

    static void run(Lab lab) throws IOException {
        CostSpace sp = lab.instance().full(2.0);
        unknownRate(lab, sp);
        headerCopies(lab);
        everyPacket(lab, sp);
    }

    private static void unknownRate(Lab lab, CostSpace sp) throws IOException {
        int[] ms = lab.quick() ? new int[] {100, 300, 1000} : new int[] {100, 300, 1000, 3000, 10000, 30000, 100000};
        int reps = lab.pick(500, 30);
        LossChannel ch = LossChannel.bernoulli(RHO);
        double p0 = 1.0 - RHO_HAT;
        List<double[]> res = lab.parallel(ms.length * reps, idx -> {
            int mi = idx / reps;
            int r = idx % reps;
            int m = ms[mi];
            Buffers b = Buffers.get();
            int[] draws = b.ints(m);
            boolean[] lost = b.flags(m);
            sp.draws(draws, m, Seeds.rng(lab.seed, "A1-sample", mi, r));
            ch.erase(lost, m, Seeds.rng(lab.seed, "A1-channel", mi, r));
            Tally t = new Tally(sp);
            // B = 1 in draw order: i.i.d. draws make the order draw-oblivious already.
            for (int s = 0; s < m; s++) {
                t.add(draws[s], !lost[s]);
            }
            double htMax = 0.0;
            for (int j = 0; j < sp.queries; j++) {
                htMax = Math.max(htMax, Math.abs(sp.rel(t.ht(j, p0), j)));
            }
            return new double[] {t.received(), t.ht(0, p0) / sp.cost[0], sp.rel(t.rc(0), 0), t.rcMaxAbs(), htMax};
        });
        try (Writer w = lab.raw("A1-unknown-rate", "m,rep,received,ht_ratio_q0,rc_rel_q0,rc_maxabs,ht_maxabs")) {
            for (int idx = 0; idx < res.size(); idx++) {
                double[] v = res.get(idx);
                w.write(f("%d,%d,%d,%.17g,%.17g,%.17g,%.17g%n", ms[idx / reps], idx % reps, (long) v[0], v[1], v[2],
                        v[3], v[4]));
            }
        }
        Table tab = lab.table("A1-unknown-rate", "m", "mean M'", "HT ratio mean [95% CI]", "HT ratio sd",
                "P[HT fails, eps=0.1]", "RC RMS rel. err. Q0 [95% CI]", "RC mean rel. err. Q0", "P[RC fails, eps=0.1]",
                "RC RMS max_Q err.");
        double[] x = new double[ms.length];
        double[][] rcCells = new double[ms.length][];
        double[] htLast = null;
        int htCovered = 0;
        for (int mi = 0; mi < ms.length; mi++) {
            double[] recv = col(res, mi, reps, 0);
            double[] ht = col(res, mi, reps, 1);
            double[] rc = col(res, mi, reps, 2);
            double[] rcMax = col(res, mi, reps, 3);
            FwStats.Ci htCi = FwStats.meanCi(ht);
            htCovered += htCi.contains((1.0 - RHO) / p0) ? 1 : 0;
            FwStats.Ci rcRms = FwStats.rmsCi(rc);
            tab.row(ms[mi], f("%.1f", FwStats.mean(recv)), ci(htCi, "%.4f"), f("%.4f", FwStats.sd(ht)),
                    prob(count(ht, v -> Math.abs(v - 1.0) > 0.1), reps), ci(rcRms, "%.4f"),
                    f("%+.4f", FwStats.mean(rc)), prob(count(rc, v -> Math.abs(v) > 0.1), reps),
                    f("%.4f", FwStats.rms(rcMax)));
            x[mi] = Math.log(FwStats.mean(recv));
            rcCells[mi] = rc;
            htLast = ht;
        }
        FwStats.Ci slope = FwStats.bootstrapLogSlope(x, rcCells, FwStats::rms, 2000,
                Seeds.of(lab.seed, "A1-boot", 0, 0));
        lab.section("A1. Misspecified HT versus RC (rho = 0.3, HT assumes 0.2)",
                f("Bernoulli channel, B = 1, sensitivity sample of the mixture, %d repetitions per m. The HT decoder "
                        + "multiplies by 1/(1-0.2); RC divides by M'. Fixed query Q0. Predicted HT ratio "
                        + "(1-0.3)/(1-0.2) = 0.875; RC error ~ M'^-1/2. Slope of log RMS(RC) on log mean M': "
                        + "%s.", reps, ci(slope, "%.3f")), tab);
        FwStats.Ci last = FwStats.meanCi(htLast);
        double failLast = (double) count(htLast, v -> Math.abs(v - 1.0) > 0.1) / reps;
        boolean floorOk = Math.abs(last.value() - 0.875) <= 0.005 && failLast >= 0.99;
        lab.verdict("A1 HT floor", "rho=0.3, HT assumes rho=0.2: fixed-query ratio -> 0.875 (a 12.5% floor), failure "
                + "-> 1 for eps < 0.125", f("m=%d: ratio %s, sd %.4f; P[|ratio-1|>0.1] = %.3f; mean-ratio CI covers "
                + "0.875 in %d/%d cells", ms[ms.length - 1], ci(last, "%.4f"), FwStats.sd(htLast), failLast, htCovered,
                ms.length), floorOk);
        lab.verdict("A1 RC rate", "RC error decays as M'^-1/2 (slope -0.5 +- 0.05)",
                f("slope of log RMS on log M' = %s over m = %d..%d", ci(slope, "%.3f"), ms[0], ms[ms.length - 1]),
                Math.abs(slope.value() + 0.5) <= 0.05);
    }

    private static void headerCopies(Lab lab) throws IOException {
        long trials = lab.pick(20_000_000L, 200_000L);
        int chunks = 200;
        long per = trials / chunks;
        LossChannel ge = LossChannel.gilbert(ALPHA, BETA);
        List<long[]> res = lab.parallel(chunks, c -> {
            SplittableRandom rng = Seeds.rng(lab.seed, "A2-channel", c, 0);
            boolean[] lost = Buffers.get().flags(201);
            long adj = 0;
            long spaced = 0;
            long first = 0;
            for (long t = 0; t < per; t++) {
                ge.erase(lost, 201, rng);
                first += lost[0] ? 1 : 0;
                adj += lost[0] && lost[1] && lost[2] ? 1 : 0;
                spaced += lost[0] && lost[100] && lost[200] ? 1 : 0;
            }
            return new long[] {first, adj, spaced};
        });
        long[] tot = new long[3];
        for (long[] r : res) {
            for (int i = 0; i < 3; i++) {
                tot[i] += r[i];
            }
        }
        long n = per * chunks;
        try (Writer w = lab.raw("A2-header-copies", "trials,first_lost,adjacent_all_lost,spaced100_all_lost")) {
            w.write(f("%d,%d,%d,%d%n", n, tot[0], tot[1], tot[2]));
        }
        double piB = ALPHA / (ALPHA + BETA);
        double lambda = 1.0 - ALPHA - BETA;
        double predAdj = piB * (1.0 - BETA) * (1.0 - BETA);
        double stay100 = piB + (1.0 - piB) * Math.pow(lambda, 100);
        double predSpaced = piB * stay100 * stay100;
        double z99 = 2.5758293035489;
        FwStats.Ci first = FwStats.wilson(tot[0], n, z99);
        FwStats.Ci adj = FwStats.wilson(tot[1], n, z99);
        FwStats.Ci spaced = FwStats.wilson(tot[2], n, z99);
        Table tab = lab.table("A2-header-copies", "event", "predicted", "measured [99% Wilson CI]");
        tab.row("one packet lost (loss rate)", f("%.5f", piB), ci(first, "%.5f"));
        tab.row("3 adjacent copies all lost", f("%.5f = pi_B (1-beta)^2", predAdj), ci(adj, "%.5f"));
        tab.row("3 copies 100 apart all lost", f("%.4e = pi_B (pi_B + (1-pi_B) lambda^100)^2", predSpaced),
                ci(spaced, "%.4e"));
        tab.row("memoryless reference rho^3", f("%.4e", piB * piB * piB), "-");
        lab.section("A2. Header copies on the Gilbert channel",
                f("Gilbert channel alpha = %.2f, beta_GE = %.2f (rho = %.2f, lambda = %.2f, mean burst %.1f), stationary "
                        + "start, 201-packet stream, %d trials. Header copies at packets {0,1,2} or {0,100,200}.",
                        ALPHA, BETA, piB, lambda, 1.0 / BETA, n), tab);
        lab.verdict("A2 header copies", "3 adjacent copies all lost w.p. ~0.083; 100 apart ~1.0e-3",
                f("adjacent %s (pred %.5f); spaced %s (pred %.4e); ratio %.0fx", ci(adj, "%.5f"), predAdj,
                        ci(spaced, "%.3e"), predSpaced, adj.value() / spaced.value()),
                adj.contains(predAdj) && spaced.contains(predSpaced));
    }

    /**
     * Header placement: reliable header (oracle m), m in every data packet, 3 adjacent header packets, 3 header packets
     * 100 apart. One 203-slot channel realization per trial; failure = header unreadable or max_Q RC error > eps.
     */
    private static void everyPacket(Lab lab, CostSpace sp) throws IOException {
        final int m = 2000;
        final int b = 10;
        final int data = m / b;
        final int slots = data + 3;
        final double eps = 0.15;
        int trials = lab.pick(200_000, 2_000);
        LossChannel ge = LossChannel.gilbert(ALPHA, BETA);
        int[][] dataSlots = {range(0, data), range(3, data), without(slots, 0, 101, 202)};
        int[][] headerSlots = {null, {0, 1, 2}, {0, 101, 202}};
        List<double[]> res = lab.parallel(trials, tr -> {
            Buffers buf = Buffers.get();
            int[] draws = buf.ints(m);
            boolean[] lost = buf.flags(slots);
            sp.draws(draws, m, Seeds.rng(lab.seed, "A3-sample", tr, 0));
            ge.erase(lost, slots, Seeds.rng(lab.seed, "A3-channel", tr, 0));
            double[] out = new double[8];
            for (int lay = 0; lay < 3; lay++) {
                Tally t = new Tally(sp);
                for (int p = 0; p < data; p++) {
                    boolean ok = !lost[dataSlots[lay][p]];
                    for (int s = p * b; s < (p + 1) * b; s++) {
                        t.add(draws[s], ok);
                    }
                }
                boolean header;
                if (lay == 0) {
                    header = true;
                } else {
                    header = false;
                    for (int h : headerSlots[lay]) {
                        header |= !lost[h];
                    }
                }
                boolean errFail = t.rcMaxAbs() > eps;
                if (lay == 0) {
                    out[0] = errFail ? 1 : 0;
                    // m in every data packet, unreadable only if no data packet arrives.
                    boolean everyHeader = t.received() > 0;
                    out[1] = everyHeader ? 0 : 1;
                    out[2] = !everyHeader || errFail ? 1 : 0;
                    out[7] = t.received();
                } else {
                    out[2 * lay + 1] = header ? 0 : 1;
                    out[2 * lay + 2] = !header || errFail ? 1 : 0;
                }
            }
            return out;
        });
        try (Writer w = lab.raw("A3-header-placement",
                "trial,fail_reliable,hdrlost_every,fail_every,hdrlost_adjacent,fail_adjacent,hdrlost_spaced,"
                        + "fail_spaced,received")) {
            for (int tr = 0; tr < trials; tr++) {
                double[] v = res.get(tr);
                w.write(f("%d,%d,%d,%d,%d,%d,%d,%d,%d%n", tr, (int) v[0], (int) v[1], (int) v[2], (int) v[3],
                        (int) v[4], (int) v[5], (int) v[6], (int) v[7]));
            }
        }
        long[] sums = new long[7];
        long discrepancies = 0;
        for (double[] v : res) {
            for (int i = 0; i < 7; i++) {
                sums[i] += (long) v[i];
            }
            discrepancies += v[0] != v[2] ? 1 : 0;
        }
        Table tab = lab.table("A3-header-placement", "scheme", "P[header unreadable] [95% CI]",
                "P[failure] [95% CI]", "predicted P[header unreadable]");
        double piB = ALPHA / (ALPHA + BETA);
        tab.row("reliable header (oracle m)", "0", prob(sums[0], trials, "%.4f"), "0");
        tab.row("m in every data packet", prob(sums[1], trials, "%.2e"), prob(sums[2], trials, "%.4f"),
                f("%.1e (all %d packets lost)", piB * Math.pow(1.0 - BETA, data - 1), data));
        tab.row("3 adjacent header packets", prob(sums[3], trials, "%.4f"), prob(sums[4], trials, "%.4f"),
                f("%.4f", piB * (1.0 - BETA) * (1.0 - BETA)));
        tab.row("3 header packets 101 apart", prob(sums[5], trials, "%.2e"), prob(sums[6], trials, "%.4f"),
                f("%.2e", piB * Math.pow(piB + (1 - piB) * Math.pow(1.0 - ALPHA - BETA, 101), 2)));
        lab.section("A3. Header placement",
                f("m = %d draws in %d packets of B = %d on the Gilbert channel above (one 203-slot realization per "
                        + "trial, %d trials); failure means no readable m or max_Q RC error > %.2f. Trials whose "
                        + "failure differs between the reliable header and m in every packet: %d.", m, data, b,
                        trials, eps, discrepancies), tab);
        lab.verdict("A3 header in every packet", "a header in every packet adds no failures",
                f("%d/%d trials differ from the reliable-header decoder; P[fail] %.4f vs %.4f; 3 adjacent header "
                        + "packets: P[fail] %.4f", discrepancies, trials, (double) sums[2] / trials,
                        (double) sums[0] / trials, (double) sums[4] / trials),
                discrepancies == 0);
    }

    static int[] range(int from, int len) {
        int[] r = new int[len];
        for (int i = 0; i < len; i++) {
            r[i] = from + i;
        }
        return r;
    }

    static int[] without(int n, int... skip) {
        int[] r = new int[n - skip.length];
        for (int i = 0, w = 0; i < n; i++) {
            boolean keep = true;
            for (int s : skip) {
                keep &= s != i;
            }
            if (keep) {
                r[w++] = i;
            }
        }
        return r;
    }

    static double[] col(List<double[]> res, int cell, int reps, int k) {
        double[] v = new double[reps];
        for (int r = 0; r < reps; r++) {
            v[r] = res.get(cell * reps + r)[k];
        }
        return v;
    }

    static long count(double[] x, DoublePredicate p) {
        long c = 0;
        for (double v : x) {
            c += p.test(v) ? 1 : 0;
        }
        return c;
    }

    static String prob(long k, long n) {
        return prob(k, n, "%.3f");
    }

    static String prob(long k, long n, String fmt) {
        return ci(FwStats.wilson(k, n, FwStats.Z95), fmt);
    }

    static String ci(FwStats.Ci c, String fmt) {
        return f(fmt + " [" + fmt + ", " + fmt + "]", c.value(), c.lo(), c.hi());
    }
}
