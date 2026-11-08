package io.github.c0rwin.coreset.testbed.faults.framework;

import static io.github.c0rwin.coreset.testbed.faults.framework.ExpA.ci;
import static io.github.c0rwin.coreset.testbed.faults.framework.Lab.f;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * D. Multi-hop and multi-level loss (Theorem "Multi-hop and multi-level loss"): forwarding relays compound the loss
 * but not the error; resampling at every level resets the loss but compounds the error as prod(1 + eps_l) - 1.
 */
final class ExpD {

    static final double[] HOPS = {0.1, 0.2, 0.3};
    static final int B = 8;

    private ExpD() {
    }

    static void run(Lab lab) throws IOException {
        forwarding(lab);
        tree(lab);
    }

    /**
     * Sends {@code len} draws (in {@code cur}) through the hops: each hop (and the source) packs the surviving records
     * by a uniform permutation into packets of {@code b} and applies its stored loss pattern. Returns the survivors.
     */
    static int forward(int[] cur, int len, int b, boolean[][] lost, int hops, SplittableRandom pack) {
        for (int h = 0; h < hops; h++) {
            Buffers.shuffle(cur, len, pack);
            int w = 0;
            for (int s = 0; s < len; s++) {
                if (!lost[h][s / b]) {
                    cur[w++] = cur[s];
                }
            }
            len = w;
        }
        return len;
    }

    private static void forwarding(Lab lab) throws IOException {
        CostSpace sp = lab.instance().full(2.0);
        final int m = 8000;
        double keep = 1.0;
        for (double r : HOPS) {
            keep *= 1.0 - r;
        }
        final double keepAll = keep;
        final int target = (int) Math.round(keep * m / B) * B;
        LossChannel[] chain = new LossChannel[HOPS.length];
        for (int h = 0; h < HOPS.length; h++) {
            chain[h] = LossChannel.bernoulli(HOPS[h]);
        }
        LossChannel[] single = {LossChannel.bernoulli(1.0 - keep)};
        // Unconditional end-to-end delivered fraction from the channel alone.
        int runs = lab.pick(20000, 500);
        List<double[]> rate = lab.parallel(runs, r -> {
            SplittableRandom chan = Seeds.rng(lab.seed, "D1-rate", r, 0);
            int len = m;
            for (LossChannel c : chain) {
                len = B * c.delivered(len / B, chan);
            }
            return new double[] {(double) len / m};
        });
        double[] frac = new double[runs];
        for (int r = 0; r < runs; r++) {
            frac[r] = rate.get(r)[0];
        }
        FwStats.Ci fracCi = FwStats.meanCi(frac);
        String[] arms = {"loss-free m'=" + target, "3 forwarding hops", "1 hop, rho = 1 - prod(1-rho_h)"};
        int samples = lab.pick(3000, 60);
        List<double[]> res = lab.parallel(arms.length * samples, idx -> {
            int arm = idx / samples;
            int i = idx % samples;
            Buffers buf = Buffers.get();
            int[] cur = buf.ints(m);
            Tally t = new Tally(sp);
            SplittableRandom sample = Seeds.rng(lab.seed, "D1-sample", arm, i);
            if (arm == 0) {
                sp.draws(cur, target, sample);
                for (int s = 0; s < target; s++) {
                    t.add(cur[s], true);
                }
                return new double[] {sp.rel(t.rc(0), 0), t.rcMaxAbs(), 0};
            }
            LossChannel[] hops = arm == 1 ? chain : single;
            boolean[][] lost = new boolean[hops.length][m / B];
            SplittableRandom chan = Seeds.rng(lab.seed, "D1-channel", arm, i);
            long attempts = 0;
            int len;
            do {
                attempts++;
                len = m;
                for (int h = 0; h < hops.length; h++) {
                    len = B * hops[h].erase(lost[h], len / B, chan);
                }
            } while (len != target);
            sp.draws(cur, m, sample);
            int got = forward(cur, m, B, lost, hops.length, Seeds.rng(lab.seed, "D1-pack", arm, i));
            for (int s = 0; s < got; s++) {
                t.add(cur[s], true);
            }
            return new double[] {sp.rel(t.rc(0), 0), t.rcMaxAbs(), attempts};
        });
        try (Writer w = lab.raw("D1-forwarding", "arm,sample,rc_rel_q0,rc_maxabs,attempts")) {
            for (int idx = 0; idx < res.size(); idx++) {
                double[] v = res.get(idx);
                w.write(f("%d,%d,%.17g,%.17g,%d%n", idx / samples, idx % samples, v[0], v[1], (long) v[2]));
            }
        }
        int perms = lab.pick(999, 199);
        Table tab = lab.table("D1-forwarding", "arm / pair", "mean err Q0", "sd err Q0", "mean max_Q |err|",
                "KS p (Q0)", "AD p (Q0)", "KS p (max_Q)", "AD p (max_Q)");
        double[][] e0 = new double[3][];
        double[][] em = new double[3][];
        for (int a = 0; a < 3; a++) {
            e0[a] = ExpA.col(res, a, samples, 0);
            em[a] = ExpA.col(res, a, samples, 1);
        }
        double minP = 1.0;
        int[][] pairs = {{1, 0}, {2, 0}, {1, 2}};
        for (int a = 0; a < 3; a++) {
            tab.row(arms[a], f("%+.5f", FwStats.mean(e0[a])), f("%.5f", FwStats.sd(e0[a])),
                    f("%.5f", FwStats.mean(em[a])), "-", "-", "-", "-");
        }
        for (int[] pr : pairs) {
            double[] ps = {FwStats.ks(e0[pr[0]], e0[pr[1]]).p(),
                FwStats.ad(e0[pr[0]], e0[pr[1]], perms, Seeds.of(lab.seed, "D1-perm", pr[0], pr[1])).p(),
                FwStats.ks(em[pr[0]], em[pr[1]]).p(),
                FwStats.ad(em[pr[0]], em[pr[1]], perms, Seeds.of(lab.seed, "D1-perm", pr[1], pr[0])).p()};
            for (double p : ps) {
                minP = Math.min(minP, p);
            }
            tab.row(arms[pr[0]] + " vs " + arms[pr[1]], "", "", "", f("%.4f", ps[0]), f("%.4f", ps[1]),
                    f("%.4f", ps[2]), f("%.4f", ps[3]));
        }
        lab.section("D1. Forwarding relays",
                f("m = %d draws, B = %d, hops with rho = 0.1, 0.2, 0.3; every relay unpacks, permutes uniformly and "
                        + "repacks; m travels in the headers. End-to-end delivered fraction over %d channel runs: %s "
                        + "(predicted prod(1-rho_h) = %.3f, price 1/%.3f = %.3f). Error law given M' = %d (exact "
                        + "conditioning by rejection on the channels), %d samples per arm:", m, B, runs,
                        ci(fracCi, "%.4f"), keepAll, keepAll, 1.0 / keepAll, target, samples), tab);
        lab.verdict("D1 forwarding price", "H forwarding hops deliver prod(1-rho_h) of the records: price "
                        + f("1/prod(1-rho_h) = %.3f", 1.0 / keepAll),
                f("delivered fraction %s vs %.4f (|z| = %.2f)", ci(fracCi, "%.4f"), keepAll, FwStats.z(frac, keepAll)),
                FwStats.z(frac, keepAll) <= 3.0);
        lab.verdict("D1 forwarding accuracy", "forwarding leaves the error unchanged given M' (same law as a "
                        + "loss-free sample of M' draws)",
                f("min p over 3 pairs x 2 metrics x (KS, AD) = %.4f", minP), minP > 0.001);
    }

    /** One run of the resampling tree and the forwarding chain of the same depth. */
    private record TreeRun(double rootQ0, double rootMax, double[] epsLevel, boolean composed, double edgeFrac,
                           double fwdFrac, double fwdMax, double fwdPricedMax) {
    }

    private static void tree(Lab lab) throws IOException {
        Instance inst = lab.instance();
        Mixture data = inst.data;
        double[] table = inst.table(2.0);
        CostSpace full = inst.full(2.0);
        final int qn = Instance.QUERIES;
        final int mNode = lab.pick(2000, 300);
        final double rho = 0.2;
        int[] depths = lab.quick() ? new int[] {1, 2} : new int[] {1, 2, 3, 4};
        int reps = lab.pick(300, 6);
        LossChannel ch = LossChannel.bernoulli(rho);
        List<TreeRun> res = lab.parallel(depths.length * reps, idx -> {
            int di = idx / reps;
            int r = idx % reps;
            int depth = depths[di];
            SplittableRandom part = Seeds.rng(lab.seed, "D2-partition", di, r);
            SplittableRandom sample = Seeds.rng(lab.seed, "D2-sample", di, r);
            SplittableRandom chan = Seeds.rng(lab.seed, "D2-channel", di, r);
            int[] perm = CostSpace.identity(data.n);
            Buffers.shuffle(perm, data.n, part);
            int leaves = 1 << depth;
            List<int[]> rows = new ArrayList<>();
            List<double[]> weights = new ArrayList<>();
            for (int v = 0; v < leaves; v++) {
                int from = (int) ((long) v * data.n / leaves);
                int to = (int) ((long) (v + 1) * data.n / leaves);
                int[] rr = Arrays.copyOfRange(perm, from, to);
                Arrays.sort(rr);
                double[] ww = new double[rr.length];
                Arrays.fill(ww, 1.0);
                rows.add(rr);
                weights.add(ww);
            }
            double[] eps = new double[depth * qn];
            double[] rootCost = new double[qn];
            long sent = 0;
            long got = 0;
            boolean[] lost = new boolean[mNode];
            int[] draws = new int[mNode];
            for (int level = 0; level < depth; level++) {
                List<int[]> nextRows = new ArrayList<>();
                List<double[]> nextW = new ArrayList<>();
                for (int v = 0; v < rows.size(); v += 2) {
                    int[][] childRows = new int[2][];
                    double[][] childW = new double[2][];
                    for (int c = 0; c < 2; c++) {
                        int[] ur = rows.get(v + c);
                        double[] uw = weights.get(v + c);
                        double[] s = data.proposal(ur, uw, inst.k, 2.0, sample);
                        CostSpace node = new CostSpace(ur, uw, s, table, qn, null, 1);
                        node.draws(draws, mNode, sample);
                        int recv = ch.erase(lost, mNode, chan);
                        sent += mNode;
                        got += recv;
                        Tally t = new Tally(node);
                        for (int d = 0; d < mNode; d++) {
                            t.add(draws[d], !lost[d]);
                        }
                        for (int j = 0; j < qn; j++) {
                            eps[level * qn + j] = Math.max(eps[level * qn + j], Math.abs(node.rel(t.rc(j), j)));
                            if (level == depth - 1) {
                                rootCost[j] += t.rc(j);
                            }
                        }
                        // RC-decoded message: each delivered draw of t carries w_t / (M' q_t), merged per member.
                        double[] msgW = new double[node.size];
                        for (int d = 0; d < mNode; d++) {
                            if (!lost[d]) {
                                msgW[draws[d]] += node.weight[draws[d]] / (recv * node.q[draws[d]]);
                            }
                        }
                        int kept = 0;
                        for (double x : msgW) {
                            kept += x > 0.0 ? 1 : 0;
                        }
                        childRows[c] = new int[kept];
                        childW[c] = new double[kept];
                        for (int t2 = 0, w = 0; t2 < node.size; t2++) {
                            if (msgW[t2] > 0.0) {
                                childRows[c][w] = ur[t2];
                                childW[c][w++] = msgW[t2];
                            }
                        }
                    }
                    int[] ur = new int[childRows[0].length + childRows[1].length];
                    double[] uw = new double[ur.length];
                    System.arraycopy(childRows[0], 0, ur, 0, childRows[0].length);
                    System.arraycopy(childRows[1], 0, ur, childRows[0].length, childRows[1].length);
                    System.arraycopy(childW[0], 0, uw, 0, childW[0].length);
                    System.arraycopy(childW[1], 0, uw, childW[0].length, childW[1].length);
                    nextRows.add(ur);
                    nextW.add(uw);
                }
                rows = nextRows;
                weights = nextW;
            }
            double rootMax = 0.0;
            boolean composed = true;
            double[] epsLevel = new double[depth];
            for (int j = 0; j < qn; j++) {
                double e = Math.abs(full.rel(rootCost[j], j));
                rootMax = Math.max(rootMax, e);
                double bound = 1.0;
                for (int l = 0; l < depth; l++) {
                    bound *= 1.0 + eps[l * qn + j];
                    epsLevel[l] = Math.max(epsLevel[l], eps[l * qn + j]);
                }
                composed &= e <= bound - 1.0 + 1e-12;
            }
            // Forwarding chain of the same depth from one source, B = 1: m draws, and m / (1-rho)^depth draws.
            double[] fwd = new double[3];
            for (int priced = 0; priced < 2; priced++) {
                int mSrc = priced == 0 ? mNode : (int) Math.ceil(mNode / Math.pow(1.0 - rho, depth));
                int[] cur = new int[mSrc];
                full.draws(cur, mSrc, sample);
                boolean[][] hopLost = new boolean[depth][mSrc];
                int len = mSrc;
                for (int h = 0; h < depth; h++) {
                    ch.erase(hopLost[h], len, chan);
                    int w = 0;
                    for (int s = 0; s < len; s++) {
                        if (!hopLost[h][s]) {
                            cur[w++] = cur[s];
                        }
                    }
                    len = w;
                }
                Tally t = new Tally(full);
                for (int s = 0; s < len; s++) {
                    t.add(cur[s], true);
                }
                if (priced == 0) {
                    fwd[0] = (double) len / mSrc;
                    fwd[1] = t.rcMaxAbs();
                } else {
                    fwd[2] = t.rcMaxAbs();
                }
            }
            return new TreeRun(full.rel(rootCost[0], 0), rootMax, epsLevel, composed, (double) got / sent, fwd[0],
                    fwd[1], fwd[2]);
        });
        try (Writer w = lab.raw("D2-tree", "depth,rep,root_rel_q0,root_maxabs,composed,edge_delivered,"
                + "fwd_delivered,fwd_maxabs,fwd_priced_maxabs,eps_levels")) {
            for (int idx = 0; idx < res.size(); idx++) {
                TreeRun t = res.get(idx);
                StringBuilder sb = new StringBuilder(f("%d,%d,%.17g,%.17g,%b,%.6f,%.6f,%.17g,%.17g,", depths[idx / reps],
                        idx % reps, t.rootQ0(), t.rootMax(), t.composed(), t.edgeFrac(), t.fwdFrac(), t.fwdMax(),
                        t.fwdPricedMax()));
                for (int l = 0; l < t.epsLevel().length; l++) {
                    sb.append(l == 0 ? "" : ";").append(f("%.9g", t.epsLevel()[l]));
                }
                w.write(sb.append('\n').toString());
            }
        }
        // Loss-free reference: the forwarding chain at depth 0 is a plain sample of mNode draws.
        int refReps = 4 * reps;
        List<double[]> ref = lab.parallel(refReps, r -> {
            int[] cur = new int[mNode];
            full.draws(cur, mNode, Seeds.rng(lab.seed, "D2-ref", r, 0));
            Tally t = new Tally(full);
            for (int s = 0; s < mNode; s++) {
                t.add(cur[s], true);
            }
            return new double[] {t.rcMaxAbs()};
        });
        double[] refMax = ExpA.col(ref, 0, refReps, 0);
        FwStats.Ci refRms = FwStats.rmsCi(refMax);
        double[] refSq = squares(refMax);
        Table tab = lab.table("D2-tree", "depth L", "resampling: RMS root max_Q err [95% CI]", "resampling: q95",
                "q95 of eps_l per level", "bound prod(1+q95 eps_l) - 1", "per-run composition violations",
                "per-edge delivered", "forwarding m=" + mNode + ": delivered (pred 0.8^L)", "forwarding: RMS max_Q err",
                "forwarding m/0.8^L: RMS max_Q err [95% CI]");
        boolean boundOk = true;
        double worstPricedZ = 0.0;
        double worstEdgeZ = 0.0;
        long violations = 0;
        double[] lx = new double[depths.length];
        double[][] rootCells = new double[depths.length][];
        for (int di = 0; di < depths.length; di++) {
            int depth = depths[di];
            double[] rootMax = new double[reps];
            double[] edge = new double[reps];
            double[] fwdFrac = new double[reps];
            double[] fwdMax = new double[reps];
            double[] priced = new double[reps];
            double[][] epsL = new double[depth][reps];
            long viol = 0;
            for (int r = 0; r < reps; r++) {
                TreeRun t = res.get(di * reps + r);
                rootMax[r] = t.rootMax();
                edge[r] = t.edgeFrac();
                fwdFrac[r] = t.fwdFrac();
                fwdMax[r] = t.fwdMax();
                priced[r] = t.fwdPricedMax();
                viol += t.composed() ? 0 : 1;
                for (int l = 0; l < depth; l++) {
                    epsL[l][r] = t.epsLevel()[l];
                }
            }
            violations += viol;
            StringBuilder q95 = new StringBuilder();
            double bound = 1.0;
            for (int l = 0; l < depth; l++) {
                double q = FwStats.quantile(epsL[l], 0.95);
                bound *= 1.0 + q;
                q95.append(l == 0 ? "" : ", ").append(f("%.4f", q));
            }
            bound -= 1.0;
            double rootQ95 = FwStats.quantile(rootMax, 0.95);
            boundOk &= rootQ95 <= bound;
            FwStats.Ci edgeCi = FwStats.meanCi(edge);
            worstEdgeZ = Math.max(worstEdgeZ, FwStats.z(edge, 1.0 - rho));
            FwStats.Ci pricedRms = FwStats.rmsCi(priced);
            double[] sq = squares(priced);
            double zms = Math.abs(FwStats.mean(sq) - FwStats.mean(refSq)) / Math.sqrt(
                    Math.pow(FwStats.sd(sq), 2) / sq.length + Math.pow(FwStats.sd(refSq), 2) / refSq.length);
            worstPricedZ = Math.max(worstPricedZ, zms);
            tab.row(depth, ci(FwStats.rmsCi(rootMax), "%.4f"), f("%.4f", rootQ95), q95, f("%.4f", bound), viol,
                    ci(edgeCi, "%.4f"), f("%.4f (%.4f)", FwStats.mean(fwdFrac), Math.pow(1 - rho, depth)),
                    f("%.4f", FwStats.rms(fwdMax)), ci(pricedRms, "%.4f"));
            lx[di] = Math.log(depth);
            rootCells[di] = rootMax;
        }
        FwStats.Ci growth = FwStats.bootstrapLogSlope(lx, rootCells, FwStats::rms, 2000,
                Seeds.of(lab.seed, "D2-boot", 0, 0));
        lab.section("D2. Resampling tree versus forwarding",
                f("Binary tree of depth L over a uniformly random partition of the mixture into 2^L leaves. Every "
                        + "non-root node draws a fresh sensitivity sample of m = %d from its input (leaf block, or "
                        + "union of its children's RC-decoded messages) and sends it, B = 1, over a Bernoulli "
                        + "channel with rho = %.1f; the root outputs the union of its children's decoded messages. "
                        + "eps_l = max over level-l nodes and queries of the node's relative error against its own "
                        + "input. Forwarding: one source sends m (or m/0.8^L) draws through L relays with the same "
                        + "channel. Loss-free reference RMS max_Q error at m = %d: %s. Growth of the resampling RMS "
                        + "with L (log-log slope): %s. %d runs per depth.", mNode, rho, mNode, ci(refRms, "%.4f"),
                        ci(growth, "%.3f"), reps), tab);
        lab.verdict("D2 resampling accuracy", "resampling at each level compounds the error as prod(1+eps_l) - 1",
                f("per-run composition violated in %d/%d runs; q95 root error <= prod(1 + q95 eps_l) - 1 at all L: %b; "
                        + "RMS grows with slope %s in log L", violations, (long) reps * depths.length, boundOk,
                        ci(growth, "%.3f")), violations == 0 && boundOk);
        lab.verdict("D2 resampling loss", "resampling resets the loss: every edge delivers 1-rho = 0.8 whatever the depth",
                f("max |z| of the per-edge delivered fraction against 0.8 over depths = %.2f", worstEdgeZ),
                worstEdgeZ <= 3.0);
        lab.verdict("D2 forwarding price", "forwarding with m/prod(1-rho_h) source draws keeps the loss-free accuracy",
                f("mean square of max_Q error vs loss-free m=%d (RMS %.4f): max |z| over depths = %.2f", mNode,
                        refRms.value(), worstPricedZ), worstPricedZ <= 3.0);
    }

    static double[] squares(double[] x) {
        double[] s = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            s[i] = x[i] * x[i];
        }
        return s;
    }
}
