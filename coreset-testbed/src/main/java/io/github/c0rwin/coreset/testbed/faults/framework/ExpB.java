package io.github.c0rwin.coreset.testbed.faults.framework;

import static io.github.c0rwin.coreset.testbed.faults.framework.ExpA.ci;
import static io.github.c0rwin.coreset.testbed.faults.framework.ExpA.col;
import static io.github.c0rwin.coreset.testbed.faults.framework.Lab.f;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * B. Bounded content-correlated loss, rho in [0.1, 0.4] (beta = 0.5, gamma = 0.2). Decoders: RC, midpoint HT
 * (p0 = 0.75), HT with known per-class probabilities, per-class RC from header counts.
 */
final class ExpB {

    static final double P_MIN = 0.6;
    static final double P_MAX = 0.9;
    static final double P_MID = 0.75;
    static final String[] DECODERS = {"RC", "midpoint HT", "known p", "per-class RC"};

    private ExpB() {
    }

    static void run(Lab lab) throws IOException {
        lowerBoundPair(lab);
        rcWorst(lab);
        mixture(lab);
    }

    /** Two locations a (f = 0) and b (f = 1 at Q = {a}), unit weights, uniform sampling, class = location. */
    static CostSpace twoPoint(int atA, int atB) {
        int n = atA + atB;
        double[] table = new double[n];
        int[] cls = new int[n];
        for (int i = atA; i < n; i++) {
            table[i] = 1.0;
            cls[i] = 1;
        }
        double[] ones = new double[n];
        Arrays.fill(ones, 1.0);
        return new CostSpace(CostSpace.identity(n), ones, ones.clone(), table, 1, cls, 2);
    }

    /**
     * Per trial: relative error at every query of the four decoders, {@code [dec * queries + j]}, then M' and the RC
     * cost estimate at query 0.
     */
    static double[] trial(CostSpace sp, double[] p, int m, SplittableRandom sample, SplittableRandom chan) {
        Tally t = new Tally(sp);
        for (int s = 0; s < m; s++) {
            int i = sp.draw(sample);
            t.add(i, chan.nextDouble() < p[sp.cls[i]]);
        }
        int qn = sp.queries;
        double[] out = new double[4 * qn + 2];
        for (int j = 0; j < qn; j++) {
            out[j] = sp.rel(t.rc(j), j);
            out[qn + j] = sp.rel(t.ht(j, P_MID), j);
            out[2 * qn + j] = sp.rel(t.known(j, p), j);
            out[3 * qn + j] = sp.rel(t.perClass(j), j);
        }
        out[4 * qn] = t.received();
        out[4 * qn + 1] = t.rc(0);
        return out;
    }

    private static void lowerBoundPair(Lab lab) throws IOException {
        // r = pmax/pmin = 1.5, n = 300 (n/r = 200): X_A = 300 a + 300 b, X_B = 400 a + 200 b.
        CostSpace[] inputs = {twoPoint(300, 300), twoPoint(400, 200)};
        double x = 1.5 * P_MAX / (2 * 1.5 - 1);
        double[][] p = {{P_MAX, P_MIN}, {x, P_MAX}};
        String[] names = {"X_A", "X_B"};
        double[] pred = {-0.2, 0.2};
        int[] ms = lab.quick() ? new int[] {100, 1000} : new int[] {100, 1000, 10000, 100000};
        int reps = lab.pick(1000, 40);
        List<double[]> res = lab.parallel(2 * ms.length * reps, idx -> {
            int in = idx / (ms.length * reps);
            int mi = idx / reps % ms.length;
            int r = idx % reps;
            return trial(inputs[in], p[in], ms[mi], Seeds.rng(lab.seed, "B1-sample", in * 64L + mi, r),
                    Seeds.rng(lab.seed, "B1-channel", in * 64L + mi, r));
        });
        writeRaw(lab, "B1-pair", res, reps, ms, names, 1);
        Table tab = lab.table("B1-pair", "input", "m", "RC mean err [95% CI]", "midpoint mean err [95% CI]",
                "known-p mean err [95% CI]", "per-class mean err [95% CI]", "known-p RMS", "per-class RMS");
        boolean plateau = true;
        StringBuilder meas = new StringBuilder();
        for (int in = 0; in < 2; in++) {
            for (int mi = 0; mi < ms.length; mi++) {
                int cell = in * ms.length + mi;
                double[][] e = new double[4][];
                for (int dec = 0; dec < 4; dec++) {
                    e[dec] = col(res, cell, reps, dec);
                }
                tab.row(names[in], ms[mi], ci(FwStats.meanCi(e[0]), "%+.4f"), ci(FwStats.meanCi(e[1]), "%+.4f"),
                        ci(FwStats.meanCi(e[2]), "%+.4f"), ci(FwStats.meanCi(e[3]), "%+.4f"),
                        f("%.4f", FwStats.rms(e[2])), f("%.4f", FwStats.rms(e[3])));
                if (mi == ms.length - 1) {
                    for (int dec = 0; dec < 2; dec++) {
                        double mean = FwStats.mean(e[dec]);
                        plateau &= Math.abs(mean - pred[in]) <= 0.005 && FwStats.mean(FwStats.abs(e[dec])) >= 0.19;
                        meas.append(f("%s %s %+.4f; ", names[in], DECODERS[dec], mean));
                    }
                }
            }
        }
        int mSame = ms.length > 1 ? 1 : 0;
        double[] estA = col(res, mSame, reps, 5);
        double[] estB = col(res, ms.length + mSame, reps, 5);
        FwStats.Test ks = FwStats.ks(estA, estB);
        FwStats.Test ad = FwStats.ad(estA, estB, lab.pick(999, 199), Seeds.of(lab.seed, "B1-perm", 0, 0));
        lab.section("B1. The lower-bound pair (rho_min = 0.1, rho_max = 0.4, r = 1.5)",
                f("X_A: 300 points at a, 300 at b, channel delivers a w.p. 0.9 and b w.p. 0.6. X_B: 400 at a, 200 at "
                        + "b, delivers a w.p. %.3f and b w.p. 0.9. Uniform sampling (weight 600/m per record), one "
                        + "record per packet, query Q = {a} (cost 300 on X_A, 200 on X_B). Predicted: RC and midpoint "
                        + "errors -0.2 on X_A and +0.2 on X_B for every m; the decoder input has the same law on both "
                        + "(two-sample tests of the RC cost estimate at m = %d: KS p = %.3f, AD p = %.3f). %d "
                        + "repetitions per cell.", x, ms[mSame], ks.p(), ad.p(), reps), tab);
        lab.verdict("B1 pair plateau", "RC and midpoint decoders plateau at relative error 0.20 on both inputs",
                f("m=%d: %s", ms[ms.length - 1], meas.toString().trim()), plateau);
        lab.verdict("B1 pair same law", "the decoder input has the same law on X_A and X_B",
                f("RC estimate at m=%d, X_A vs X_B: KS p = %.3f, AD p = %.3f", ms[mSame], ks.p(), ad.p()),
                ks.p() > 0.001 && ad.p() > 0.001);
    }

    private static void rcWorst(Lab lab) throws IOException {
        CostSpace sp = twoPoint(100, 1);
        double[] p = {P_MIN, P_MAX};
        double pBar = (100 * P_MIN + P_MAX) / 101.0;
        double predRc = P_MAX / pBar - 1.0;
        double predMid = P_MAX / P_MID - 1.0;
        int[] ms = lab.quick() ? new int[] {100, 1000} : new int[] {100, 1000, 10000, 100000};
        int reps = lab.pick(1000, 40);
        List<double[]> res = lab.parallel(ms.length * reps, idx -> trial(sp, p, ms[idx / reps],
                Seeds.rng(lab.seed, "B2-sample", idx / reps, idx % reps),
                Seeds.rng(lab.seed, "B2-channel", idx / reps, idx % reps)));
        writeRaw(lab, "B2-rc-worst", res, reps, ms, new String[] {"worst"}, 1);
        Table tab = lab.table("B2-rc-worst", "m", "RC mean err [95% CI]", "RC sd", "midpoint mean err [95% CI]",
                "midpoint sd", "known-p mean err [95% CI]", "known-p RMS", "per-class mean err [95% CI]",
                "per-class RMS");
        double[] x = new double[ms.length];
        double[][] known = new double[ms.length][];
        double[][] per = new double[ms.length][];
        double[][] devRc = new double[ms.length][];
        double[][] devMid = new double[ms.length][];
        for (int mi = 0; mi < ms.length; mi++) {
            double[][] e = new double[4][];
            for (int dec = 0; dec < 4; dec++) {
                e[dec] = col(res, mi, reps, dec);
            }
            tab.row(ms[mi], ci(FwStats.meanCi(e[0]), "%+.4f"), f("%.4f", FwStats.sd(e[0])),
                    ci(FwStats.meanCi(e[1]), "%+.4f"), f("%.4f", FwStats.sd(e[1])), ci(FwStats.meanCi(e[2]), "%+.4f"),
                    f("%.4f", FwStats.rms(e[2])), ci(FwStats.meanCi(e[3]), "%+.4f"), f("%.4f", FwStats.rms(e[3])));
            x[mi] = Math.log(ms[mi]);
            known[mi] = e[2];
            per[mi] = e[3];
            devRc[mi] = shift(e[0], predRc);
            devMid[mi] = shift(e[1], predMid);
        }
        int last = ms.length - 1;
        double[] rcLast = col(res, last, reps, 0);
        double[] midLast = col(res, last, reps, 1);
        FwStats.Ci kSlope = FwStats.bootstrapLogSlope(x, known, FwStats::rms, 2000, Seeds.of(lab.seed, "B2-boot", 0, 0));
        FwStats.Ci pSlope = FwStats.bootstrapLogSlope(x, per, FwStats::rms, 2000, Seeds.of(lab.seed, "B2-boot", 1, 0));
        FwStats.Ci rcSlope = FwStats.bootstrapLogSlope(x, devRc, FwStats::rms, 2000, Seeds.of(lab.seed, "B2-boot", 2, 0));
        FwStats.Ci midSlope = FwStats.bootstrapLogSlope(x, devMid, FwStats::rms, 2000,
                Seeds.of(lab.seed, "B2-boot", 3, 0));
        lab.section("B2. The RC-worst instance",
                f("100 unit points at a delivered w.p. 0.6, one point b delivered w.p. 0.9, uniform sampling, Q = {a} "
                        + "(cost 1, carried by b). Predicted limits: RC 0.9/(60.9/101) - 1 = %.4f, midpoint %.4f, known "
                        + "p and per-class 0 at rate m^-1/2. Slopes of log RMS on log m: known p %s, per-class %s; of "
                        + "the RMS distance to the predicted limit: RC %s, midpoint %s. %d repetitions per m.", predRc,
                        predMid, ci(kSlope, "%.3f"), ci(pSlope, "%.3f"), ci(rcSlope, "%.3f"), ci(midSlope, "%.3f"),
                        reps), tab);
        FwStats.Ci rc = FwStats.meanCi(rcLast);
        FwStats.Ci mid = FwStats.meanCi(midLast);
        // Convergence in probability: the mean sits at the limit and the spread around it shrinks like m^-1/2.
        lab.verdict("B2 RC limit", f("RC-worst instance: RC error -> %.2f", predRc),
                f("m=%d: mean %s, sd %.4f; RMS distance to %.4f falls with slope %s in log m", ms[last],
                        ci(rc, "%.4f"), FwStats.sd(rcLast), predRc, ci(rcSlope, "%.3f")),
                FwStats.z(rcLast, predRc) <= 3.0 && Math.abs(rcSlope.value() + 0.5) <= 0.1);
        lab.verdict("B2 midpoint limit", "RC-worst instance: midpoint error -> 0.20",
                f("m=%d: mean %s, sd %.4f; RMS distance to %.4f falls with slope %s in log m", ms[last],
                        ci(mid, "%.4f"), FwStats.sd(midLast), predMid, ci(midSlope, "%.3f")),
                FwStats.z(midLast, predMid) <= 3.0 && Math.abs(midSlope.value() + 0.5) <= 0.1);
        FwStats.Ci kLast = FwStats.meanCi(known[last]);
        FwStats.Ci pLast = FwStats.meanCi(per[last]);
        lab.verdict("B2 known p / per-class", "known p or per-class header counts: error -> 0 at rate m^-1/2",
                f("m=%d: mean err known %s, per-class %s; RMS slope known %s, per-class %s", ms[last],
                        ci(kLast, "%+.4f"), ci(pLast, "%+.4f"), ci(kSlope, "%.3f"), ci(pSlope, "%.3f")),
                FwStats.z(known[last], 0.0) <= 3.0 && FwStats.z(per[last], 0.0) <= 3.0
                        && Math.abs(kSlope.value() + 0.5) <= 0.05 && Math.abs(pSlope.value() + 0.5) <= 0.05);
    }

    static double[] shift(double[] x, double c) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i] - c;
        }
        return y;
    }

    /**
     * Realistic scenario: the mixture, sensitivity sampling, delivery probability by component, decreasing linearly
     * from 0.9 to 0.6 with the distance of the component mean from the origin (a sink at the origin).
     */
    private static void mixture(Lab lab) throws IOException {
        Instance inst = lab.instance();
        CostSpace sp = inst.full(2.0);
        int comps = inst.data.components;
        Integer[] byDist = new Integer[comps];
        double[] dist = new double[comps];
        for (int c = 0; c < comps; c++) {
            byDist[c] = c;
            for (double v : inst.data.means[c]) {
                dist[c] += v * v;
            }
        }
        Arrays.sort(byDist, (a, b) -> Double.compare(dist[a], dist[b]));
        double[] p = new double[comps];
        for (int rank = 0; rank < comps; rank++) {
            p[byDist[rank]] = P_MAX - (P_MAX - P_MIN) * rank / (comps - 1);
        }
        int qn = sp.queries;
        // Exact means given M' >= 1: RC estimates cost of the tilted family (w p / pbar), midpoint sum w p f / p0.
        double pBar = 0.0;
        for (int t = 0; t < sp.size; t++) {
            pBar += sp.q[t] * p[sp.cls[t]];
        }
        double[] predRc = new double[qn];
        double[] predMid = new double[qn];
        double[] tableZ2 = inst.table(2.0);
        for (int t = 0; t < sp.size; t++) {
            double wp = sp.weight[t] * p[sp.cls[t]];
            for (int j = 0; j < qn; j++) {
                double fv = tableZ2[sp.rows[t] * qn + j];
                predRc[j] += wp * fv / pBar;
                predMid[j] += wp * fv / P_MID;
            }
        }
        for (int j = 0; j < qn; j++) {
            predRc[j] = predRc[j] / sp.cost[j] - 1.0;
            predMid[j] = predMid[j] / sp.cost[j] - 1.0;
        }
        int[] ms = lab.quick() ? new int[] {500, 2000} : new int[] {500, 2000, 8000, 32000};
        int reps = lab.pick(400, 20);
        List<double[]> res = lab.parallel(ms.length * reps, idx -> trial(sp, p, ms[idx / reps],
                Seeds.rng(lab.seed, "B3-sample", idx / reps, idx % reps),
                Seeds.rng(lab.seed, "B3-channel", idx / reps, idx % reps)));
        writeRaw(lab, "B3-mixture", res, reps, ms, new String[] {"mixture"}, qn);
        Table tab = lab.table("B3-mixture", "m", "decoder", "max_Q |mean err|", "max_Q |mean - predicted| / se",
                "RMS of max_Q |err| [95% CI]");
        double maxZbiased = 0.0;
        double maxZunbiased = 0.0;
        double[] x = new double[ms.length];
        double[][] perMax = new double[ms.length][];
        for (int mi = 0; mi < ms.length; mi++) {
            x[mi] = Math.log(ms[mi]);
            for (int dec = 0; dec < 4; dec++) {
                double worstBias = 0.0;
                double worstZ = 0.0;
                double[] maxAbs = new double[reps];
                for (int j = 0; j < qn; j++) {
                    double[] e = col(res, mi, reps, dec * qn + j);
                    double pred = dec == 0 ? predRc[j] : dec == 1 ? predMid[j] : 0.0;
                    double mean = FwStats.mean(e);
                    worstBias = Math.max(worstBias, Math.abs(mean));
                    worstZ = Math.max(worstZ, Math.abs(mean - pred) / (FwStats.sd(e) / Math.sqrt(reps)));
                    for (int r = 0; r < reps; r++) {
                        maxAbs[r] = Math.max(maxAbs[r], Math.abs(e[r]));
                    }
                }
                if (dec < 2) {
                    maxZbiased = Math.max(maxZbiased, worstZ);
                } else if (mi > 0) {
                    maxZunbiased = Math.max(maxZunbiased, worstZ);
                }
                if (dec == 3) {
                    perMax[mi] = maxAbs;
                }
                tab.row(ms[mi], DECODERS[dec], f("%.4f", worstBias), f("%.2f", worstZ),
                        ci(FwStats.rmsCi(maxAbs), "%.4f"));
            }
        }
        Table perQ = lab.table("B3-mixture-queries", "query", "predicted RC bias", "measured RC bias [95% CI]",
                "predicted midpoint bias", "measured midpoint bias", "per-class bias [95% CI]");
        int last = ms.length - 1;
        double maxPred = 0.0;
        for (int j = 0; j < qn; j++) {
            maxPred = Math.max(maxPred, Math.max(Math.abs(predRc[j]), Math.abs(predMid[j])));
            perQ.row("Q" + j, f("%+.4f", predRc[j]), ci(FwStats.meanCi(col(res, last, reps, j)), "%+.4f"),
                    f("%+.4f", predMid[j]), ci(FwStats.meanCi(col(res, last, reps, qn + j)), "%+.4f"),
                    ci(FwStats.meanCi(col(res, last, reps, 3 * qn + j)), "%+.4f"));
        }
        FwStats.Ci pSlope = FwStats.bootstrapLogSlope(x, perMax, FwStats::rms, 2000, Seeds.of(lab.seed, "B3-boot", 0, 0));
        lab.section("B3. Content-correlated loss on the Gaussian mixture",
                f("Delivery probability per mixture component, from 0.9 (component mean nearest the origin) to 0.6 "
                        + "(farthest), linear in the rank; the class (component) is readable from the record. "
                        + "Sensitivity sampling, one record per packet, %d repetitions per m. The RC and midpoint "
                        + "biases are predicted exactly (tilted family). Per-class RMS slope on log m: %s. Biases "
                        + "at m = %d per query:", reps, ci(pSlope, "%.3f"), ms[last]), tab);
        lab.section(perQ.markdown());
        lab.verdict("B3 mixture bias", "content-correlated loss on a mixture: RC and midpoint keep a bias that does not "
                        + "shrink with m (equal to the tilted-family bias)",
                f("largest predicted |bias| %.4f; measured vs predicted max |z| = %.2f over %d m x 8 queries x 2 "
                        + "decoders", maxPred, maxZbiased, ms.length), maxZbiased <= 4.0 && maxPred >= 0.01);
        lab.verdict("B3 per-class removes bias", "per-class header counts remove the bias, error ~ m^-1/2",
                f("per-class and known-p max |mean err|/se = %.2f (m >= %d); RMS slope of max_Q error %s",
                        maxZunbiased, ms.length > 1 ? ms[1] : ms[0], ci(pSlope, "%.3f")),
                maxZunbiased <= 4.0 && Math.abs(pSlope.value() + 0.5) <= 0.05);
    }

    private static void writeRaw(Lab lab, String name, List<double[]> res, int reps, int[] ms, String[] inputs,
                                 int qn) throws IOException {
        StringBuilder h = new StringBuilder("input,m,rep,received,rc_estimate_q0");
        for (String dec : new String[] {"rc", "mid", "known", "perclass"}) {
            for (int j = 0; j < qn; j++) {
                h.append(',').append(dec).append("_err_q").append(j);
            }
        }
        try (Writer w = lab.raw(name, h.toString())) {
            for (int idx = 0; idx < res.size(); idx++) {
                double[] v = res.get(idx);
                StringBuilder sb = new StringBuilder();
                sb.append(inputs[idx / (ms.length * reps)]).append(',').append(ms[idx / reps % ms.length]).append(',')
                        .append(idx % reps).append(',').append((long) v[4 * qn]).append(',')
                        .append(f("%.17g", v[4 * qn + 1]));
                for (int k = 0; k < 4 * qn; k++) {
                    sb.append(',').append(f("%.9g", v[k]));
                }
                w.write(sb.append('\n').toString());
            }
        }
    }
}
