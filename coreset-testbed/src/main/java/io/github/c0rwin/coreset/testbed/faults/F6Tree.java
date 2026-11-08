package io.github.c0rwin.coreset.testbed.faults;

import static io.github.c0rwin.coreset.testbed.faults.FaultLab.f;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.cell;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.metrics;

import io.github.c0rwin.coreset.testbed.faults.Partitions.Kind;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * F6: aggregation trees with crashing nodes (Propositions F1, F2), scored on exact data so only the topology's loss
 * term is measured. n_b leaves hold a random or projection-sorted partition; every non-root node of an a-ary tree
 * crashes before forwarding with probability f (with checkpointing to r peers it is lost only if all r+1 fail), taking
 * its subtree with it. With internal HT every parent rescales its surviving children by their known subtree sizes;
 * without it only the root rescales by the surviving data size.
 */
final class F6Tree {

    private F6Tree() {
    }

    static void run(FaultLab lab) throws IOException {
        try (Recorder rec = lab.recorder("F6")) {
            Instance inst = lab.instance("zipf");
            int leaves = lab.quick() ? 16 : 64;
            int[] arities = lab.quick() ? new int[] {2, 4, 16} : new int[] {2, 4, 8, 64};
            double[] fails = {0.01, 0.05, 0.1};
            int seeds = lab.seeds(200, 10);
            for (Kind kind : List.of(Kind.RANDOM, Kind.PROJECTION_SORTED)) {
                List<List<Recorder.Row>> out = lab.parallel(seeds, s -> {
                    Placement pl = new Placement(inst, Partitions.split(kind, inst.cluster, inst.points, leaves,
                            LabSeeds.rng(lab.seed, "F6/partition/" + kind.id(), s, 0)));
                    List<Recorder.Row> rows = new ArrayList<>();
                    for (int a : arities) {
                        int depth = depth(leaves, a);
                        for (double fail : fails) {
                            for (int r = 0; r <= 1; r++) {
                                // One crash draw per node and replica, shared by HT on/off.
                                SplittableRandom rng = LabSeeds.rng(lab.seed, "F6/fault/" + kind.id() + "/" + a + "/"
                                        + fail, s, 0);
                                boolean[][] lost = crashes(leaves, a, depth, fail, r, rng);
                                for (boolean ht : new boolean[] {true, false}) {
                                    double[] est = estimate(pl, a, depth, lost, ht);
                                    double alive = est[est.length - 1];
                                    double[] cost = Arrays.copyOf(est, inst.candidates());
                                    double[] rr = inst.ratios(cost, 1.0);
                                    double fe = Math.pow(fail, r + 1);
                                    rows.add(new Recorder.Row(cell("partition", kind.id(), "arity", a, "depth", depth,
                                            "f", fail, "checkpoint", r, "internalHT", ht), s,
                                            metrics("distortion", Measure.distortion(rr), "meanAbsR",
                                                    Measure.meanAbs(rr), "lostFraction", 1.0 - alive / pl.totalSize,
                                                    "predictedLost", 1.0 - Math.pow(1.0 - fe, depth)),
                                            ht && r == 0 && kind == Kind.RANDOM ? rr : null));
                                }
                            }
                        }
                    }
                    return rows;
                });
                out.forEach(rec::addAll);
                lab.log("F6 %s done", kind.id());
            }
            verdicts(lab, rec, inst, arities, leaves);
        }
    }

    /** Levels below the root: leaves sit at depth L = log_a(leaves) (a flat tree has L = 1). */
    static int depth(int leaves, int a) {
        int d = 0;
        for (long span = 1; span < leaves; span *= a) {
            d++;
        }
        return d;
    }

    /** {@code lost[d][i]}: node i at depth d (1..L) crashed, i.e. it and its r checkpoint peers all failed. */
    static boolean[][] crashes(int leaves, int a, int depth, double fail, int r, SplittableRandom rng) {
        boolean[][] lost = new boolean[depth + 1][];
        for (int d = 1; d <= depth; d++) {
            int width = (int) Math.ceil(leaves / Math.pow(a, depth - d));
            lost[d] = new boolean[width];
            for (int i = 0; i < width; i++) {
                boolean all = true;
                for (int c = 0; c <= 1; c++) {
                    boolean down = rng.nextDouble() < fail;
                    all &= c > r || down;
                }
                lost[d][i] = all;
            }
        }
        lost[0] = new boolean[1];
        return lost;
    }

    /** Root estimate: cost per candidate, then the surviving raw data size as the last entry. */
    static double[] estimate(Placement pl, int a, int depth, boolean[][] lost, boolean ht) {
        int cols = pl.cost[0].length;
        int leaves = pl.nodes();
        double[][] est = new double[leaves][];
        double[] size = new double[leaves];
        double[] alive = new double[leaves];
        for (int i = 0; i < leaves; i++) {
            size[i] = pl.size[i];
            if (!lost[depth][i]) {
                est[i] = pl.cost[i].clone();
                alive[i] = pl.size[i];
            }
        }
        for (int d = depth - 1; d >= 0; d--) {
            int width = lost[d].length;
            double[][] nextEst = new double[width][];
            double[] nextSize = new double[width];
            double[] nextAlive = new double[width];
            for (int p = 0; p < width; p++) {
                double[] acc = new double[cols];
                double total = 0.0;
                double got = 0.0;
                double raw = 0.0;
                boolean any = false;
                for (int c = p * a; c < Math.min(est.length, (p + 1) * a); c++) {
                    total += size[c];
                    if (est[c] != null) {
                        any = true;
                        got += size[c];
                        raw += alive[c];
                        for (int j = 0; j < cols; j++) {
                            acc[j] += est[c][j];
                        }
                    }
                }
                nextSize[p] = total;
                if (any && !lost[d][p]) {
                    double scale = ht ? total / got : 1.0;
                    for (int j = 0; j < cols; j++) {
                        acc[j] *= scale;
                    }
                    nextEst[p] = acc;
                    nextAlive[p] = raw;
                }
            }
            est = nextEst;
            size = nextSize;
            alive = nextAlive;
        }
        double[] out = new double[cols + 1];
        if (est[0] != null) {
            double scale = ht ? 1.0 : size[0] / alive[0];
            for (int j = 0; j < cols; j++) {
                out[j] = scale * est[0][j];
            }
            out[cols] = alive[0];
        }
        return out;
    }

    static Map<String, Object> c(String part, int a, int depth, double fail, int r, boolean ht) {
        return cell("partition", part, "arity", a, "depth", depth, "f", fail, "checkpoint", r, "internalHT", ht);
    }

    private static void verdicts(FaultLab lab, Recorder rec, Instance inst, int[] arities, int leaves) {
        StringBuilder md = new StringBuilder("## F6: aggregation tree with crashing nodes (" + leaves
                + " leaves, exact data)\n\n| partition | arity | L | f | checkpoint | internal HT | lost fraction | "
                + "1-(1-f^(r+1))^L | mean abs r | distortion p99 | max |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (Recorder.Cell cl : rec.cells(p -> true)) {
            double[] d = cl.values("distortion");
            md.append(f("| %s | %s | %s | %s | %s | %s | %.4f | %.4f | %.4g | %.4f | %.3g |%n", cl.params.get("partition"),
                    cl.params.get("arity"), cl.params.get("depth"), cl.params.get("f"), cl.params.get("checkpoint"),
                    cl.params.get("internalHT"), Stats.mean(cl.values("lostFraction")),
                    Stats.mean(cl.values("predictedLost")), Stats.mean(cl.values("meanAbsR")), Stats.quantile(d, 0.99),
                    Stats.max(d)));
        }
        md.append('\n');
        lab.section(md.toString());
        StringBuilder lost = new StringBuilder();
        boolean lostOk = true;
        StringBuilder ck = new StringBuilder();
        boolean ckOk = true;
        double[] xs = new double[arities.length];
        double[] ys = new double[arities.length];
        double fMain = 0.05;
        for (int i = 0; i < arities.length; i++) {
            int a = arities[i];
            int depth = depth(leaves, a);
            for (double fail : new double[] {0.01, 0.05, 0.1}) {
                double obs = Stats.mean(rec.values(c("random", a, depth, fail, 0, true), "lostFraction"));
                double exp = 1.0 - Math.pow(1.0 - fail, depth);
                lostOk &= Math.abs(obs / exp - 1.0) <= 0.15;
                if (fail == 0.1) {
                    lost.append(f("a=%d L=%d: %.4f vs %.4f; ", a, depth, obs, exp));
                    double o1 = Stats.mean(rec.values(c("random", a, depth, fail, 1, true), "lostFraction"));
                    double lf2 = depth * fail * fail;
                    ck.append(f("a=%d: %.4f vs L f^2 = %.4f; ", a, o1, lf2));
                    ckOk &= Math.abs(o1 / lf2 - 1.0) <= 0.25;
                }
            }
            xs[i] = Math.log(depth);
            ys[i] = Math.log(Stats.mean(rec.values(c("random", a, depth, fMain, 0, true), "meanAbsR")));
        }
        lab.verdict("F6/F1 lost fraction", "lost fraction = 1-(1-f)^L", lost.toString() + "(all f within 15%: "
                + lostOk + ")", lostOk ? "PASS" : "FAIL");
        Stats.Fit fit = Stats.ols(xs, ys);
        lab.verdict("F6/F1 sqrt(L)", "error proportional to sqrt(L) at fixed f",
                f("f=%.2f, random, internal HT: slope of log mean|r| vs log L = %.2f +/- %.2f", fMain, fit.b(),
                        fit.ci95()),
                Math.abs(fit.b() - 0.5) <= 0.15 ? "PASS" : "FAIL");
        lab.verdict("F6/F2 checkpoint", "checkpoint r=1: loss = L f^2", ck.toString(), ckOk ? "PASS" : "FAIL");
        int a = arities[0];
        int depth = depth(leaves, a);
        double[] rp = rec.values(c("random", a, depth, fMain, 0, true), "distortion");
        double[] pp = rec.values(c("projection-sorted", a, depth, fMain, 0, true), "distortion");
        double[] pc = rec.values(c("projection-sorted", a, depth, fMain, 1, true), "distortion");
        double er = Stats.quantile(rp, 0.99) - 1.0;
        double ep = Stats.quantile(pp, 0.99) - 1.0;
        lab.verdict("F6/F1 projection-sorted", "projection-sorted trees without checkpointing fail catastrophically",
                f("a=%d, f=%.2f: p99 distortion %.3f (projection) vs %.4f (random); max %.3g; P[> 2] = %.3f; with "
                        + "checkpoint r=1 p99 %.4f", a, fMain, 1 + ep, 1 + er, Stats.max(pp), Stats.fraction(pp, 2.0),
                        Stats.quantile(pc, 0.99)),
                ep >= 5 * er && Stats.fraction(pp, 2.0) > 0.0 ? "PASS" : ep >= 5 * er ? "INCONCLUSIVE" : "FAIL");
    }
}
