package io.github.c0rwin.coreset.testbed.faults;

import static io.github.c0rwin.coreset.testbed.faults.FaultLab.f;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.cell;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.metrics;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.algo.SampledCoreset;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import io.github.c0rwin.coreset.testbed.faults.Partitions.Kind;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * F5: error composition in merge-and-reduce (Theorem E1). The data is streamed in blockCount equal blocks through
 * {@link MergeReduceTree} (each block compressed, then merged binary-counter style, then the final reduce), in random
 * or cluster-sorted order, optionally with per-draw erasures on every transfer. Per candidate, Var r(Q) over seeds is
 * compared with (S/m) sum_l kappa_l(Q) summed over the levels whose reduces actually sample.
 */
final class F5Depth {

    private F5Depth() {
    }

    /** Counts compressions and those that sampled (input larger than m). */
    static final class Counting implements CoresetAlgorithm {
        final CoresetAlgorithm inner;
        final int m;
        int reduces;
        int sampled;

        Counting(CoresetAlgorithm inner, int m) {
            this.inner = inner;
            this.m = m;
        }

        @Override
        public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
            reduces++;
            sampled += input.size() > m ? 1 : 0;
            return inner.compress(input, rng, metrics);
        }
    }

    /**
     * Sensitivity reduce whose output crosses a lossy link: each draw is erased with probability rho and the survivors
     * are reweighted by realized counts ({@code w_j c'_j / (m' q_j)}); an unsampled output loses points instead and is
     * rescaled by its surviving weight. Fault randomness is its own stream, consumed in tree order. The final reduce
     * runs where the root lives, so the caller switches the link off before it.
     */
    static final class LossyLink implements CoresetAlgorithm {
        final SensitivityCoreset inner;
        final double rho;
        final SplittableRandom faults;
        boolean enabled = true;

        LossyLink(SensitivityCoreset inner, double rho, SplittableRandom faults) {
            this.inner = inner;
            this.rho = rho;
            this.faults = faults;
        }

        @Override
        public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
            SampledCoreset sc = inner.compressWithProvenance(input, rng, metrics);
            if (!enabled) {
                return sc.coreset();
            }
            PointSet pts = sc.coreset().pointSet();
            int n = pts.size();
            double[] w = new double[n];
            if (sc.exact()) {
                double total = 0.0;
                double kept = 0.0;
                for (int i = 0; i < n; i++) {
                    total += pts.weight(i);
                    if (faults.nextDouble() >= rho) {
                        w[i] = pts.weight(i);
                        kept += w[i];
                    }
                }
                double scale = kept > 0 ? total / kept : 0.0;
                for (int i = 0; i < n; i++) {
                    w[i] *= scale;
                }
            } else {
                int[] counts = sc.drawCounts();
                int got = 0;
                int[] recv = new int[n];
                for (int i = 0; i < n; i++) {
                    for (int c = 0; c < counts[i]; c++) {
                        recv[i] += faults.nextDouble() >= rho ? 1 : 0;
                    }
                    got += recv[i];
                }
                double scale = got > 0 ? (double) sc.draws() / got : 0.0;
                for (int i = 0; i < n; i++) {
                    w[i] = pts.weight(i) * recv[i] / counts[i] * scale;
                }
            }
            int keep = 0;
            int[] rows = new int[n];
            double[] kw = new double[n];
            for (int i = 0; i < n; i++) {
                if (w[i] > 0.0) {
                    rows[keep] = i;
                    kw[keep++] = w[i];
                }
            }
            return new Coreset(PointSet.of(pts.select(rows, kw, keep)), sc.coreset().offset());
        }
    }

    static void run(FaultLab lab) throws IOException {
        try (Recorder rec = lab.recorder("F5")) {
            Instance inst = lab.instance("zipf");
            int[] blocks = lab.quick() ? new int[] {1, 2, 4, 8} : new int[] {1, 2, 4, 8, 16, 32, 64, 128, 256};
            int[] ms = lab.quick() ? new int[] {50, 100} : new int[] {500, 1000, 4000};
            int seeds = lab.seeds(200, 3);
            for (Kind order : List.of(Kind.RANDOM, Kind.CLUSTER_SORTED)) {
                List<List<Recorder.Row>> out = lab.parallel(seeds, s -> runSeed(lab, inst, order, blocks, ms, s));
                out.forEach(rec::addAll);
                lab.log("F5 %s done", order.id());
            }
            verdicts(lab, rec, inst, blocks, ms);
        }
    }

    /** Input size of a level-l reduce and whether it samples, for N rows in b blocks and budget m. */
    static boolean[] sampledLevels(int n, int b, int m) {
        int levels = Integer.numberOfTrailingZeros(b);
        boolean[] out = new boolean[levels + 1];
        double size = (double) n / b;
        out[0] = size > m;
        double child = Math.min(m, size);
        for (int l = 1; l <= levels; l++) {
            size = 2 * child;
            out[l] = size > m;
            child = Math.min(m, size);
        }
        return out;
    }

    private static List<Recorder.Row> runSeed(FaultLab lab, Instance inst, Kind order, int[] blockCounts, int[] ms,
                                              int s) {
        int[] ord = Partitions.order(order, inst.cluster, inst.points, LabSeeds.rng(lab.seed, "F5/order/" + order.id(),
                s, 0));
        double[] w = new double[ord.length];
        for (int i = 0; i < ord.length; i++) {
            w[i] = inst.points.weight(ord[i]);
        }
        PointSet stream = PointSet.of(inst.points.select(ord, w, ord.length));
        int cols = inst.candidates();
        List<Recorder.Row> rows = new ArrayList<>();
        for (int b : blockCounts) {
            double[][] blockCost = new double[b][cols];
            int[] ends = new int[b];
            for (int i = 0; i < b; i++) {
                int from = i == 0 ? 0 : ends[i - 1];
                ends[i] = (int) ((long) inst.n * (i + 1) / b);
                for (int t = from; t < ends[i]; t++) {
                    inst.table.addRow(ord[t], w[t], blockCost[i]);
                }
            }
            int levels = Integer.numberOfTrailingZeros(b);
            double[][] kappaL = new double[levels + 1][cols];
            for (int l = 0; l <= levels; l++) {
                int span = 1 << l;
                for (int g = 0; g < b; g += span) {
                    for (int j = 0; j < cols; j++) {
                        double c = 0.0;
                        for (int i = g; i < g + span; i++) {
                            c += blockCost[i][j];
                        }
                        double x = c / inst.costFull[j];
                        kappaL[l][j] += x * x;
                    }
                }
            }
            for (int m : ms) {
                boolean[] samples = sampledLevels(inst.n, b, m);
                double[] kappaSum = new double[cols];
                for (int l = 0; l <= levels; l++) {
                    if (samples[l]) {
                        for (int j = 0; j < cols; j++) {
                            kappaSum[j] += kappaL[l][j];
                        }
                    }
                }
                for (String algo : List.of("sensitivity", "uniform", "sensitivity-lossy")) {
                    if (algo.equals("sensitivity-lossy") && m != ms[ms.length > 1 ? 1 : 0]) {
                        continue;
                    }
                    SensitivityCoreset sens = new SensitivityCoreset(inst.k, m, Seeding.kMeansPlusPlus(0));
                    CoresetAlgorithm base = switch (algo) {
                        case "uniform" -> new UniformCoreset(m);
                        case "sensitivity" -> sens;
                        default -> new LossyLink(sens, 0.1, LabSeeds.rng(lab.seed, "F5/fault/" + order.id(), s,
                                (long) b * 100000 + m));
                    };
                    Counting alg = new Counting(base, m);
                    MergeReduceTree tree = new MergeReduceTree(alg, MergeRandomness.derived(LabSeeds.of(lab.seed,
                            "F5/tree/" + order.id() + "/" + algo, s, (long) b * 100000 + m), 0), MetricsSink.NOOP);
                    SplittableRandom leaf = LabSeeds.rng(lab.seed, "F5/leaf/" + order.id() + "/" + algo, s,
                            (long) b * 100000 + m);
                    long t0 = System.nanoTime();
                    for (int i = 0; i < b; i++) {
                        int from = i == 0 ? 0 : ends[i - 1];
                        tree.add(alg.compress(Coreset.of(stream.slice(from, ends[i])), leaf, MetricsSink.NOOP));
                    }
                    if (base instanceof LossyLink link) {
                        link.enabled = false;
                    }
                    Coreset fin = tree.compressedSnapshot();
                    double ms0 = (System.nanoTime() - t0) / 1e6;
                    double[] cost = CostTable.of(fin.pointSet(), inst.candidates, null).weightedSums(fin.pointSet());
                    for (int j = 0; j < cols; j++) {
                        cost[j] += fin.offset();
                    }
                    double[] r = inst.ratios(cost, 1.0);
                    double ks = 0.0;
                    for (double k : kappaSum) {
                        ks += k;
                    }
                    rows.add(new Recorder.Row(cell("order", order.id(), "algorithm", algo, "m", m, "blocks", b,
                            "levels", levels), s,
                            metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r), "size",
                                    fin.size(), "reduces", alg.reduces, "sampledReduces", alg.sampled,
                                    "sampledLevels", count(samples), "kappaSumMean", ks / cols, "wallMs", ms0),
                            concat(r, kappaSum)));
                }
            }
        }
        return rows;
    }

    static int count(boolean[] b) {
        int c = 0;
        for (boolean x : b) {
            c += x ? 1 : 0;
        }
        return c;
    }

    /** r(Q) then sum_l kappa_l(Q) per candidate, in one vector. */
    static double[] concat(double[] a, double[] b) {
        double[] out = new double[a.length + b.length];
        System.arraycopy(a, 0, out, 0, a.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    // ---------------------------------------------------------------- verdicts

    static Map<String, Object> c(String order, String algo, int m, int b) {
        return cell("order", order, "algorithm", algo, "m", m, "blocks", b, "levels", Integer.numberOfTrailingZeros(b));
    }

    /** {mean over candidates of MS r_j, mean over candidates of predicted (S/m) sum kappa_j}. */
    static double[] variance(Recorder rec, Instance inst, String order, String algo, int m, int b) {
        Recorder.Cell cl = rec.get(c(order, algo, m, b));
        int cols = inst.candidates();
        double ms = 0.0;
        double pred = 0.0;
        for (double[] v : cl.rs) {
            for (int j = 0; j < cols; j++) {
                ms += v[j] * v[j];
                pred += inst.sensTotal / m * v[cols + j];
            }
        }
        return new double[] {ms / (cl.rs.size() * cols), pred / (cl.rs.size() * cols)};
    }

    private static void verdicts(FaultLab lab, Recorder rec, Instance inst, int[] blocks, int[] ms) {
        StringBuilder md = new StringBuilder("## F5: merge-reduce depth\n\n`MS r` = mean over candidates of the "
                + "mean-square r(Q) over seeds (the reduces are unbiased, so this is the variance); `E1 bound` = mean "
                + "over candidates of (S/m) sum over sampling levels of kappa_l(Q), S = " + f("%.1f", inst.sensTotal)
                + ".\n\n| order | algorithm | m | blocks | L | sampled reduces | sampled levels | distortion mean | "
                + "MS r | E1 bound | MS / bound |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (Recorder.Cell cl : rec.cells(p -> true)) {
            String order = (String) cl.params.get("order");
            String algo = (String) cl.params.get("algorithm");
            int m = (Integer) cl.params.get("m");
            int b = (Integer) cl.params.get("blocks");
            double[] v = variance(rec, inst, order, algo, m, b);
            md.append(f("| %s | %s | %d | %d | %d | %.1f | %.0f | %.4f | %.3g | %.3g | %.3f |%n", order, algo, m, b,
                    Integer.numberOfTrailingZeros(b), Stats.mean(cl.values("sampledReduces")),
                    Stats.mean(cl.values("sampledLevels")), Stats.mean(cl.values("distortion")), v[0], v[1],
                    v[0] / v[1]));
        }
        md.append('\n');
        // Depth law: slope of log MS r against log(L + 1) over L >= 3, per order (sensitivity).
        StringBuilder slopes = new StringBuilder();
        double randomSlope = 0.0;
        double sortedSlope = 0.0;
        double sortedBoundSlope = 0.0;
        int mMain = ms[0];
        for (String order : List.of("random", "cluster-sorted")) {
            for (int m : ms) {
                List<Double> xs = new ArrayList<>();
                List<Double> ys = new ArrayList<>();
                List<Double> bs = new ArrayList<>();
                for (int b : blocks) {
                    int l = Integer.numberOfTrailingZeros(b);
                    if (l >= (lab.quick() ? 1 : 3)) {
                        double[] v = variance(rec, inst, order, "sensitivity", m, b);
                        xs.add(Math.log(l + 1.0));
                        ys.add(Math.log(v[0]));
                        bs.add(Math.log(v[1]));
                    }
                }
                double[] x = xs.stream().mapToDouble(Double::doubleValue).toArray();
                Stats.Fit fit = Stats.ols(x, ys.stream().mapToDouble(Double::doubleValue).toArray());
                Stats.Fit bound = Stats.ols(x, bs.stream().mapToDouble(Double::doubleValue).toArray());
                slopes.append(f("%s m=%d: %.2f +/- %.2f (sum kappa: %.2f); ", order, m, fit.b(), fit.ci95(),
                        bound.b()));
                if (m == mMain) {
                    if (order.equals("random")) {
                        randomSlope = fit.b();
                    } else {
                        sortedSlope = fit.b();
                        sortedBoundSlope = bound.b();
                    }
                }
            }
        }
        md.append("Slope of log MS r vs log(L+1), L >= 3 (sensitivity): ").append(slopes).append("\n\n");
        // Coefficient of MS r on (S/m) sum kappa over every sensitivity cell.
        List<Double> xs = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        for (String order : List.of("random", "cluster-sorted")) {
            for (int m : ms) {
                for (int b : blocks) {
                    double[] v = variance(rec, inst, order, "sensitivity", m, b);
                    if (v[1] > 0) {
                        xs.add(v[1]);
                        ys.add(v[0]);
                    }
                }
            }
        }
        double[] co = Stats.throughOrigin(xs.stream().mapToDouble(Double::doubleValue).toArray(),
                ys.stream().mapToDouble(Double::doubleValue).toArray());
        md.append(f("Through-origin fit MS r = c (S/m) sum kappa over all sensitivity cells: c = %.3f (R^2 %.3f).%n%n",
                co[0], co[1]));
        lab.section(md.toString());
        lab.verdict("F5/E1 random order", "random order: Var r(Q) flat in L beyond ~3 levels (depth-free)",
                f("sensitivity m=%d: slope of log Var vs log(L+1) = %.2f (all: %s)", mMain, randomSlope, slopes),
                Math.abs(randomSlope) <= 0.3 ? "PASS" : "FAIL");
        lab.verdict("F5/E1 cluster-sorted", "cluster-sorted: Var grows linearly in L+1 (sqrt L in RMS)",
                f("sensitivity m=%d: slope %.2f (1 = linear in L+1); the E1 bound (S/m) sum_l kappa_l on this order "
                        + "grows with slope %.2f", mMain, sortedSlope, sortedBoundSlope),
                Math.abs(sortedSlope - 1.0) <= 0.3 ? "PASS" : sortedSlope > 0.3 ? "INCONCLUSIVE" : "FAIL");
        lab.verdict("F5/E1 coefficient", "fitted coefficient of Var on sum_l kappa_l = S/m (c = 1 in units of S/m)",
                f("c = %.3f (R^2 %.3f): Var follows sum_l kappa_l but at %.3f S/m; E1 is an upper bound with slack "
                        + "S_v <= S and sum cost^2/q <= S cost^2", co[0], co[1], co[0]),
                co[0] >= 0.5 && co[0] <= 1.5 ? "PASS" : co[0] < 0.5 ? "FAIL (bound holds, not tight)" : "FAIL");
        double lossy = variance(rec, inst, "random", "sensitivity-lossy", ms[ms.length > 1 ? 1 : 0],
                blocks[blocks.length - 1])[0];
        double clean = variance(rec, inst, "random", "sensitivity", ms[ms.length > 1 ? 1 : 0],
                blocks[blocks.length - 1])[0];
        md.setLength(0);
        lab.verdict("F5/E1(iv) per-level erasure", "per-level erasures multiply level terms by 1/(1-rho_l) "
                        + "(rho_l = 0.1 on every transfer, realized-count decoding)",
                f("random order, m=%d, %d blocks: Var ratio lossy/clean = %.3f (1/(1-rho) = %.3f)",
                        ms[ms.length > 1 ? 1 : 0], blocks[blocks.length - 1], lossy / clean, 1 / 0.9),
                Math.abs(lossy / clean - 1.0 / 0.9) <= 0.25 ? "PASS" : "FAIL");
    }
}
