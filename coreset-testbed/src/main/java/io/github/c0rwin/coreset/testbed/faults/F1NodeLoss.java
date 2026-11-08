package io.github.c0rwin.coreset.testbed.faults;

import static io.github.c0rwin.coreset.testbed.faults.FaultLab.f;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.cell;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.metrics;

import io.github.c0rwin.coreset.testbed.faults.NodeFaults.Decoder;
import io.github.c0rwin.coreset.testbed.faults.NodeFaults.Loss;
import io.github.c0rwin.coreset.testbed.faults.Partitions.Kind;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * F1: nodes crash before computing their coreset, so their raw data is lost (theory A0, A1, A1', A2).
 *
 * <p>Level "raw" scores the exact union of the surviving parts (epsilon_0 = 0), which isolates the loss term
 * epsilon_1 of Theorem A1 and costs no distance evaluation. Level "coreset" runs the pipeline: every node sends a
 * coreset of its part and the coordinator decodes the surviving ones, scored by distortion and downstream ratio.
 * Both levels see the same partitions and the same lost nodes for a given seed. "replication" places the
 * high-sensitivity points on several nodes (Theorem A2).
 */
final class F1NodeLoss {

    static final double WHOLE = 0.9;
    static final List<Loss> LOSSES = List.of(Loss.fixed(0.125), Loss.fixed(0.25), Loss.fixed(0.5), Loss.iid(0.05),
            Loss.iid(0.1), Loss.iid(0.2), Loss.iid(0.3));
    static final List<Loss> FIXED = LOSSES.subList(0, 3);
    static final double[] TAUS = {Double.POSITIVE_INFINITY, 1e-2, 1e-3, 1e-4};

    private F1NodeLoss() {
    }

    static void run(FaultLab lab) throws IOException {
        try (Recorder rec = lab.recorder("F1")) {
            int seeds = lab.seeds(100, 12);
            List<Decoder> all = List.of(Decoder.values());
            List<Decoder> sized = List.of(Decoder.REALIZED, Decoder.SIZE);
            raw(lab, rec, "zipf", List.of(Kind.RANDOM, Kind.CLUSTER_SORTED, Kind.PROJECTION_SORTED, Kind.ZIPF_SIZES),
                    new int[] {8, 16, 64}, LOSSES, all, seeds);
            raw(lab, rec, "zipf-100k", List.of(Kind.RANDOM), new int[] {16}, FIXED, sized, seeds);
            raw(lab, rec, "zipf-400k", List.of(Kind.RANDOM), new int[] {16}, FIXED, sized, seeds);
            raw(lab, rec, "zipf-heavy", List.of(Kind.RANDOM, Kind.CLUSTER_SORTED), new int[] {16}, LOSSES, sized,
                    seeds);
            raw(lab, rec, "esa22", List.of(Kind.RANDOM, Kind.CLUSTER_SORTED, Kind.INPUT), new int[] {16}, LOSSES,
                    sized, seeds);
            List<Loss> pipeLosses = List.of(Loss.none(), Loss.fixed(0.125), Loss.fixed(0.25), Loss.fixed(0.5),
                    Loss.iid(0.1), Loss.iid(0.3));
            pipeline(lab, rec, "zipf", lab.seeds(30, 4), lab.quick() ? 4 : 16, lab.quick() ? 100 : 500,
                    List.of(Kind.RANDOM, Kind.CLUSTER_SORTED, Kind.ZIPF_SIZES), List.of(NodeCoreset.Algo.values()),
                    pipeLosses);
            List<Loss> repl = List.of(Loss.fixed(0.125), Loss.fixed(0.25), Loss.fixed(0.5), Loss.iid(0.1),
                    Loss.iid(0.3));
            replication(lab, rec, "zipf-heavy", List.of(Kind.RANDOM, Kind.CLUSTER_SORTED), repl, seeds);
            replication(lab, rec, "zipf", List.of(Kind.RANDOM, Kind.CLUSTER_SORTED), repl, seeds);
            replication(lab, rec, "esa22", List.of(Kind.RANDOM, Kind.CLUSTER_SORTED, Kind.INPUT), repl, seeds);
            verdicts(lab, rec);
        }
    }

    static SplittableRandom partitionRng(FaultLab lab, Instance inst, Kind kind, int n, int seed) {
        return LabSeeds.rng(lab.seed, "partition/" + kind.id() + "/" + n, seed, LabSeeds.hash(inst.id));
    }

    static SplittableRandom faultRng(FaultLab lab, Instance inst, Kind kind, int n, Loss loss, int seed) {
        return LabSeeds.rng(lab.seed, "node-fault/" + kind.id() + "/" + n + "/" + loss.id(), seed,
                LabSeeds.hash(inst.id));
    }

    static void raw(FaultLab lab, Recorder rec, String id, List<Kind> kinds, int[] nodes, List<Loss> losses,
                    List<Decoder> decoders, int seeds) {
        Instance inst = lab.instance(id);
        for (Kind kind : kinds) {
            for (int n : nodes) {
                List<List<Recorder.Row>> out = lab.parallel(seeds, s -> {
                    Placement pl = new Placement(inst, Partitions.split(kind, inst.cluster, inst.points, n,
                            partitionRng(lab, inst, kind, n, s)));
                    List<Recorder.Row> rows = new ArrayList<>();
                    for (Loss loss : losses) {
                        boolean[] lost = loss.realize(n, faultRng(lab, inst, kind, n, loss, s));
                        int l = Placement.count(lost);
                        double alive = pl.aliveSize(lost);
                        double[] cost = pl.aliveCost(lost);
                        double clusterLost = pl.maxClusterLost(lost, inst);
                        for (Decoder d : decoders) {
                            double[] r = inst.ratios(cost, d.scale(loss, n, l, pl.totalSize, alive));
                            boolean keep = kind == Kind.RANDOM || kind == Kind.ZIPF_SIZES;
                            rows.add(new Recorder.Row(cell("level", "raw", "instance", id, "partition", kind.id(),
                                    "nodes", n, "loss", loss.id(), "decoder", d.id), s,
                                    metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r),
                                            "lostNodes", l, "lostFraction", 1.0 - alive / pl.totalSize,
                                            "lostSensitivity", pl.lostSens(lost), "maxClusterLost", clusterLost,
                                            "wholeClusterLost", clusterLost >= WHOLE ? 1 : 0),
                                    keep ? r : null));
                        }
                    }
                    return rows;
                });
                out.forEach(rec::addAll);
            }
        }
        lab.log("F1 raw %s done", id);
    }

    /** Level "coreset": every node builds a coreset of its part; lost nodes send nothing. */
    static void pipeline(FaultLab lab, Recorder rec, String id, int seeds, int n, int m, List<Kind> kinds,
                         List<NodeCoreset.Algo> algos, List<Loss> losses) {
        Instance inst = lab.instance(id);
        final int mMean = lab.quick() ? 20 : 50;
        for (Kind kind : kinds) {
            List<List<Recorder.Row>> out = lab.parallel(seeds, s -> {
                Placement pl = new Placement(inst, Partitions.split(kind, inst.cluster, inst.points, n,
                        partitionRng(lab, inst, kind, n, s)));
                boolean[] whole = pl.holdsWholeCluster(inst, WHOLE);
                List<Recorder.Row> rows = new ArrayList<>();
                for (NodeCoreset.Algo algo : algos) {
                    List<NodeCoreset> nodes = new ArrayList<>(n);
                    for (int v = 0; v < n; v++) {
                        SplittableRandom rng = LabSeeds.rng(lab.seed, "node-sample/" + kind.id() + "/" + algo.id(),
                                s, v);
                        nodes.add(NodeCoreset.build(inst, algo, pl.parts[v], algo == NodeCoreset.Algo.MEAN ? mMean
                                : m, rng));
                    }
                    for (Loss loss : losses) {
                        boolean[] lost = loss.realize(n, faultRng(lab, inst, kind, n, loss, s));
                        int l = Placement.count(lost);
                        double alive = pl.aliveSize(lost);
                        double[] cost = new double[inst.candidates()];
                        List<NodeCoreset> survivors = new ArrayList<>();
                        long bytes = 0;
                        int wholeLost = 0;
                        for (int v = 0; v < n; v++) {
                            if (!lost[v]) {
                                survivors.add(nodes.get(v));
                                bytes += nodes.get(v).bytes;
                                for (int j = 0; j < cost.length; j++) {
                                    cost[j] += nodes.get(v).cost[j];
                                }
                            } else if (whole[v]) {
                                wholeLost = 1;
                            }
                        }
                        long solve = LabSeeds.of(lab.seed, "solve/F1/" + kind.id() + "/" + algo.id() + "/"
                                + loss.id(), s, 0);
                        double ds = Measure.downstream(inst, NodeCoreset.union(survivors, 1.0), solve);
                        Double a0 = null;
                        if (s < 10 && l > 0 && l < n) {
                            double scale = Decoder.SIZE.scale(loss, n, l, pl.totalSize, alive);
                            a0 = Measure.downstream(inst, NodeCoreset.union(survivors, scale), solve) - ds;
                        }
                        for (Decoder d : Decoder.values()) {
                            double[] r = inst.ratios(cost, d.scale(loss, n, l, pl.totalSize, alive));
                            boolean main = d == Decoder.SIZE;
                            rows.add(new Recorder.Row(cell("level", "coreset", "instance", id, "partition",
                                    kind.id(), "nodes", n, "loss", loss.id(), "decoder", d.id, "algorithm",
                                    algo.id()), s,
                                    metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r),
                                            "downstream", main ? ds : null, "a0DownstreamDiff", main ? a0 : null,
                                            "lostNodes", l, "lostFraction", 1.0 - alive / pl.totalSize,
                                            "lostSensitivity", pl.lostSens(lost), "wholeClusterLost", wholeLost,
                                            "bytes", bytes, "messages", survivors.size(),
                                            "maxClusterLost", pl.maxClusterLost(lost, inst)),
                                    main ? r : null));
                        }
                    }
                }
                return rows;
            });
            out.forEach(rec::addAll);
            lab.log("F1 pipeline %s %s done", id, kind.id());
        }
    }

    /**
     * Theorem A2: point p with {@code s_p > tau} is stored on {@code r_p = 1 + ceil(log_{1/f}(s_p / tau))} distinct
     * nodes (its own plus random others), the rest once. The decoder deduplicates and rescales each replication class
     * by its realized survival count. Sensitivities are the k-means++ bounds on the full data (an oracle upper bound).
     */
    private static void replication(FaultLab lab, Recorder rec, String id, List<Kind> kinds, List<Loss> losses,
                                    int seeds) {
        Instance inst = lab.instance(id);
        final int n = lab.quick() ? 4 : 16;
        final int cols = inst.candidates();
        List<int[]> heavyByTau = new ArrayList<>();
        for (double tau : TAUS) {
            List<Integer> h = new ArrayList<>();
            for (int r = 0; r < inst.n; r++) {
                if (inst.sens[r] > tau) {
                    h.add(r);
                }
            }
            heavyByTau.add(h.stream().mapToInt(Integer::intValue).toArray());
        }
        for (Kind kind : kinds) {
            List<List<Recorder.Row>> out = lab.parallel(seeds, s -> {
                int[][] parts = Partitions.split(kind, inst.cluster, inst.points, n, partitionRng(lab, inst, kind, n,
                        s));
                int[] owner = Partitions.owner(parts, inst.n);
                List<Recorder.Row> rows = new ArrayList<>();
                for (int t = 0; t < TAUS.length; t++) {
                    double tau = TAUS[t];
                    int[] heavy = heavyByTau.get(t);
                    boolean[] isHeavy = new boolean[inst.n];
                    for (int h : heavy) {
                        isHeavy[h] = true;
                    }
                    double[][] lightCost = new double[n][cols];
                    double[] lightSize = new double[n];
                    double[] lightSens = new double[n];
                    for (int r = 0; r < inst.n; r++) {
                        if (!isHeavy[r]) {
                            double w = inst.points.weight(r);
                            inst.table.addRow(r, w, lightCost[owner[r]]);
                            lightSize[owner[r]] += w;
                            lightSens[owner[r]] += inst.sens[r];
                        }
                    }
                    int[][] replicas = new int[heavy.length][];
                    for (int i = 0; i < heavy.length; i++) {
                        replicas[i] = replicaOrder(lab, owner[heavy[i]], n, s, heavy[i]);
                    }
                    for (Loss loss : losses) {
                        boolean[] lost = loss.realize(n, faultRng(lab, inst, kind, n, loss, s));
                        double fail = loss.prior();
                        double[] classCount = new double[n + 1];
                        double[] classSurv = new double[n + 1];
                        double[][] classCost = new double[n + 1][cols];
                        double lostSize = 0.0;
                        double lostSens = 0.0;
                        long extra = 0;
                        for (int v = 0; v < n; v++) {
                            classCount[1] += lightSize[v];
                            if (lost[v]) {
                                lostSize += lightSize[v];
                                lostSens += lightSens[v];
                            } else {
                                classSurv[1] += lightSize[v];
                                for (int j = 0; j < cols; j++) {
                                    classCost[1][j] += lightCost[v][j];
                                }
                            }
                        }
                        for (int i = 0; i < heavy.length; i++) {
                            int p = heavy[i];
                            int copies = copies(inst.sens[p], tau, fail, n);
                            extra += copies - 1;
                            boolean survives = false;
                            for (int c = 0; c < copies && !survives; c++) {
                                survives = !lost[replicas[i][c]];
                            }
                            double w = inst.points.weight(p);
                            classCount[copies] += w;
                            if (survives) {
                                classSurv[copies] += w;
                                inst.table.addRow(p, w, classCost[copies]);
                            } else {
                                lostSize += w;
                                lostSens += inst.sens[p];
                            }
                        }
                        double[] cost = new double[cols];
                        for (int c = 1; c <= n; c++) {
                            if (classSurv[c] > 0.0) {
                                double factor = classCount[c] / classSurv[c];
                                for (int j = 0; j < cols; j++) {
                                    cost[j] += factor * classCost[c][j];
                                }
                            }
                        }
                        double[] r = inst.ratios(cost, 1.0);
                        double bound = Double.isInfinite(tau) ? 0.0
                                : inst.sensTotal / tau * Math.log(1.0 / tau) / Math.log(1.0 / fail);
                        rows.add(new Recorder.Row(cell("level", "replication", "instance", id, "partition",
                                kind.id(), "nodes", n, "loss", loss.id(), "tau", tauId(tau)), s,
                                metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r),
                                        "heavyPoints", heavy.length, "extraCopies", extra, "extraBound", bound,
                                        "lostFraction", lostSize / inst.n, "lostSensitivity",
                                        lostSens / inst.sensTotal), null));
                    }
                }
                return rows;
            });
            out.forEach(rec::addAll);
        }
        lab.log("F1 replication %s done", id);
    }

    static String tauId(double tau) {
        return Double.isInfinite(tau) ? "none" : f("%.0e", tau);
    }

    static int copies(double s, double tau, double fail, int nodes) {
        if (!(s > tau)) {
            return 1;
        }
        int r = 1 + (int) Math.ceil(Math.log(s / tau) / Math.log(1.0 / fail));
        return Math.min(nodes, Math.max(2, r));
    }

    /** The owner first, then the other nodes in a random order fixed by (seed, point). */
    static int[] replicaOrder(FaultLab lab, int owner, int n, int seed, int point) {
        int[] order = new int[n];
        order[0] = owner;
        for (int v = 0, at = 1; v < n; v++) {
            if (v != owner) {
                order[at++] = v;
            }
        }
        SplittableRandom rng = LabSeeds.rng(lab.seed, "replica", seed, point);
        for (int i = n - 1; i > 1; i--) {
            int j = 1 + rng.nextInt(i);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        return order;
    }

    // ---------------------------------------------------------------- verdicts

    private static void verdicts(FaultLab lab, Recorder rec) {
        StringBuilder md = new StringBuilder("## F1: node crash before compute (raw data lost)\n\n");
        a1Shape(lab, rec, md);
        a0Decoders(lab, rec, md);
        downstream(lab, rec, md);
        sorted(lab, rec, md);
        a2(lab, rec, md);
        lab.section(md.toString());
    }

    static Map<String, Object> rawCell(String inst, Kind kind, int n, String loss, Decoder d) {
        return cell("level", "raw", "instance", inst, "partition", kind.id(), "nodes", n, "loss", loss, "decoder", d.id);
    }

    /** Mean over seeds and candidates of |r|, and the per-candidate RMS divided by sqrt(phi kappa_j / (1 - phi)). */
    static double[] loss(Instance inst, Recorder.Cell c, double phi) {
        return loss(inst.kappa, inst.n, c, phi);
    }

    static double[] loss(double[] kappa, int n, Recorder.Cell c, double phi) {
        int cols = kappa.length;
        double[] ms = new double[cols];
        double abs = 0.0;
        for (double[] r : c.rs) {
            for (int j = 0; j < cols; j++) {
                ms[j] += r[j] * r[j];
                abs += Math.abs(r[j]);
            }
        }
        double calib = 0.0;
        for (int j = 0; j < cols; j++) {
            double pred = Math.sqrt(phi / (1.0 - phi) * (kappa[j] - 1.0 / n) * n / (n - 1.0));
            calib += Math.sqrt(ms[j] / c.rs.size()) / pred;
        }
        return new double[] {abs / (c.rs.size() * cols), calib / cols};
    }

    private static void a1Shape(FaultLab lab, Recorder rec, StringBuilder md) {
        int n = 16;
        double[] phis = {0.125, 0.25, 0.5};
        md.append("### A1: random partition, fixed lost fraction, `ht-realized`\n\n")
                .append("Exact survivors (epsilon_0 = 0). `mean|r|` averages |r(Q)| over seeds and the candidate pool; "
                        + "`calib` is the mean over candidates of RMS r_j / sqrt(phi kappa_pt(Q_j)/(1-phi)) "
                        + "(1 = the exact without-replacement variance).\n\n")
                .append("| instance | nodes | N | s_max/S | phi | mean abs r | x1 = sqrt(s_max phi)/(1-phi) | "
                        + "mean abs r / x1 | calib |\n|---|---|---|---|---|---|---|---|---|\n");
        List<String> ids = List.of("zipf-100k", "zipf", "zipf-400k");
        double[] cByN = new double[ids.size()];
        double[] ratio81 = new double[ids.size()];
        double[] calibAll = new double[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            Instance inst = lab.instance(ids.get(i));
            double[] x1 = new double[3];
            double[] y = new double[3];
            for (int p = 0; p < 3; p++) {
                double phi = phis[p];
                Recorder.Cell c = rec.get(rawCell(inst.id, Kind.RANDOM, n, Loss.fixed(phi).id(), Decoder.REALIZED));
                double[] lc = loss(inst, c, phi);
                y[p] = lc[0];
                x1[p] = Math.sqrt(inst.sMax * phi) / (1.0 - phi);
                calibAll[i] += lc[1] / 3.0;
                md.append(f("| %s | %d | %d | %.3g | %s | %.4g | %.4g | %.3f | %.3f |%n", inst.id, n, inst.n,
                        inst.sMax, Channel.trim(phi), y[p], x1[p], y[p] / x1[p], lc[1]));
            }
            cByN[i] = Stats.throughOrigin(x1, y)[0];
            ratio81[i] = y[2] / y[0];
        }
        // the same at n in {8, 64} on zipf: phi-loss is a uniform sample whatever n, so c should not move.
        Instance z = lab.instance("zipf");
        for (int nn : new int[] {8, 64}) {
            for (double phi : phis) {
                Recorder.Cell c = rec.get(rawCell("zipf", Kind.RANDOM, nn, Loss.fixed(phi).id(), Decoder.REALIZED));
                double[] lc = loss(z, c, phi);
                double x1 = Math.sqrt(z.sMax * phi) / (1.0 - phi);
                md.append(f("| zipf | %d | %d | %.3g | %s | %.4g | %.4g | %.3f | %.3f |%n", nn, z.n, z.sMax,
                        Channel.trim(phi), lc[0], x1, lc[0] / x1, lc[1]));
            }
        }
        double pred1 = (Math.sqrt(0.5) / 0.5) / (Math.sqrt(0.125) / 0.875);
        double pred2 = Math.sqrt(0.5 / 0.5) / Math.sqrt(0.125 / 0.875);
        double meanRatio = Stats.mean(ratio81);
        md.append(f("%nShape: mean|r|(phi=1/2) / mean|r|(phi=1/8) = %.3f (mean over N); linear in sqrt(phi)/(1-phi) "
                + "predicts %.3f, the exact without-replacement law sqrt(phi/(1-phi)) predicts %.3f.%n%n", meanRatio,
                pred1, pred2));
        boolean linear = Math.abs(meanRatio / pred1 - 1.0) <= 0.1;
        lab.verdict("F1/A1 shape", "random partition: mean|r| = c sqrt(s_max phi)/(1-phi), linear in sqrt(phi)/(1-phi)",
                f("mean|r| ratio phi 1/2 : 1/8 = %.2f vs %.2f (linear in sqrt(phi)/(1-phi)) and %.2f (sqrt(phi/(1-phi)));"
                        + " per-candidate RMS / sqrt(phi kappa_pt/(1-phi)) = %.2f-%.2f", meanRatio, pred1, pred2,
                        Stats.min(calibAll), Stats.max(calibAll)),
                linear ? "PASS" : "FAIL");
        double spread = Stats.max(cByN) / Stats.min(cByN);
        double[] nsmax = new double[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            Instance inst = lab.instance(ids.get(i));
            nsmax[i] = inst.n * inst.sMax;
        }
        lab.verdict("F1/A1 N-independence", "c independent of N (N in {1e5, 4e5}; scaled: 1e5, 2e5, 4e5)",
                f("c = %.3f, %.3f, %.3f at N = 100k, 200k, 400k (max/min %.3f; s_max is the max of the k-means++ "
                        + "bound, N s_max/S = %.1f, %.1f, %.1f); with kappa_pt in place of s_max the calibration is "
                        + "%.3f, %.3f, %.3f", cByN[0], cByN[1], cByN[2], spread, nsmax[0], nsmax[1], nsmax[2],
                        calibAll[0], calibAll[1], calibAll[2]),
                spread <= 1.15 ? "PASS" : "FAIL");
    }

    private static void a0Decoders(FaultLab lab, Recorder rec, StringBuilder md) {
        md.append("### A0: decoders (zipf, n = 16, raw level: cost-estimate error)\n\n"
                + "| partition | loss | none mean abs r | ht-prior | ht-realized | ht-size |\n|---|---|---|---|---|---|\n");
        for (Kind kind : List.of(Kind.RANDOM, Kind.ZIPF_SIZES)) {
            for (Loss loss : List.of(Loss.fixed(0.25), Loss.iid(0.1), Loss.iid(0.3))) {
                md.append("| ").append(kind.id()).append(" | ").append(loss.id());
                for (Decoder d : Decoder.values()) {
                    md.append(f(" | %.4f", Stats.mean(rec.values(rawCell("zipf", kind, 16, loss.id(), d), "meanAbsR"))));
                }
                md.append(" |\n");
            }
        }
        double maxRandom = 0.0;
        double maxZipf = 0.0;
        for (Recorder.Cell c : rec.cells(p -> "coreset".equals(p.get("level")) && "ht-size".equals(p.get("decoder")))) {
            double[] d = c.values("a0DownstreamDiff");
            double m = 0.0;
            for (double x : d) {
                m = Math.max(m, Math.abs(x));
            }
            if ("random".equals(c.params.get("partition"))) {
                maxRandom = Math.max(maxRandom, m);
            } else if ("zipf-sizes".equals(c.params.get("partition"))) {
                maxZipf = Math.max(maxZipf, m);
            }
        }
        double noneZ = Stats.mean(rec.values(rawCell("zipf", Kind.ZIPF_SIZES, 16, "f=0.1", Decoder.NONE), "meanAbsR"));
        double sizeZ = Stats.mean(rec.values(rawCell("zipf", Kind.ZIPF_SIZES, 16, "f=0.1", Decoder.SIZE), "meanAbsR"));
        double prioZ = Stats.mean(rec.values(rawCell("zipf", Kind.ZIPF_SIZES, 16, "f=0.1", Decoder.PRIOR), "meanAbsR"));
        md.append(f("%nDownstream ratio with the survivors rescaled by `ht-size` minus unscaled (same solver seed): "
                + "max |diff| = %.2g on random parts, %.2g on zipf-sizes parts.%n%n", maxRandom, maxZipf));
        lab.verdict("F1/A0 equal parts", "none / ht-prior / ht-realized give identical downstream ratios on equal parts",
                f("max |downstream(ht-size) - downstream(none)| = %.2g (random parts)", maxRandom),
                maxRandom < 1e-6 ? "PASS" : "FAIL");
        lab.verdict("F1/A0 zipf-sizes", "decoders differ on zipf-sizes parts, where `none` is biased",
                f("downstream identical across decoders (max |diff| %.2g: one global factor cannot move the argmin); "
                        + "cost estimate differs: mean|r| none %.3f, ht-prior %.3f, ht-size %.3f at f=0.1", maxZipf,
                        noneZ, prioZ, sizeZ),
                maxZipf < 1e-6 && noneZ > 2 * sizeZ ? "FAIL (downstream) / PASS (cost estimate)" : "INCONCLUSIVE");
    }

    static Map<String, Object> coreCell(String part, String loss, String algo) {
        return cell("level", "coreset", "instance", "zipf", "partition", part, "nodes", 16, "loss", loss, "decoder",
                "ht-size", "algorithm", algo);
    }

    private static void downstream(FaultLab lab, Recorder rec, StringBuilder md) {
        int n = lab.quick() ? 4 : 16;
        md.append("### Coreset pipeline (zipf, n = ").append(n).append(", decoder ht-size)\n\n"
                + "| partition | algorithm | loss | distortion mean | p90 | downstream mean | p90 | max | "
                + "P[downstream > 2] | P[whole cluster lost] |\n|---|---|---|---|---|---|---|---|---|---|\n");
        for (Recorder.Cell c : rec.cells(p -> "coreset".equals(p.get("level")) && "ht-size".equals(p.get("decoder")))) {
            double[] d = c.values("distortion");
            double[] ds = c.values("downstream");
            md.append(f("| %s | %s | %s | %.4f | %.4f | %.4f | %.4f | %.3f | %.2f | %.2f |%n", c.params.get("partition"),
                    c.params.get("algorithm"), c.params.get("loss"), Stats.mean(d), Stats.quantile(d, 0.9),
                    Stats.mean(ds), Stats.quantile(ds, 0.9), Stats.max(ds), Stats.fraction(ds, 2.0),
                    Stats.mean(c.values("wholeClusterLost"))));
        }
        md.append('\n');
        if (lab.quick()) {
            return;
        }
        double[] at8 = rec.values(cell("level", "coreset", "instance", "zipf", "partition", "random", "nodes", n,
                "loss", "phi=0.125", "decoder", "ht-size", "algorithm", "sensitivity"), "downstream");
        double[] at0 = rec.values(cell("level", "coreset", "instance", "zipf", "partition", "random", "nodes", n,
                "loss", "none", "decoder", "ht-size", "algorithm", "sensitivity"), "downstream");
        double m8 = Stats.mean(at8);
        double m0 = Stats.mean(at0);
        lab.verdict("F1/A1 downstream", "downstream ratio barely moves: < 1.02 at phi = 1/8 (zipf, random partition)",
                f("sensitivity m=500/node: mean %.4f (p90 %.4f, max %.4f) vs %.4f without loss (loss adds %+.4f)", m8,
                        Stats.quantile(at8, 0.9), Stats.max(at8), m0, m8 - m0),
                m8 < 1.02 ? "PASS" : m8 - m0 < 0.02 ? "FAIL (solver baseline above 1.02; loss adds < 0.02)" : "FAIL");
    }

    private static void sorted(FaultLab lab, Recorder rec, StringBuilder md) {
        if (lab.quick()) {
            return;
        }
        int whole = 0;
        int wholeBig = 0;
        int notWhole = 0;
        int notWholeBig = 0;
        int runs = 0;
        double wholeFreq = 0.0;
        for (String loss : List.of("phi=0.125", "phi=0.25", "phi=0.5", "f=0.1", "f=0.3")) {
            for (String algo : List.of("sensitivity", "uniform", "mean")) {
                Recorder.Cell c = rec.get(coreCell("cluster-sorted", loss, algo));
                double[] ds = c.values("downstream");
                double[] wl = c.values("wholeClusterLost");
                for (int i = 0; i < ds.length; i++) {
                    runs++;
                    if (wl[i] > 0) {
                        whole++;
                        wholeBig += ds[i] > 2.0 ? 1 : 0;
                    } else {
                        notWhole++;
                        notWholeBig += ds[i] > 2.0 ? 1 : 0;
                    }
                }
                wholeFreq += Stats.sum(wl);
            }
        }
        double pBig = (double) (wholeBig + notWholeBig) / runs;
        double pWhole = wholeFreq / runs;
        double cond = whole == 0 ? Double.NaN : (double) wholeBig / whole;
        double condNot = notWhole == 0 ? Double.NaN : (double) notWholeBig / notWhole;
        String measured = f("P[downstream > 2] = %.2f overall vs P[a lost node held >= 90%% of a cluster] = %.2f; "
                + "P[> 2 | whole cluster lost] = %.2f, P[> 2 | not] = %.2f (%d runs)", pBig, pWhole, cond, condNot, runs);
        md.append("Cluster-sorted (A1'): ").append(measured).append("\n\n");
        boolean pass = cond >= 0.8 && Math.abs(pBig - pWhole) <= 0.1;
        lab.verdict("F1/A1' cluster-sorted", "downstream >> 2 whenever a lost node held a whole cluster, "
                + "frequency = fraction of such nodes", measured, pass ? "PASS" : cond >= 0.5 ? "INCONCLUSIVE" : "FAIL");
    }

    private static void a2(FaultLab lab, Recorder rec, StringBuilder md) {
        int n = lab.quick() ? 4 : 16;
        md.append("### A2: sensitivity-aware replication (raw level, realized per-class HT)\n\n"
                + "| instance | partition | loss | tau | distortion p50 | p90 | max | heavy points | extra copies | "
                + "bound (S/tau) log_{1/f}(1/tau) | lost fraction |\n|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (Recorder.Cell c : rec.cells(p -> "replication".equals(p.get("level")))) {
            double[] d = c.values("distortion");
            md.append(f("| %s | %s | %s | %s | %.4f | %.4f | %.3g | %.0f | %.0f | %.3g | %.4f |%n",
                    c.params.get("instance"), c.params.get("partition"), c.params.get("loss"), c.params.get("tau"),
                    Stats.quantile(d, 0.5), Stats.quantile(d, 0.9), Stats.max(d), Stats.mean(c.values("heavyPoints")),
                    Stats.mean(c.values("extraCopies")), Stats.mean(c.values("extraBound")),
                    Stats.mean(c.values("lostFraction"))));
        }
        md.append('\n');
        if (lab.quick()) {
            return;
        }
        String loss = "phi=0.125";
        Map<String, Object> esaIn = cell("level", "replication", "instance", "esa22", "partition", "input", "nodes", n,
                "loss", loss, "tau", "1e-03");
        Map<String, Object> esaRand = cell("level", "replication", "instance", "esa22", "partition", "random", "nodes",
                n, "loss", loss, "tau", "none");
        Map<String, Object> esaInNone = cell("level", "replication", "instance", "esa22", "partition", "input", "nodes",
                n, "loss", loss, "tau", "none");
        double inTau = Stats.quantile(rec.values(esaIn, "distortion"), 0.5);
        double inNone = Stats.quantile(rec.values(esaInNone, "distortion"), 0.5);
        double rand = Stats.quantile(rec.values(esaRand, "distortion"), 0.5);
        double heavy = Stats.mean(rec.values(esaIn, "heavyPoints"));
        Instance esa = lab.instance("esa22");
        lab.verdict("F1/A2 ESA'22", "tau = 1e-3 brings the ESA'22 distortion down to random-partition levels",
                f("input-order partition, phi=1/8: median distortion %.4f without replication, %.4f with tau=1e-3 "
                        + "(%.0f of %d points above tau; mean s_p = S/N = %.2g, so the bound flags no structure), vs "
                        + "%.4f on a random partition", inNone, inTau, heavy, esa.n, esa.sensTotal / esa.n, rand),
                inTau <= 1.0 + 1.1 * (rand - 1.0) ? "PASS" : "FAIL");
        Map<String, Object> hvS = cell("level", "replication", "instance", "zipf-heavy", "partition", "random", "nodes",
                n, "loss", "phi=0.5", "tau", "1e-03");
        Map<String, Object> hvN = cell("level", "replication", "instance", "zipf-heavy", "partition", "random", "nodes",
                n, "loss", "phi=0.5", "tau", "none");
        double withT = Stats.quantile(rec.values(hvS, "distortion"), 0.9);
        double without = Stats.quantile(rec.values(hvN, "distortion"), 0.9);
        md.append(f("zipf-heavy, random partition, phi = 1/2: p90 distortion %.4f without replication, %.4f with "
                + "tau = 1e-3 (%.0f extra copies for %d points).%n%n", without, withT,
                Stats.mean(rec.values(hvS, "extraCopies")), lab.instance("zipf-heavy").n));
        // Extra storage at fixed tau and f versus N: placement is deterministic given the bounds.
        StringBuilder ns = new StringBuilder();
        double[] extra = new double[3];
        double bound = 0.0;
        int i = 0;
        for (String id : List.of("zipf-100k", "zipf", "zipf-400k")) {
            Instance inst = lab.instance(id);
            long e = 0;
            for (int r = 0; r < inst.n; r++) {
                e += copies(inst.sens[r], 1e-3, 0.1, n) - 1;
            }
            extra[i++] = e;
            bound = Math.max(bound, inst.sensTotal / 1e-3 * Math.log(1e3) / Math.log(10.0));
            ns.append(f("N=%d: %d; ", inst.n, e));
        }
        boolean flat = extra[2] <= 1.15 * extra[0] && extra[2] <= bound;
        lab.verdict("F1/A2 storage", "extra stored points <= (S/tau) log_{1/f}(1/tau), flat in N (tau=1e-3, f=0.1)",
                f("extra copies %s bound %.0f", ns, bound), flat ? "PASS" : "FAIL");
    }
}
