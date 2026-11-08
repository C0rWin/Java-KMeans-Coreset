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

/**
 * F2: nodes crash after computing their coreset (Proposition A3). Node v's coreset is also held by the next r nodes of
 * a random ring; it reaches the coordinator unless all r+1 holders failed. Every replication level sees the same
 * failures, so the comparison over r is paired.
 */
final class F2PostLoss {

    static final double[] FAILS = {0.05, 0.1, 0.2, 0.3};
    static final int MAX_R = 2;

    private F2PostLoss() {
    }

    /** True when node v's coreset is delivered: some holder in ring positions pos(v) .. pos(v)+r is up. */
    static boolean delivered(int[] ring, int[] pos, int v, int r, boolean[] failed) {
        int n = ring.length;
        for (int c = 0; c <= r; c++) {
            if (!failed[ring[(pos[v] + c) % n]]) {
                return true;
            }
        }
        return false;
    }

    static void run(FaultLab lab) throws IOException {
        try (Recorder rec = lab.recorder("F2")) {
            Instance inst = lab.instance("zipf");
            int seeds = lab.seeds(30, 4);
            int reps = lab.seeds(20, 5);
            int m = lab.quick() ? 100 : 500;
            int[] nodeCounts = lab.quick() ? new int[] {4, 8} : new int[] {16, 64};
            for (int n : nodeCounts) {
                List<Kind> kinds = n == nodeCounts[0] ? List.of(Kind.RANDOM, Kind.CLUSTER_SORTED)
                        : List.of(Kind.RANDOM);
                for (Kind kind : kinds) {
                    List<List<Recorder.Row>> out = lab.parallel(seeds, s -> runSeed(lab, inst, kind, n, m, reps, s,
                            n == nodeCounts[0]));
                    out.forEach(rec::addAll);
                    lab.log("F2 n=%d %s done", n, kind.id());
                }
            }
            verdicts(lab, rec, inst, nodeCounts);
        }
    }

    private static List<Recorder.Row> runSeed(FaultLab lab, Instance inst, Kind kind, int n, int m, int reps, int s,
                                              boolean solve) {
        Placement pl = new Placement(inst, Partitions.split(kind, inst.cluster, inst.points, n,
                F1NodeLoss.partitionRng(lab, inst, kind, n, s)));
        List<NodeCoreset> nodes = new ArrayList<>(n);
        double[] all = new double[inst.candidates()];
        for (int v = 0; v < n; v++) {
            NodeCoreset c = NodeCoreset.build(inst, NodeCoreset.Algo.SENSITIVITY, pl.parts[v], m,
                    LabSeeds.rng(lab.seed, "F2/sample/" + kind.id() + "/" + n, s, v));
            nodes.add(c);
            for (int j = 0; j < all.length; j++) {
                all[j] += c.cost[j];
            }
        }
        double[] r0 = inst.ratios(all, 1.0);
        int[] ring = Partitions.permutation(n, LabSeeds.rng(lab.seed, "F2/ring/" + kind.id() + "/" + n, s, 0));
        int[] pos = new int[n];
        for (int i = 0; i < n; i++) {
            pos[ring[i]] = i;
        }
        List<Recorder.Row> rows = new ArrayList<>();
        for (double fail : FAILS) {
            Loss loss = Loss.iid(fail);
            for (int t = 0; t < reps; t++) {
                boolean[] failed = loss.realize(n, LabSeeds.rng(lab.seed, "F2/fault/" + kind.id() + "/" + n + "/"
                        + loss.id(), s, t));
                for (int r = 0; r <= MAX_R; r++) {
                    boolean[] missing = new boolean[n];
                    double[] cost = new double[all.length];
                    List<NodeCoreset> got = new ArrayList<>();
                    long bytes = 0;
                    for (int v = 0; v < n; v++) {
                        bytes += (long) r * nodes.get(v).bytes;
                        if (delivered(ring, pos, v, r, failed)) {
                            got.add(nodes.get(v));
                            bytes += nodes.get(v).bytes;
                            for (int j = 0; j < cost.length; j++) {
                                cost[j] += nodes.get(v).cost[j];
                            }
                        } else {
                            missing[v] = true;
                        }
                    }
                    int l = Placement.count(missing);
                    double alive = pl.aliveSize(missing);
                    double phi = (double) l / n;
                    double[] rr = inst.ratios(cost, Decoder.SIZE.scale(loss, n, l, pl.totalSize, alive));
                    double excess = 0.0;
                    double predicted = 0.0;
                    for (int j = 0; j < rr.length; j++) {
                        excess += (rr[j] - r0[j]) * (rr[j] - r0[j]);
                        predicted += l == n ? Double.NaN : phi / (1.0 - phi) * (inst.kappa[j] + r0[j] * r0[j]);
                    }
                    Double ds = null;
                    if (solve && t == 0) {
                        ds = Measure.downstream(inst, NodeCoreset.union(got, 1.0), LabSeeds.of(lab.seed,
                                "solve/F2/" + kind.id(), s, (long) (fail * 1000) * 10 + r));
                    }
                    rows.add(new Recorder.Row(cell("instance", inst.id, "partition", kind.id(), "nodes", n, "loss",
                            loss.id(), "replicas", r), s * reps + t,
                            metrics("lossRate", phi, "predictedLossRate", Math.pow(fail, r + 1), "distortion",
                                    Measure.distortion(rr), "meanAbsR", Measure.meanAbs(rr), "excessMS",
                                    excess / rr.length, "predictedExcessMS", predicted / rr.length, "downstream", ds,
                                    "lostFraction", 1.0 - alive / pl.totalSize, "lostSensitivity",
                                    pl.lostSens(missing), "bytes", bytes, "messages", (long) r * n + got.size()),
                            t == 0 ? rr : null));
                }
            }
        }
        return rows;
    }

    private static void verdicts(FaultLab lab, Recorder rec, Instance inst, int[] nodeCounts) {
        StringBuilder md = new StringBuilder("## F2: node crash after compute (coreset message lost), ring "
                + "replication of coresets\n\n| nodes | partition | f | r | coreset loss rate | f^(r+1) | "
                + "excess MS r(Q) | predicted (A1 at phi_eff) | distortion p50 | p99 | downstream mean | max | "
                + "bytes/node |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        int checked = 0;
        int ok = 0;
        StringBuilder worst = new StringBuilder();
        double exSum = 0.0;
        double prSum = 0.0;
        for (Recorder.Cell c : rec.cells(p -> true)) {
            int n = (Integer) c.params.get("nodes");
            double obs = Stats.mean(c.values("lossRate"));
            double exp = Stats.mean(c.values("predictedLossRate"));
            double[] d = c.values("distortion");
            double[] ds = c.values("downstream");
            double ex = Stats.mean(c.values("excessMS"));
            double pr = Stats.mean(Stats.finite(c.values("predictedExcessMS")));
            md.append(f("| %d | %s | %s | %s | %.4f | %.4f | %.3g | %.3g | %.4f | %.4f | %s | %s | %.0f |%n", n,
                    c.params.get("partition"), c.params.get("loss"), c.params.get("replicas"), obs, exp, ex, pr,
                    Stats.quantile(d, 0.5), Stats.quantile(d, 0.99),
                    ds.length == 0 ? "" : f("%.4f", Stats.mean(ds)), ds.length == 0 ? "" : f("%.3f", Stats.max(ds)),
                    Stats.mean(c.values("bytes")) / n));
            double events = exp * n * c.values("lossRate").length;
            if ("random".equals(c.params.get("partition")) && events >= 30) {
                checked++;
                boolean good = Math.abs(obs / exp - 1.0) <= 0.2;
                ok += good ? 1 : 0;
                if (!good) {
                    worst.append(f("n=%d %s r=%s: %.4f vs %.4f; ", n, c.params.get("loss"), c.params.get("replicas"),
                            obs, exp));
                }
            }
            if ("random".equals(c.params.get("partition")) && (Integer) c.params.get("replicas") == 0) {
                exSum += ex;
                prSum += pr;
            }
        }
        md.append('\n');
        lab.section(md.toString());
        lab.verdict("F2/A3 loss rate", "observed coreset loss rate = f^(r+1)",
                f("%d/%d cells (random partition, >= 30 expected losses) within 20%% of f^(r+1)%s", ok, checked,
                        worst.length() == 0 ? "" : "; off: " + worst),
                checked > 0 && ok == checked ? "PASS" : checked == 0 ? "INCONCLUSIVE" : "FAIL");
        double ratio = exSum / prSum;
        lab.verdict("F2/A3 error", "error = A1 at phi_eff (r = 0, random partition)",
                f("mean-square loss perturbation r(Q) - r_no-loss(Q) / phi/(1-phi) (kappa_pt + r0^2) = %.2f (pooled over "
                        + "f and n)", ratio),
                ratio >= 0.67 && ratio <= 1.5 ? "PASS" : "FAIL");
    }
}
