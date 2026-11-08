package io.github.c0rwin.coreset.testbed.faults;

import static io.github.c0rwin.coreset.testbed.faults.FaultLab.f;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.cell;

import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.testbed.data.Dataset;
import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.faults.NodeFaults.Decoder;
import io.github.c0rwin.coreset.testbed.faults.NodeFaults.Loss;
import io.github.c0rwin.coreset.testbed.faults.Packets.Layout;
import io.github.c0rwin.coreset.testbed.faults.Partitions.Kind;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.ToDoubleFunction;

/**
 * The real-data pass: F1 (exact-data loss and the coreset pipeline), F2, F3 (per-draw versus merged, Bernoulli and
 * bursty, rateless, the B2 formula) and F5 on the z-scored covertype, census1990, kddcup99 and HIGGS, for k in
 * {10, 25} and m in {1000, 4000}. HIGGS runs on a seeded 1M-row subsample, plus one F3 B2 run on all 11M rows;
 * F5 runs on seeded 500k-row subsamples of the three large datasets. Datasets are loaded from the local cache only and
 * released after use.
 */
final class RealData {

    static final List<String> NAMES = List.of("covertype", "census1990", "kddcup99", "higgs");
    static final int[] KS = {10, 25};
    static final int[] MS = {1000, 4000};
    static final int NODES = 16;
    static final int F5_ROWS = 500_000;

    /** {@code <name>[-<rows>]-k<k>}, rows as {@code 500k} or {@code 1m}; rows 0 means the full dataset. */
    record Id(String name, int rows, int k) {

        static Id parse(String id) {
            String[] p = id.split("-");
            int k = Integer.parseInt(p[p.length - 1].substring(1));
            int rows = 0;
            if (p.length == 3) {
                String r = p[1];
                rows = r.endsWith("m") ? Integer.parseInt(r.substring(0, r.length() - 1)) * 1_000_000
                        : Integer.parseInt(r.substring(0, r.length() - 1)) * 1_000;
            }
            return new Id(p[0], rows, k);
        }

        static String of(String name, int rows, int k) {
            String r = rows == 0 ? "" : rows % 1_000_000 == 0 ? "-" + rows / 1_000_000 + "m" : "-" + rows / 1000 + "k";
            return name + r + "-k" + k;
        }
    }

    private RealData() {
    }

    static boolean isReal(String id) {
        String[] p = id.split("-");
        return p.length >= 2 && NAMES.contains(p[0]) && p[p.length - 1].matches("k\\d+");
    }

    static DatasetSpec spec(String name) {
        return new DatasetSpec(name + "-zscore", "public", Map.of("name", name, "variant", "zscore"));
    }

    /** A seeded uniform subsample of {@code rows} rows, kept in file order. */
    static Dataset subsample(Dataset full, String id, int rows, long seed) {
        PointSet pts = full.coreset().pointSet();
        int[] perm = Partitions.permutation(pts.size(), LabSeeds.rng(seed, "subsample/" + Id.parse(id).name(), rows,
                0));
        int[] pick = Arrays.copyOf(perm, rows);
        Arrays.sort(pick);
        double[] w = new double[rows];
        for (int i = 0; i < rows; i++) {
            w[i] = pts.weight(pick[i]);
        }
        List<String> steps = new ArrayList<>(full.preprocessing());
        steps.add("subsample:" + rows + ":seed");
        return Dataset.of(id, PointSet.of(pts.select(pick, w, rows)), full.sha256(), full.license(), full.source(),
                steps, Optional.empty());
    }

    /** Facts about an instance, kept after the instance itself is released. */
    record Facts(String id, int n, int d, int k, double sMax, double kappaMax, int smallest, double prepareS,
                 double[] kappa, double sensTotal) {
    }

    static void run(FaultLab lab) throws IOException {
        List<Facts> facts = new ArrayList<>();
        List<String> main = new ArrayList<>();
        Map<String, String> f5Of = new LinkedHashMap<>();
        try (Recorder r1 = lab.recorder("REAL-F1"); Recorder r2 = lab.recorder("REAL-F2");
             Recorder r3 = lab.recorder("REAL-F3"); Recorder r5 = lab.recorder("REAL-F5")) {
            for (String name : NAMES) {
                for (int k : KS) {
                    String id = Id.of(name, name.equals("higgs") ? 1_000_000 : 0, k);
                    String f5 = name.equals("covertype") ? id : Id.of(name, F5_ROWS, k);
                    main.add(id);
                    f5Of.put(id, f5);
                    facts.add(facts(lab.instance(id)));
                    runMain(lab, id, r1, r2, r3);
                    if (!f5.equals(id)) {
                        lab.drop(id, false);
                        facts.add(facts(lab.instance(f5)));
                    }
                    runF5(lab, f5, r5);
                    lab.drop(f5, false);
                    lab.log("REAL %s done", id);
                }
                lab.drop(Id.of(name, 0, KS[0]), true);
            }
            // One run on all 11M HIGGS rows: per-draw versus merged at a single loss rate.
            String full = Id.of("higgs", 0, 10);
            facts.add(facts(lab.instance(full)));
            F3Erasure.sweep(lab, r3, full, MS, new double[] {0.0, 0.3}, 30);
            lab.drop(full, true);
            verdicts(lab, r1, r2, r3, r5, main, f5Of, full, facts);
        }
    }

    static Facts facts(Instance inst) {
        double kmax = 0.0;
        for (double x : inst.kappa) {
            kmax = Math.max(kmax, x);
        }
        return new Facts(inst.id, inst.n, inst.dim(), inst.k, inst.sMax, kmax, inst.clusterSize[inst.smallestCluster],
                inst.prepareMs / 1e3, inst.kappa.clone(), inst.sensTotal);
    }

    private static void runMain(FaultLab lab, String id, Recorder r1, Recorder r2, Recorder r3) {
        List<Loss> losses = List.of(Loss.fixed(0.125), Loss.fixed(0.25), Loss.fixed(0.5), Loss.iid(0.1),
                Loss.iid(0.3));
        F1NodeLoss.raw(lab, r1, id, List.of(Kind.RANDOM, Kind.CLUSTER_SORTED), new int[] {NODES}, losses,
                List.of(Decoder.REALIZED, Decoder.SIZE), 50);
        F1NodeLoss.pipeline(lab, r1, id, 20, NODES, 500, List.of(Kind.RANDOM, Kind.CLUSTER_SORTED),
                List.of(NodeCoreset.Algo.SENSITIVITY),
                List.of(Loss.none(), Loss.fixed(0.125), Loss.fixed(0.5), Loss.iid(0.3)));
        Instance inst = lab.instance(id);
        lab.parallel(20, s -> F2PostLoss.runSeed(lab, inst, Kind.RANDOM, NODES, 500, 10, s, false))
                .forEach(r2::addAll);
        F3Erasure.sample(lab, r3, id, MS, 100, new F3Erasure.Grid(List.of("bernoulli", "ge-b20"),
                List.of(Layout.PER_DRAW, Layout.MERGED), F3Erasure.RHOS, true, false));
        F3Erasure.sweep(lab, r3, id, MS, new double[] {0.0, 0.3}, 30);
    }

    static final int[] F5_BLOCKS = {1, 8, 16, 32, 64, 128};
    static final int F5_SEEDS = 60;

    private static void runF5(FaultLab lab, String id, Recorder r5) {
        Instance inst = lab.instance(id);
        for (Kind order : List.of(Kind.RANDOM, Kind.CLUSTER_SORTED)) {
            lab.parallel(F5_SEEDS, s -> F5Depth.runSeed(lab, inst, order, F5_BLOCKS, MS, s, List.of("sensitivity")))
                    .forEach(rows -> {
                        for (Recorder.Row row : rows) {
                            Map<String, Object> c = new LinkedHashMap<>();
                            c.put("instance", id);
                            c.putAll(row.cell());
                            r5.add(new Recorder.Row(c, row.seed(), row.metrics(), row.r()));
                        }
                    });
        }
    }

    // ---------------------------------------------------------------- verdicts

    /** One check across instances: its label, prediction, and per instance the measured text and status. */
    private static final class Check {
        final String label;
        final String prediction;
        final Map<String, String[]> cells = new LinkedHashMap<>();

        Check(String label, String prediction) {
            this.label = label;
            this.prediction = prediction;
        }

        void put(FaultLab lab, String id, String measured, String status) {
            cells.put(id, new String[] {measured, status});
            lab.verdict("REAL/" + id + "/" + label, prediction, measured, status);
        }
    }

    private static void verdicts(FaultLab lab, Recorder r1, Recorder r2, Recorder r3, Recorder r5, List<String> main,
                                 Map<String, String> f5Of, String full, List<Facts> facts) {
        Check kappa = new Check("A1 kappa law", "RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate "
                + "(random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2");
        Check shape = new Check("A1 s_max shape", "mean |r| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 "
                + "+/- 10%");
        Check down = new Check("A1 downstream", "downstream ratio < 1.02 at phi = 1/8 (random partition, "
                + "sensitivity m=500/node)");
        Check sorted = new Check("A1' cluster-sorted", "downstream > 2 whenever a lost node held >= 90% of a "
                + "cluster, with that frequency");
        Check a3 = new Check("F2/A3", "coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff "
                + "(ratio 0.67-1.5)");
        Check b1 = new Check("F3/B1 slope", "per-draw + realized decoder: slope of log(distortion-1) vs "
                + "log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap "
                + "CI excludes 0.5; else INCONCLUSIVE)");
        Check rateless = new Check("F3/C3 rateless", "rateless distortion independent of rho (|slope| < 0.1 or CI "
                + "contains 0)");
        Check bursts = new Check("F3/B1 bursts", "GE (mean burst 20) mean excess within 15% of Bernoulli at "
                + "rho = 0.3, per-draw");
        Check b2 = new Check("F3/B2 formula", "erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt "
                + "candidate: split and merged within 35% of Theorem B2");
        Check e1 = new Check("F5/E1 random order", "Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) "
                + "within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed "
                + "bootstrap)");
        Check e1k = new Check("F5/E1 kappa", "Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= "
                + "0.9); coefficient reported");
        int n = NODES;
        for (String id : main) {
            // A1 laws on exact data.
            double[] y = new double[3];
            double[] calib = new double[3];
            double[] phis = {0.125, 0.25, 0.5};
            Facts fx = facts.stream().filter(x -> x.id().equals(id)).findFirst().orElseThrow();
            for (int p = 0; p < 3; p++) {
                Recorder.Cell c = r1.get(F1NodeLoss.rawCell(id, Kind.RANDOM, n, Loss.fixed(phis[p]).id(),
                        Decoder.REALIZED));
                double[] lc = F1NodeLoss.loss(fx.kappa(), fx.n(), c, phis[p]);
                y[p] = lc[0];
                calib[p] = lc[1];
            }
            boolean kOk = true;
            for (double cb : calib) {
                kOk &= cb >= 0.85 && cb <= 1.15;
            }
            kappa.put(lab, id, f("calibration %.3f, %.3f, %.3f", calib[0], calib[1], calib[2]), kOk ? "PASS" : "FAIL");
            double ratio = y[2] / y[0];
            double c8 = y[0] / (Math.sqrt(fx.sMax() * 0.125) / 0.875);
            shape.put(lab, id, f("ratio %.2f (exact law 2.65); c = %.3f at phi=1/8 with s_max/S = %.3g", ratio, c8,
                    fx.sMax()), Math.abs(ratio / 3.5 - 1.0) <= 0.1 ? "PASS" : "FAIL");
            // Coreset pipeline.
            double[] at8 = r1.values(pipe(id, "random", "phi=0.125"), "downstream");
            double[] at0 = r1.values(pipe(id, "random", "none"), "downstream");
            double m8 = Stats.mean(at8);
            double m0 = Stats.mean(at0);
            down.put(lab, id, f("%.4f (no loss %.4f, max %.4f)", m8, m0, Stats.max(at8)),
                    m8 < 1.02 ? "PASS" : m8 - m0 < 0.02 ? "FAIL (baseline above 1.02; loss adds < 0.02)" : "FAIL");
            int whole = 0;
            int wholeBig = 0;
            int notBig = 0;
            int runs = 0;
            for (String loss : List.of("phi=0.125", "phi=0.5", "f=0.3")) {
                Recorder.Cell c = r1.get(pipe(id, "cluster-sorted", loss));
                double[] ds = c.values("downstream");
                double[] wl = c.values("wholeClusterLost");
                for (int i = 0; i < ds.length; i++) {
                    runs++;
                    if (wl[i] > 0) {
                        whole++;
                        wholeBig += ds[i] > 2 ? 1 : 0;
                    } else {
                        notBig += ds[i] > 2 ? 1 : 0;
                    }
                }
            }
            double pW = (double) whole / runs;
            double pBig = (double) (wholeBig + notBig) / runs;
            double cond = whole == 0 ? Double.NaN : (double) wholeBig / whole;
            sorted.put(lab, id, f("P[ds > 2] = %.2f vs P[whole cluster lost] = %.2f; P[> 2 | whole] = %s; max ds "
                            + "%.2f", pBig, pW, whole == 0 ? "n/a" : f("%.2f", cond),
                            maxOver(r1, id, "cluster-sorted")),
                    whole == 0 ? "INCONCLUSIVE (no node holds a whole cluster)"
                            : cond >= 0.8 && Math.abs(pBig - pW) <= 0.1 ? "PASS" : "FAIL");
            // F2.
            int ok = 0;
            int checked = 0;
            double ex = 0.0;
            double pr = 0.0;
            for (Recorder.Cell c : r2.cells(p -> id.equals(p.get("instance")))) {
                double obs = Stats.mean(c.values("lossRate"));
                double exp = Stats.mean(c.values("predictedLossRate"));
                if (exp * n * c.values("lossRate").length >= 30) {
                    checked++;
                    ok += Math.abs(obs / exp - 1.0) <= 0.2 ? 1 : 0;
                }
                if ((Integer) c.params.get("replicas") == 0) {
                    ex += Stats.mean(c.values("excessMS"));
                    pr += Stats.mean(Stats.finite(c.values("predictedExcessMS")));
                }
            }
            a3.put(lab, id, f("%d/%d loss-rate cells within 20%%; error ratio %.2f", ok, checked, ex / pr),
                    ok == checked && checked > 0 && ex / pr >= 0.67 && ex / pr <= 1.5 ? "PASS" : "FAIL");
            // F3.
            double[] s1 = F3Erasure.b1Slope(r3, id, MS[0], "per-draw");
            double[] s4 = F3Erasure.b1Slope(r3, id, MS[1], "per-draw");
            b1.put(lab, id, f("m=1000: %.3f [%.3f, %.3f]; m=4000: %.3f [%.3f, %.3f]", s1[0], s1[1], s1[2], s4[0],
                    s4[1], s4[2]), slopeStatus(s1, 0.5, 0.1, s4));
            double[] rl = F3Erasure.b1Slope(r3, id, MS[0], "rateless");
            rateless.put(lab, id, f("m=1000: slope %.3f [%.3f, %.3f]", rl[0], rl[1], rl[2]),
                    Math.abs(rl[0]) < 0.1 || (rl[1] <= 0 && rl[2] >= 0) ? "PASS" : "FAIL");
            double[] bern = r3.values(F3Erasure.sCell(id, MS[0], "bernoulli", 0.3, "per-draw", "ht-realized"),
                    "distortion");
            double[] ge = r3.values(F3Erasure.sCell(id, MS[0], "ge-b20", 0.3, "per-draw", "ht-realized"),
                    "distortion");
            double rel = (Stats.mean(ge) - 1) / (Stats.mean(bern) - 1);
            bursts.put(lab, id, f("GE/Bernoulli mean excess %.3f; sd %.4f vs %.4f", rel, Stats.sd(ge), Stats.sd(bern)),
                    Math.abs(rel - 1) <= 0.15 ? "PASS" : "FAIL");
            b2.put(lab, id, b2Text(r3, id, fx), b2Status(r3, id, fx));
            // F5 on the matching instance.
            String f5 = f5Of.get(id);
            Facts f5x = facts.stream().filter(x -> x.id().equals(f5)).findFirst().orElseThrow();
            double[] sl = depthSlope(r5, f5, MS[0], f5x);
            double[] sl4 = depthSlope(r5, f5, MS[1], f5x);
            e1.put(lab, id, f("%s: m=1000 slope %.2f [%.2f, %.2f], m=4000 %.2f [%.2f, %.2f]", f5, sl[0], sl[1], sl[2],
                    sl4[0], sl4[1], sl4[2]), slopeStatus(sl, 0.0, 0.3, sl4));
            List<Double> vx = new ArrayList<>();
            List<Double> vy = new ArrayList<>();
            for (String order : List.of("random", "cluster-sorted")) {
                for (int m : MS) {
                    for (int b : F5_BLOCKS) {
                        double[] v = variance(r5, f5, order, m, b, f5x);
                        vx.add(v[1]);
                        vy.add(v[0]);
                    }
                }
            }
            double[] co = Stats.throughOrigin(vx.stream().mapToDouble(Double::doubleValue).toArray(),
                    vy.stream().mapToDouble(Double::doubleValue).toArray());
            e1k.put(lab, id, f("Var = %.3f (S/m) sum kappa, R^2 %.3f", co[0], co[1]), co[1] >= 0.9 ? "PASS" : "FAIL");
        }
        // All 11M HIGGS rows: per-draw versus merged at rho = 0.3.
        Facts hf = facts.stream().filter(x -> x.id().equals(full)).findFirst().orElseThrow();
        StringBuilder hText = new StringBuilder();
        boolean hOk = true;
        for (int m : MS) {
            for (String pk : List.of("per-draw", "merged")) {
                double d0 = Stats.mean(r3.values(F3Erasure.swCell(full, m, 0.0, pk, "ht-realized"), "distortion"));
                double d3 = Stats.mean(r3.values(F3Erasure.swCell(full, m, 0.3, pk, "ht-realized"), "distortion"));
                double ratio = (d3 - 1) / (d0 - 1);
                hText.append(f("m=%d %s: excess x%.3f; ", m, pk, ratio));
                hOk &= Math.abs(ratio - 1 / Math.sqrt(0.7)) <= 0.1;
            }
        }
        lab.verdict("REAL/" + full + "/F3/B1 single rho", "excess distortion x(1-rho)^-1/2 = x1.195 at rho = 0.3, "
                + "per-draw and merged (HIGGS, all 11M rows)", hText.toString(), hOk ? "PASS" : "FAIL");
        lab.verdict("REAL/" + full + "/F3/B2 formula", "erasure RMS at rho = 0.3, m = 4000: split and merged within "
                + "35% of Theorem B2", b2Text(r3, full, hf), b2Status(r3, full, hf));
        section(lab, List.of(kappa, shape, down, sorted, a3, b1, rateless, bursts, b2, e1, e1k), main, facts);
    }

    private static Map<String, Object> pipe(String id, String partition, String loss) {
        return cell("level", "coreset", "instance", id, "partition", partition, "nodes", NODES, "loss", loss,
                "decoder", "ht-size", "algorithm", "sensitivity");
    }

    private static double maxOver(Recorder r1, String id, String partition) {
        double m = 0.0;
        for (String loss : List.of("phi=0.125", "phi=0.5", "f=0.3")) {
            m = Math.max(m, Stats.max(r1.values(pipe(id, partition, loss), "downstream")));
        }
        return m;
    }

    private static int maxKappa(double[] kappa) {
        int j = 0;
        for (int i = 1; i < kappa.length; i++) {
            j = kappa[i] > kappa[j] ? i : j;
        }
        return j;
    }

    private static double[] b2Numbers(Recorder r3, String id, Facts fx) {
        int j = maxKappa(fx.kappa());
        double a = 0.0;
        List<double[]> props = r3.get(cell("part", "proposal", "instance", id)).rs;
        for (double[] v : props) {
            a += v[j] / props.size();
        }
        double kap = fx.kappa()[j];
        double g = 0.3 / 0.7;
        int m = MS[1];
        double sp = Math.sqrt(F3Erasure.erasureMS(r3, id, m, 0.3, "per-draw", "ht-prior", j));
        double me = Math.sqrt(F3Erasure.erasureMS(r3, id, m, 0.3, "merged", "ht-prior", j));
        double spP = Math.sqrt(g * a / m);
        double meP = Math.sqrt(g * ((a - kap) / m + (1.0 - 1.0 / m) * kap));
        return new double[] {sp, spP, me, meP, Math.sqrt(g * kap), kap};
    }

    private static String b2Text(Recorder r3, String id, Facts fx) {
        double[] v = b2Numbers(r3, id, fx);
        return f("split %.4f (B2 %.4f), merged %.4f (B2 %.4f), floor %.4f, kappa_pt %.3g", v[0], v[1], v[2], v[3],
                v[4], v[5]);
    }

    private static String b2Status(Recorder r3, String id, Facts fx) {
        double[] v = b2Numbers(r3, id, fx);
        return Math.abs(v[0] / v[1] - 1) <= 0.35 && Math.abs(v[2] / v[3] - 1) <= 0.35 ? "PASS" : "FAIL";
    }

    /** {mean over candidates of MS r_j, mean over candidates of (S/m) sum kappa_j} for one F5 cell. */
    private static double[] variance(Recorder r5, String id, String order, int m, int b, Facts fx) {
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("instance", id);
        c.putAll(F5Depth.c(order, "sensitivity", m, b));
        Recorder.Cell cl = r5.get(c);
        int cols = fx.kappa().length;
        double ms = 0.0;
        double pred = 0.0;
        for (double[] v : cl.rs) {
            for (int j = 0; j < cols; j++) {
                ms += v[j] * v[j];
                pred += fx.sensTotal() / m * v[cols + j];
            }
        }
        return new double[] {ms / (cl.rs.size() * cols), pred / (cl.rs.size() * cols)};
    }

    /** Slope of log mean-square r against log(L+1) over the random-order cells with L >= 3: {point, lo, hi}. */
    private static double[] depthSlope(Recorder r5, String id, int m, Facts fx) {
        List<List<double[]>> perBlock = new ArrayList<>();
        List<Double> xs = new ArrayList<>();
        for (int b : F5_BLOCKS) {
            int l = Integer.numberOfTrailingZeros(b);
            if (l >= 3) {
                Map<String, Object> c = new LinkedHashMap<>();
                c.put("instance", id);
                c.putAll(F5Depth.c("random", "sensitivity", m, b));
                perBlock.add(r5.get(c).rs);
                xs.add(Math.log(l + 1.0));
            }
        }
        int cols = fx.kappa().length;
        double[] x = xs.stream().mapToDouble(Double::doubleValue).toArray();
        ToDoubleFunction<int[]> slope = idx -> {
            double[] y = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                double s = 0.0;
                for (int k : idx) {
                    double[] v = perBlock.get(i).get(k);
                    for (int j = 0; j < cols; j++) {
                        s += v[j] * v[j];
                    }
                }
                y[i] = Math.log(s / (idx.length * cols));
            }
            return Stats.ols(x, y).b();
        };
        int seeds = perBlock.get(0).size();
        double[] ci = Stats.bootstrap(seeds, slope, 1000, 11L);
        return new double[] {slope.applyAsDouble(Partitions.identity(seeds)), ci[0], ci[1]};
    }

    /** PASS when both point slopes are within tol of target, FAIL when a CI excludes target, else INCONCLUSIVE. */
    private static String slopeStatus(double[] a, double target, double tol, double[] b) {
        boolean within = Math.abs(a[0] - target) <= tol && Math.abs(b[0] - target) <= tol;
        boolean excluded = (a[1] > target || a[2] < target) && Math.abs(a[0] - target) > tol
                || (b[1] > target || b[2] < target) && Math.abs(b[0] - target) > tol;
        return within ? "PASS" : excluded ? "FAIL" : "INCONCLUSIVE";
    }

    private static void section(FaultLab lab, List<Check> checks, List<String> main, List<Facts> facts) {
        StringBuilder md = new StringBuilder("## Real datasets (z-scored; reference = best of 10 Lloyd restarts)\n\n"
                + "| instance | n | d | k | s_max/S | max kappa_pt | smallest cluster | prepare s |\n"
                + "|---|---|---|---|---|---|---|---|\n");
        for (Facts x : facts) {
            md.append(f("| %s | %d | %d | %d | %.3g | %.3g | %d | %.0f |%n", x.id(), x.n(), x.d(), x.k(), x.sMax(),
                    x.kappaMax(), x.smallest(), x.prepareS()));
        }
        md.append("\n| check |");
        for (String id : main) {
            md.append(' ').append(id).append(" |");
        }
        md.append("\n|---|").append("---|".repeat(main.size())).append('\n');
        for (Check c : checks) {
            md.append("| ").append(c.label).append(" |");
            for (String id : main) {
                String[] v = c.cells.get(id);
                md.append(" **").append(v[1]).append("** ").append(v[0].replace("|", "\\|")).append(" |");
            }
            md.append('\n');
        }
        md.append("\nPredictions:\n\n");
        for (Check c : checks) {
            md.append("- ").append(c.label).append(": ").append(c.prediction).append('\n');
        }
        lab.section(md.toString());
    }
}
