package io.github.c0rwin.coreset.testbed.faults;

import static io.github.c0rwin.coreset.testbed.faults.FaultLab.f;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.cell;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.metrics;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import io.github.c0rwin.coreset.testbed.faults.Partitions.Kind;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * F4: stragglers with coded redundancy (C1-C3). n nodes each own one block of a random or cluster-sorted partition;
 * up to s of them never answer. Schemes: uncoded with size-aware HT, block replication on s+1 nodes (what any linear
 * code reduces to, C1), fractional-repetition groups computing one shared-seed coreset of the group's union (C2), and
 * rateless groups of s+mu nodes that draw m/mu i.i.d. points each from one shared proposal and are pooled from any mu
 * answers (C3). Random straggler sets are shared by all schemes of a seed.
 */
final class F4Stragglers {

    private F4Stragglers() {
    }

    /** What the coordinator decoded: cost per candidate, points for the solver, and the traffic. */
    record Outcome(double[] cost, PointSet points, long receivedPoints, long receivedBytes, long sentBytes,
                   long maxLoad, int unitsLost, double drawFraction, double focusMS, double focusFraction) {
    }

    static void run(FaultLab lab) throws IOException {
        try (Recorder rec = lab.recorder("F4")) {
            Instance inst = lab.instance("zipf");
            int n = lab.quick() ? 8 : 16;
            int m = lab.quick() ? 100 : 500;
            int seeds = lab.seeds(30, 3);
            for (Kind kind : List.of(Kind.RANDOM, Kind.CLUSTER_SORTED)) {
                List<List<Recorder.Row>> out = lab.parallel(seeds, s -> runSeed(lab, inst, kind, n, m, s));
                out.forEach(rec::addAll);
                lab.log("F4 %s done", kind.id());
            }
            verdicts(lab, rec, inst, n, m);
        }
    }

    static int[][] groups(int n, int size) {
        int g = Math.max(1, n / size);
        int[][] out = new int[g][];
        for (int i = 0, at = 0; i < g; i++) {
            int len = n / g + (i < n % g ? 1 : 0);
            out[i] = new int[len];
            for (int t = 0; t < len; t++) {
                out[i][t] = at++;
            }
        }
        return out;
    }

    static int[] union(int[][] parts, int[] members) {
        int len = 0;
        for (int v : members) {
            len += parts[v].length;
        }
        int[] rows = new int[len];
        int at = 0;
        for (int v : members) {
            System.arraycopy(parts[v], 0, rows, at, parts[v].length);
            at += parts[v].length;
        }
        Arrays.sort(rows);
        return rows;
    }

    private static List<Recorder.Row> runSeed(FaultLab lab, Instance inst, Kind kind, int n, int m, int s) {
        Placement pl = new Placement(inst, Partitions.split(kind, inst.cluster, inst.points, n,
                F1NodeLoss.partitionRng(lab, inst, kind, n, s)));
        int[][] parts = pl.parts;
        int small = inst.smallestCluster;
        List<NodeCoreset> blocks = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            blocks.add(NodeCoreset.build(inst, NodeCoreset.Algo.SENSITIVITY, parts[v], m,
                    LabSeeds.rng(lab.seed, "F4/block/" + kind.id(), s, v)));
        }
        double[] all = new double[inst.candidates()];
        for (NodeCoreset b : blocks) {
            add(all, b.cost, 1.0);
        }
        double[] r0 = inst.ratios(all, 1.0);
        List<Recorder.Row> rows = new ArrayList<>();
        for (int st = 1; st <= 3; st++) {
            final int sMax = st;
            int[][] fr = groups(n, st + 1);
            List<NodeCoreset> frCoresets = new ArrayList<>();
            for (int g = 0; g < fr.length; g++) {
                frCoresets.add(NodeCoreset.build(inst, NodeCoreset.Algo.SENSITIVITY, union(parts, fr[g]), m,
                        LabSeeds.rng(lab.seed, "F4/fr/" + kind.id() + "/" + st, s, g)));
            }
            // Offline at the group's data size: a fresh partition into as many parts as FR has groups.
            Placement fresh = new Placement(inst, Partitions.split(kind, inst.cluster, inst.points, fr.length,
                    LabSeeds.rng(lab.seed, "F4/fresh/" + kind.id() + "/" + st, s, 0)));
            List<NodeCoreset> freshCoresets = new ArrayList<>();
            for (int g = 0; g < fr.length; g++) {
                freshCoresets.add(NodeCoreset.build(inst, NodeCoreset.Algo.SENSITIVITY, fresh.parts[g], m,
                        LabSeeds.rng(lab.seed, "F4/fresh-sample/" + kind.id() + "/" + st, s, g)));
            }
            for (String pattern : List.of("none", "random-s", "random-s+1", "random-s+2", "adversarial-s",
                    "adversarial-s+1")) {
                int count = switch (pattern) {
                    case "none" -> 0;
                    case "random-s", "adversarial-s" -> sMax;
                    case "random-s+1", "adversarial-s+1" -> sMax + 1;
                    default -> sMax + 2;
                };
                boolean adversarial = pattern.startsWith("adversarial");
                boolean[] randomSet = randomStragglers(n, count, LabSeeds.rng(lab.seed, "F4/stragglers/" + pattern
                        + "/" + kind.id(), s, sMax));
                List<Outcome> outs = new ArrayList<>();
                List<String> names = new ArrayList<>();
                boolean[] set = adversarial ? topHolders(pl, small, count, null) : randomSet;
                names.add("uncoded");
                outs.add(uncoded(pl, blocks, set));
                names.add("replicate");
                outs.add(replicate(pl, blocks, set, sMax));
                set = adversarial ? groupAttack(pl, fr, small, count) : randomSet;
                names.add("fr");
                outs.add(fractional(pl, fr, frCoresets, set, focusGroup(pl, fr, small)));
                for (int mu : new int[] {1, 2, 4}) {
                    int[][] rg = groups(n, sMax + mu);
                    set = adversarial ? groupAttack(pl, rg, small, count) : randomSet;
                    names.add("rateless-mu" + mu);
                    outs.add(rateless(lab, inst, pl, rg, set, m, mu, kind, sMax, s, focusGroup(pl, rg, small)));
                }
                if (pattern.equals("none")) {
                    names.add("offline-g");
                    outs.add(uncoded(fresh, freshCoresets, new boolean[fr.length]));
                }
                for (int i = 0; i < outs.size(); i++) {
                    Outcome o = outs.get(i);
                    double[] r = inst.ratios(o.cost(), 1.0);
                    double excess = 0.0;
                    for (int j = 0; j < r.length; j++) {
                        excess += (r[j] - r0[j]) * (r[j] - r0[j]);
                    }
                    // Solver seed independent of the straggler pattern: identical decoded sets give identical answers.
                    double ds = Measure.downstream(inst, o.points(), LabSeeds.of(lab.seed, "solve/F4/" + kind.id()
                            + "/" + names.get(i), s, sMax));
                    rows.add(new Recorder.Row(cell("partition", kind.id(), "s", sMax, "stragglers", pattern, "scheme",
                            names.get(i)), s,
                            metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r),
                                    "downstream", ds, "excessMS", excess / r.length, "receivedPoints",
                                    o.receivedPoints(), "receivedBytes", o.receivedBytes(), "sentBytes", o.sentBytes(),
                                    "maxNodeLoad", o.maxLoad(), "unitsLost", o.unitsLost(), "drawFraction",
                                    o.drawFraction(), "stragglerCount", count, "focusMS",
                                    Double.isNaN(o.focusMS()) ? null : o.focusMS(), "focusFraction",
                                    Double.isNaN(o.focusFraction()) ? null : o.focusFraction()),
                            names.get(i).equals("uncoded") ? r : null));
                }
            }
        }
        return rows;
    }

    static void add(double[] acc, double[] x, double scale) {
        for (int j = 0; j < acc.length; j++) {
            acc[j] += scale * x[j];
        }
    }

    static boolean[] randomStragglers(int n, int count, SplittableRandom rng) {
        boolean[] out = new boolean[n];
        int[] p = Partitions.permutation(n, rng);
        for (int i = 0; i < Math.min(count, n); i++) {
            out[p[i]] = true;
        }
        return out;
    }

    /** The s nodes holding the most points of the smallest reference cluster (ties: lower id). */
    static boolean[] topHolders(Placement pl, int cluster, int s, int[] restrict) {
        int n = pl.nodes();
        Integer[] order = new Integer[restrict == null ? n : restrict.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = restrict == null ? i : restrict[i];
        }
        Arrays.sort(order, (a, b) -> pl.clusterCount[b][cluster] != pl.clusterCount[a][cluster]
                ? Integer.compare(pl.clusterCount[b][cluster], pl.clusterCount[a][cluster]) : Integer.compare(a, b));
        boolean[] out = new boolean[n];
        for (int i = 0; i < Math.min(s, order.length); i++) {
            out[order[i]] = true;
        }
        return out;
    }

    /** The group holding the most points of the smallest cluster (the adversary's target). */
    static int focusGroup(Placement pl, int[][] groups, int cluster) {
        int best = 0;
        int bestCount = -1;
        for (int g = 0; g < groups.length; g++) {
            int c = 0;
            for (int v : groups[g]) {
                c += pl.clusterCount[v][cluster];
            }
            if (c > bestCount) {
                bestCount = c;
                best = g;
            }
        }
        return best;
    }

    /** Kills s members of the group holding the most of the smallest cluster. */
    static boolean[] groupAttack(Placement pl, int[][] groups, int cluster, int s) {
        return topHolders(pl, cluster, s, groups[focusGroup(pl, groups, cluster)]);
    }

    /** Mean over candidates of the squared relative error of a group's estimate of its own data. */
    static double groupMS(Placement pl, int[] group, double[] est) {
        double[] exact = new double[est.length];
        for (int v : group) {
            add(exact, pl.cost[v], 1.0);
        }
        double s = 0.0;
        for (int j = 0; j < est.length; j++) {
            double r = est[j] / exact[j] - 1.0;
            s += r * r;
        }
        return s / est.length;
    }

    private static Outcome uncoded(Placement pl, List<NodeCoreset> blocks, boolean[] straggler) {
        int n = pl.nodes();
        double alive = pl.aliveSize(straggler);
        double scale = alive > 0 ? pl.totalSize / alive : 0.0;
        double[] cost = new double[blocks.get(0).cost.length];
        List<NodeCoreset> got = new ArrayList<>();
        long pts = 0;
        long bytes = 0;
        long load = 0;
        for (int v = 0; v < n; v++) {
            load = Math.max(load, pl.parts[v].length);
            if (!straggler[v]) {
                got.add(blocks.get(v));
                add(cost, blocks.get(v).cost, scale);
                pts += blocks.get(v).size();
                bytes += blocks.get(v).bytes;
            }
        }
        return new Outcome(cost, NodeCoreset.union(got, scale), pts, bytes, bytes, load, Placement.count(straggler),
                1.0, Double.NaN, Double.NaN);
    }

    /** Block v on nodes v..v+s (mod n); each node sends its s+1 block coresets; the first copy of a block is used. */
    private static Outcome replicate(Placement pl, List<NodeCoreset> blocks, boolean[] straggler, int s) {
        int n = pl.nodes();
        boolean[] lost = new boolean[n];
        long received = 0;
        long load = 0;
        for (int v = 0; v < n; v++) {
            boolean any = false;
            long l = 0;
            for (int c = 0; c <= s; c++) {
                any |= !straggler[(v + c) % n];
                int held = ((v - c) % n + n) % n;
                l += pl.parts[held].length;
                if (!straggler[v]) {
                    received += blocks.get(held).bytes;
                }
            }
            lost[v] = !any;
            load = Math.max(load, l);
        }
        Outcome o = uncoded(pl, blocks, lost);
        return new Outcome(o.cost(), o.points(), o.receivedPoints(), received, received, load, o.unitsLost(), 1.0,
                Double.NaN, Double.NaN);
    }

    private static Outcome fractional(Placement pl, int[][] groups, List<NodeCoreset> coresets,
                                      boolean[] straggler, int focus) {
        double total = 0.0;
        double alive = 0.0;
        boolean[] groupLost = new boolean[groups.length];
        long sent = 0;
        long load = 0;
        for (int g = 0; g < groups.length; g++) {
            boolean any = false;
            double size = 0.0;
            for (int v : groups[g]) {
                any |= !straggler[v];
                size += pl.size[v];
                sent += straggler[v] ? 0 : coresets.get(g).bytes;
            }
            load = Math.max(load, (long) size);
            groupLost[g] = !any;
            total += size;
            alive += any ? size : 0.0;
        }
        double scale = alive > 0 ? total / alive : 0.0;
        double[] cost = new double[coresets.get(0).cost.length];
        List<NodeCoreset> got = new ArrayList<>();
        long pts = 0;
        long bytes = 0;
        for (int g = 0; g < groups.length; g++) {
            if (!groupLost[g]) {
                got.add(coresets.get(g));
                add(cost, coresets.get(g).cost, scale);
                pts += coresets.get(g).size();
                bytes += coresets.get(g).bytes;
            }
        }
        double focusMS = groupLost[focus] ? Double.NaN : groupMS(pl, groups[focus], coresets.get(focus).cost);
        return new Outcome(cost, NodeCoreset.union(got, scale), pts, bytes, sent, load, Placement.count(groupLost),
                1.0, focusMS, groupLost[focus] ? 0.0 : 1.0);
    }

    /** Rateless groups: shared proposal per group, m/mu independent draws per member, first mu answers pooled. */
    private static Outcome rateless(FaultLab lab, Instance inst, Placement pl, int[][] groups, boolean[] straggler,
                                    int m, int mu, Kind kind, int s, int seed, int focus) {
        int cols = inst.candidates();
        int per = (m + mu - 1) / mu;
        double total = 0.0;
        double alive = 0.0;
        long pts = 0;
        long bytes = 0;
        long sent = 0;
        long load = 0;
        int lostGroups = 0;
        long drawsGot = 0;
        long drawsWanted = 0;
        double focusMS = Double.NaN;
        double focusFraction = 0.0;
        List<int[]> rowsList = new ArrayList<>();
        List<double[]> wList = new ArrayList<>();
        double[] costs = new double[cols];
        List<double[]> perGroupCost = new ArrayList<>();
        for (int g = 0; g < groups.length; g++) {
            int[] rows = union(pl.parts, groups[g]);
            double size = 0.0;
            for (int v : groups[g]) {
                size += pl.size[v];
            }
            total += size;
            load = Math.max(load, rows.length);
            String tag = "F4/rateless/" + kind.id() + "/" + s + "/" + mu;
            Sample.Proposal p = Sample.proposal(Sample.input(inst.points, rows, null), rows, inst.k,
                    LabSeeds.rng(lab.seed, tag + "/proposal", seed, g));
            AliasSampler alias = AliasSampler.of(p.s());
            int[] arrival = Partitions.permutation(groups[g].length, LabSeeds.rng(lab.seed, tag + "/arrival", seed, g));
            List<int[]> pooled = new ArrayList<>();
            int answered = 0;
            for (int a : arrival) {
                int v = groups[g][a];
                int[] draws = alias.sortedDraws(per, LabSeeds.rng(lab.seed, tag + "/draw", seed, v));
                Sample own = Sample.merge(p, draws, per);
                long msg = Measure.messageBytes(Coreset.of(own.materialize(inst.points, 1.0)));
                if (straggler[v]) {
                    continue;
                }
                sent += msg;
                if (answered < mu) {
                    pooled.add(draws);
                    bytes += msg;
                    answered++;
                }
            }
            drawsWanted += (long) per * mu;
            if (answered == 0) {
                lostGroups++;
                perGroupCost.add(null);
                continue;
            }
            int got = answered * per;
            drawsGot += got;
            int[] mergedDraws = new int[got];
            int at = 0;
            for (int[] d : pooled) {
                System.arraycopy(d, 0, mergedDraws, at, d.length);
                at += d.length;
            }
            Arrays.sort(mergedDraws);
            Sample smp = Sample.merge(p, mergedDraws, got);
            double[] c = new double[cols];
            smp.addCost(inst.table, 1.0, c);
            perGroupCost.add(c);
            if (g == focus) {
                focusMS = groupMS(pl, groups[g], c);
                focusFraction = (double) got / (per * mu);
            }
            alive += size;
            pts += smp.size();
            double[] w = new double[smp.size()];
            for (int i = 0; i < w.length; i++) {
                w[i] = smp.weight(i);
            }
            rowsList.add(smp.rows);
            wList.add(w);
        }
        double scale = alive > 0 ? total / alive : 0.0;
        for (double[] c : perGroupCost) {
            if (c != null) {
                add(costs, c, scale);
            }
        }
        int len = 0;
        for (int[] r : rowsList) {
            len += r.length;
        }
        int[] rows = new int[len];
        double[] w = new double[len];
        int at = 0;
        for (int i = 0; i < rowsList.size(); i++) {
            for (int t = 0; t < rowsList.get(i).length; t++) {
                rows[at] = rowsList.get(i)[t];
                w[at++] = scale * wList.get(i)[t];
            }
        }
        PointSet pts2 = PointSet.of(inst.points.select(rows, w, len));
        return new Outcome(costs, pts2, pts, bytes, sent, load, lostGroups, (double) drawsGot / drawsWanted, focusMS,
                focusFraction);
    }

    // ---------------------------------------------------------------- verdicts

    static Map<String, Object> c(String part, int s, String pattern, String scheme) {
        return cell("partition", part, "s", s, "stragglers", pattern, "scheme", scheme);
    }

    private static void verdicts(FaultLab lab, Recorder rec, Instance inst, int n, int m) {
        StringBuilder md = new StringBuilder("## F4: stragglers with coding (n = " + n + ", m = " + m + ")\n\n"
                + "| partition | s | stragglers | scheme | distortion mean | p90 | downstream mean | max | received "
                + "points | received bytes | sent bytes | max node load | units lost | draw fraction |\n"
                + "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (Recorder.Cell cl : rec.cells(p -> true)) {
            double[] d = cl.values("distortion");
            double[] ds = cl.values("downstream");
            md.append(f("| %s | %s | %s | %s | %.4f | %.4f | %.4f | %.3f | %.0f | %.0f | %.0f | %.0f | %.2f | %.3f |%n",
                    cl.params.get("partition"), cl.params.get("s"), cl.params.get("stragglers"),
                    cl.params.get("scheme"), Stats.mean(d), Stats.quantile(d, 0.9), Stats.mean(ds), Stats.max(ds),
                    Stats.mean(cl.values("receivedPoints")), Stats.mean(cl.values("receivedBytes")),
                    Stats.mean(cl.values("sentBytes")), Stats.mean(cl.values("maxNodeLoad")),
                    Stats.mean(cl.values("unitsLost")), Stats.mean(cl.values("drawFraction"))));
        }
        md.append('\n');
        lab.section(md.toString());
        StringBuilder bytes = new StringBuilder();
        boolean bytesOk = true;
        StringBuilder eq = new StringBuilder();
        boolean eqOk = true;
        boolean tolerated = true;
        for (int s = 1; s <= 3; s++) {
            double fr = Stats.mean(rec.values(c("random", s, "random-s", "fr"), "receivedPoints"));
            double un = Stats.mean(rec.values(c("random", s, "none", "uncoded"), "receivedPoints"));
            double g = groups(n, s + 1).length;
            bytes.append(f("s=%d: FR %.0f pts vs uncoded %.0f (ratio %.3f, g/n = %.3f); ", s, fr, un, fr / un, g / n));
            bytesOk &= Math.abs((fr / un) / (g / n) - 1.0) <= 0.15;
            double[] a = rec.values(c("random", s, "random-s", "fr"), "distortion");
            double[] b = rec.values(c("random", s, "none", "offline-g"), "distortion");
            double diff = Stats.mean(a) - Stats.mean(b);
            double se = Math.sqrt(Stats.sd(a) * Stats.sd(a) / a.length + Stats.sd(b) * Stats.sd(b) / b.length);
            eq.append(f("s=%d: FR %.4f vs offline-g %.4f (diff %.4f, 2se %.4f); ", s, Stats.mean(a), Stats.mean(b),
                    diff, 2 * se));
            eqOk &= Math.abs(diff) <= 2.5 * se;
            for (String part : List.of("random", "cluster-sorted")) {
                for (String pat : List.of("random-s", "adversarial-s")) {
                    tolerated &= Stats.max(rec.values(c(part, s, pat, "fr"), "unitsLost")) == 0.0;
                }
            }
        }
        lab.verdict("F4/C2 FR bytes", "FR received bytes ~ (n/(s+1)) m, less than uncoded n m", bytes.toString(),
                bytesOk ? "PASS" : "FAIL");
        lab.verdict("F4/C2 FR distortion", "FR distortion equals offline at the group's data size (s stragglers)",
                eq.toString() + (tolerated ? "no FR group lost under any <= s straggler set" : "FR groups lost!"),
                eqOk && tolerated ? "PASS" : "FAIL");
        int s = 2;
        double per1 = Stats.mean(rec.values(c("random", s, "none", "rateless-mu1"), "sentBytes")) / n;
        double per2 = Stats.mean(rec.values(c("random", s, "none", "rateless-mu2"), "sentBytes")) / n;
        double per4 = Stats.mean(rec.values(c("random", s, "none", "rateless-mu4"), "sentBytes")) / n;
        lab.verdict("F4/C3 rateless bytes", "rateless per-node bytes proportional to 1/mu",
                f("s=2, no stragglers: per-node bytes mu=1 %.0f, mu=2 %.0f (x%.2f), mu=4 %.0f (x%.2f)", per1, per2,
                        per2 / per1, per4, per4 / per1),
                Math.abs(per2 / per1 - 0.5) <= 0.1 && Math.abs(per4 / per1 - 0.25) <= 0.07 ? "PASS" : "FAIL");
        // One straggler beyond s inside the attacked group: B1 predicts MS r_G x m/m' for rateless, FR loses it.
        StringBuilder grace = new StringBuilder();
        boolean graceOk = true;
        for (int st = 1; st <= 3; st++) {
            for (int mu : new int[] {2, 4}) {
                double base = 0.0;
                double hit = 0.0;
                double frac = 0.0;
                for (String part : List.of("random", "cluster-sorted")) {
                    base += Stats.mean(rec.values(c(part, st, "none", "rateless-mu" + mu), "focusMS")) / 2;
                    hit += Stats.mean(rec.values(c(part, st, "adversarial-s+1", "rateless-mu" + mu), "focusMS")) / 2;
                    frac += Stats.mean(rec.values(c(part, st, "adversarial-s+1", "rateless-mu" + mu),
                            "focusFraction")) / 2;
                }
                double pred = 1.0 / frac;
                grace.append(f("s=%d mu=%d: group MS x%.2f at m'/m=%.2f (B1: x%.2f); ", st, mu, hit / base, frac,
                        pred));
                graceOk &= Math.abs(hit / base / pred - 1.0) <= 0.3;
            }
        }
        double[] frHit = rec.values(c("cluster-sorted", 1, "adversarial-s+1", "fr"), "downstream");
        double[] rlHit = rec.values(c("cluster-sorted", 1, "adversarial-s+1", "rateless-mu2"), "downstream");
        lab.verdict("F4/C3 degradation", "rateless beyond s stragglers degrades gracefully following B1 (instead of "
                        + "losing the group)",
                grace + f("cluster-sorted s=1, s+1 stragglers in the attacked group: downstream FR %.3f vs "
                        + "rateless mu=2 %.4f", Stats.mean(frHit), Stats.mean(rlHit)),
                graceOk ? "PASS" : "FAIL");
        double ex = 0.0;
        double pr = 0.0;
        for (int st = 1; st <= 3; st++) {
            for (String pat : List.of("random-s", "random-s+1", "random-s+2")) {
                Recorder.Cell cl = rec.get(c("random", st, pat, "uncoded"));
                ex += Stats.mean(cl.values("excessMS"));
                double phi = Stats.mean(cl.values("stragglerCount")) / n;
                double kap = 0.0;
                List<double[]> base = rec.get(c("random", st, "none", "uncoded")).rs;
                for (int j = 0; j < inst.candidates(); j++) {
                    double r0 = 0.0;
                    for (double[] r : base) {
                        r0 += r[j] * r[j];
                    }
                    kap += inst.kappa[j] + r0 / base.size();
                }
                pr += phi / (1.0 - phi) * kap / inst.candidates();
            }
        }
        lab.verdict("F4/A1 uncoded random", "uncoded + HT under random stragglers follows A1",
                f("mean-square straggler perturbation of r(Q) / A1 prediction phi/(1-phi)(kappa_pt + r0^2) = %.2f "
                        + "(pooled s, patterns)",
                        ex / pr),
                ex / pr >= 0.67 && ex / pr <= 1.5 ? "PASS" : "FAIL");
        double[] adv = rec.values(c("cluster-sorted", 1, "adversarial-s", "uncoded"), "downstream");
        double[] frAdv = rec.values(c("cluster-sorted", 1, "adversarial-s", "fr"), "downstream");
        double[] frNone = rec.values(c("cluster-sorted", 1, "none", "fr"), "downstream");
        double moved = 0.0;
        for (int i = 0; i < frAdv.length; i++) {
            moved = Math.max(moved, Math.abs(frAdv[i] - frNone[i]));
        }
        lab.verdict("F4/A1' adversarial straggler", "uncoded + HT fails under the adversarial straggler (holder of the "
                        + "smallest cluster); coded schemes do not",
                f("cluster-sorted, s=1: uncoded downstream mean %.3f (max %.3f); FR %.4f, max |FR(adversarial) - "
                        + "FR(no straggler)| = %.2g over paired seeds", Stats.mean(adv), Stats.max(adv),
                        Stats.mean(frAdv), moved),
                Stats.mean(adv) >= 1.5 && moved <= 1e-9 ? "PASS" : "FAIL");
    }
}
