package io.github.c0rwin.coreset.testbed.faults;

import static io.github.c0rwin.coreset.testbed.faults.FaultLab.f;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.cell;
import static io.github.c0rwin.coreset.testbed.faults.Recorder.metrics;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.algo.Clustering;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import io.github.c0rwin.coreset.testbed.faults.Packets.Layout;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * F3: packet erasures between one node and the coordinator (Lemma B1, Theorem B2, Proposition B4). A sensitivity
 * coreset is drawn on the full data, packetized, sent over a Bernoulli or Gilbert-Elliott channel and decoded with no,
 * prior or realized-count rescaling. The rateless variant keeps sending fresh i.i.d. draws until m arrived (one ACK).
 * All packetizations and loss rates of a seed share the sample and the channel's random stream. The mean coreset is
 * tested with its global offset, with clustering-feature triples, split cells, replicated cells and ideal MDS FEC.
 */
final class F3Erasure {

    static final double[] RHOS = {0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5};
    static final String[] CHANNELS = {"bernoulli", "ge-b1", "ge-b5", "ge-b20"};
    static final String[] DECODERS = {"none", "ht-prior", "ht-realized"};
    static final double DELTA = 0.01;
    static final double ETA = 0.1;

    private F3Erasure() {
    }

    static Channel channel(String id, double rho) {
        return switch (id) {
            case "bernoulli" -> Channel.bernoulli(rho);
            case "ge-b1" -> Channel.gilbertElliott(rho, 1);
            case "ge-b5" -> Channel.gilbertElliott(rho, 5);
            case "ge-b20" -> Channel.gilbertElliott(rho, 20);
            default -> throw new IllegalArgumentException(id);
        };
    }

    static void run(FaultLab lab) throws IOException {
        try (Recorder rec = lab.recorder("F3")) {
            int[] ms = lab.quick() ? new int[] {100, 200} : new int[] {1000, 4000};
            sample(lab, rec, "zipf", ms, lab.seeds(100, 6));
            sample(lab, rec, "zipf-heavy", ms, lab.seeds(60, 4));
            sample(lab, rec, "esa22", ms, lab.seeds(60, 4));
            int[] sweep = lab.quick() ? new int[] {50, 100, 200} : new int[] {250, 500, 1000, 2000, 4000, 8000};
            for (String id : List.of("zipf", "zipf-heavy", "esa22")) {
                sweep(lab, rec, id, sweep, lab.seeds(100, 6));
            }
            mean(lab, rec, "zipf", lab.seeds(10, 2), lab.seeds(20, 5));
            mean(lab, rec, "zipf-heavy", lab.seeds(10, 2), lab.seeds(20, 5));
            verdicts(lab, rec, ms, sweep);
        }
    }

    static Sample.Proposal proposal(FaultLab lab, Instance inst, int s) {
        return Sample.proposal(inst.points, Partitions.identity(inst.n), inst.k,
                LabSeeds.rng(lab.seed, "F3/proposal", s, LabSeeds.hash(inst.id)));
    }

    /** Per-draw cost on every candidate of each item: {@code unit_i * cost(row_i, j)}, row-major. */
    static double[] unitCosts(Instance inst, Sample smp) {
        int cols = inst.candidates();
        double[] u = new double[smp.size() * cols];
        for (int i = 0; i < smp.size(); i++) {
            int o = smp.rows[i] * cols;
            for (int j = 0; j < cols; j++) {
                u[i * cols + j] = smp.unit[i] * inst.table.cost[o + j];
            }
        }
        return u;
    }

    static double[] receivedCost(int[] recv, double[] unitCost, int cols) {
        double[] c = new double[cols];
        for (int i = 0; i < recv.length; i++) {
            if (recv[i] > 0) {
                for (int j = 0; j < cols; j++) {
                    c[j] += recv[i] * unitCost[i * cols + j];
                }
            }
        }
        return c;
    }

    static double scale(String decoder, double rho, int m, int got) {
        if (got == 0) {
            return 0.0;
        }
        return switch (decoder) {
            case "none" -> 1.0;
            case "ht-prior" -> 1.0 / (1.0 - rho);
            case "ht-realized" -> (double) m / got;
            default -> throw new IllegalArgumentException(decoder);
        };
    }

    private static void sample(FaultLab lab, Recorder rec, String id, int[] ms, int seeds) {
        Instance inst = lab.instance(id);
        final int cols = inst.candidates();
        final long pointBytes = Measure.pointBytes(inst.dim());
        final int dsSeeds = lab.seeds(20, 2);
        final boolean solveHere = !id.equals("zipf-heavy");
        List<List<Recorder.Row>> out = lab.parallel(seeds, s -> {
            List<Recorder.Row> rows = new ArrayList<>();
            Sample.Proposal p = proposal(lab, inst, s);
            AliasSampler alias = AliasSampler.of(p.s());
            for (int m : ms) {
                SplittableRandom drng = LabSeeds.rng(lab.seed, "F3/draws/" + m, s, LabSeeds.hash(id));
                int[] seq = new int[3 * m];
                for (int i = 0; i < seq.length; i++) {
                    seq[i] = alias.next(drng);
                }
                int[] first = Arrays.copyOf(seq, m);
                Arrays.sort(first);
                Sample smp = Sample.merge(p, first, m);
                double[] unit = unitCosts(inst, smp);
                List<Packets> layouts = new ArrayList<>();
                for (Layout l : Layout.values()) {
                    layouts.add(Packets.of(l, smp, inst.cluster, LabSeeds.rng(lab.seed, "F3/pack/" + l.id(), s, m)));
                }
                int[] recv = new int[smp.size()];
                boolean[] lost = new boolean[3 * m];
                for (String ch : CHANNELS) {
                    for (double rho : RHOS) {
                        Channel channel = channel(ch, rho);
                        for (Packets pk : layouts) {
                            channel.erase(lost, pk.packets(), LabSeeds.rng(lab.seed, "F3/fault/" + ch, s, m));
                            int got = pk.receive(lost, recv);
                            double[] cost = receivedCost(recv, unit, cols);
                            int missing = clustersMissing(inst, smp, recv);
                            for (String dec : DECODERS) {
                                if (!ch.equals("bernoulli") && !dec.equals("ht-realized")) {
                                    continue;
                                }
                                double sc = scale(dec, rho, m, got);
                                double[] r = inst.ratios(cost, sc);
                                Double ds = null;
                                boolean realized = dec.equals("ht-realized");
                                if (realized && solveHere && s < dsSeeds && m == ms[0]
                                        && (ch.equals("bernoulli") || ch.equals("ge-b20"))
                                        && pk.layout != Layout.RANDOM_B8 && pk.layout != Layout.RANDOM_B64) {
                                    ds = Measure.downstream(inst, received(inst, smp, recv, sc), LabSeeds.of(lab.seed,
                                            "solve/F3", s, m));
                                }
                                rows.add(new Recorder.Row(cell("part", "sample", "instance", id, "m", m, "channel", ch,
                                        "rho", rho, "packetization", pk.layout.id(), "decoder", dec), s,
                                        metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r),
                                                "mPrime", got, "lostDrawFraction", 1.0 - (double) got / m,
                                                "packets", pk.packets(), "bytes", pk.bytes(pointBytes),
                                                "clustersMissing", missing, "downstream", ds),
                                        realized && ch.equals("bernoulli") ? r : null));
                            }
                        }
                        rows.add(rateless(lab, inst, p, seq, m, ch, rho, s, pointBytes, solveHere && s < dsSeeds
                                && m == ms[0] && (ch.equals("bernoulli") || ch.equals("ge-b20"))));
                    }
                }
            }
            return rows;
        });
        out.forEach(rec::addAll);
        lab.log("F3 sample %s done", id);
    }

    /** Sends the i.i.d. draw sequence until m draws arrived; the receiver weights them 1/(m' q). */
    private static Recorder.Row rateless(FaultLab lab, Instance inst, Sample.Proposal p, int[] seq, int m, String ch,
                                         double rho, int s, long pointBytes, boolean solve) {
        int cols = inst.candidates();
        boolean[] lost = new boolean[seq.length];
        channel(ch, rho).erase(lost, seq.length, LabSeeds.rng(lab.seed, "F3/fault-rateless/" + ch, s, m));
        int got = 0;
        int sent = 0;
        double[] acc = new double[cols];
        int[] taken = new int[m];
        while (sent < seq.length && got < m) {
            if (!lost[sent]) {
                int g = seq[sent];
                inst.table.addRow(p.rows()[g], p.inWeight()[g] / p.q(g), acc);
                taken[got++] = g;
            }
            sent++;
        }
        double[] r = inst.ratios(acc, got == 0 ? 0.0 : 1.0 / got);
        Double ds = null;
        if (solve && got > 0) {
            int[] sorted = Arrays.copyOf(taken, got);
            Arrays.sort(sorted);
            Sample smp = Sample.merge(p, sorted, got);
            ds = Measure.downstream(inst, PointSet.of(smp.materialize(inst.points, 1.0)),
                    LabSeeds.of(lab.seed, "solve/F3", s, m));
        }
        return new Recorder.Row(cell("part", "sample", "instance", inst.id, "m", m, "channel", ch, "rho", rho,
                "packetization", "rateless", "decoder", "ht-realized"), s,
                metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r), "mPrime", got,
                        "lostDrawFraction", 1.0 - (double) got / m, "packets", sent, "bytes",
                        sent * (Measure.PACKET_HEADER + pointBytes), "downstream", ds),
                ch.equals("bernoulli") ? r : null);
    }

    static int clustersMissing(Instance inst, Sample smp, int[] recv) {
        boolean[] seen = new boolean[inst.clusterSize.length];
        for (int i = 0; i < recv.length; i++) {
            if (recv[i] > 0) {
                seen[inst.cluster[smp.rows[i]]] = true;
            }
        }
        int missing = 0;
        for (int c = 0; c < seen.length; c++) {
            missing += inst.clusterSize[c] > 0 && !seen[c] ? 1 : 0;
        }
        return missing;
    }

    static PointSet received(Instance inst, Sample smp, int[] recv, double scale) {
        int n = 0;
        for (int x : recv) {
            n += x > 0 ? 1 : 0;
        }
        int[] rows = new int[n];
        double[] w = new double[n];
        for (int i = 0, at = 0; i < recv.length; i++) {
            if (recv[i] > 0) {
                rows[at] = smp.rows[i];
                w[at++] = scale * recv[i] * smp.unit[i];
            }
        }
        return PointSet.of(inst.points.select(rows, w, n));
    }

    /** Theorem B2: merged versus split packets as m grows, Bernoulli loss, prior and realized decoders. */
    private static void sweep(FaultLab lab, Recorder rec, String id, int[] ms, int seeds) {
        Instance inst = lab.instance(id);
        final int cols = inst.candidates();
        double[] rhos = {0.0, 0.1, 0.3};
        List<List<Recorder.Row>> out = lab.parallel(seeds, s -> {
            List<Recorder.Row> rows = new ArrayList<>();
            Sample.Proposal p = proposal(lab, inst, s);
            AliasSampler alias = AliasSampler.of(p.s());
            for (int m : ms) {
                Sample smp = Sample.merge(p, alias.sortedDraws(m, LabSeeds.rng(lab.seed, "F3/sweep-draws", s, m)), m);
                double[] unit = unitCosts(inst, smp);
                int[] recv = new int[smp.size()];
                for (Layout l : List.of(Layout.PER_DRAW, Layout.MERGED)) {
                    Packets pk = Packets.of(l, smp, inst.cluster, LabSeeds.rng(lab.seed, "F3/sweep-pack", s, m));
                    boolean[] lost = new boolean[pk.packets()];
                    for (double rho : rhos) {
                        Channel.bernoulli(rho).erase(lost, pk.packets(), LabSeeds.rng(lab.seed, "F3/sweep-fault", s,
                                m));
                        int got = pk.receive(lost, recv);
                        double[] cost = receivedCost(recv, unit, cols);
                        for (String dec : List.of("ht-prior", "ht-realized")) {
                            double[] r = inst.ratios(cost, scale(dec, rho, m, got));
                            rows.add(new Recorder.Row(cell("part", "sweep", "instance", id, "m", m, "rho", rho,
                                    "packetization", l.id(), "decoder", dec), s,
                                    metrics("distortion", Measure.distortion(r), "meanAbsR", Measure.meanAbs(r),
                                            "mPrime", got, "distinct", smp.size()), r));
                        }
                    }
                }
            }
            return rows;
        });
        out.forEach(rec::addAll);
        lab.log("F3 sweep %s done", id);
    }

    /** Proposition B4: the mean coreset under cell loss, and its fixes. */
    private static void mean(FaultLab lab, Recorder rec, String id, int seeds, int reps) {
        Instance inst = lab.instance(id);
        final int cols = inst.candidates();
        final int mMean = lab.quick() ? 20 : 200;
        final long msgBytes = Measure.PACKET_HEADER + Measure.pointBytes(inst.dim()) + 16;
        double[] rhos = Arrays.copyOfRange(RHOS, 1, RHOS.length);
        List<List<Recorder.Row>> out = lab.parallel(seeds, s -> {
            List<Recorder.Row> rows = new ArrayList<>();
            MeanCells mc = MeanCells.build(inst.points, mMean, NodeCoreset.MEAN_LLOYD,
                    LabSeeds.rng(lab.seed, "F3/mean", s, LabSeeds.hash(id)));
            final int k = mc.cells();
            PointSet meansSet = PointSet.of(mc.means);
            CostTable mt = CostTable.of(meansSet, inst.candidates, null);
            double[][] cellCost = new double[k][cols];
            double[] full = new double[cols];
            for (int c = 0; c < k; c++) {
                mt.addRow(c, mc.count(c), cellCost[c]);
                for (int j = 0; j < cols; j++) {
                    full[j] += cellCost[c][j];
                }
            }
            double total = inst.points.totalWeight();
            double[] base = full.clone();
            for (int j = 0; j < cols; j++) {
                base[j] += mc.offset;
            }
            double baseDist = Measure.distortion(inst.ratios(base, 1.0));
            int far = 0;
            for (int c = 0; c < k; c++) {
                double[] without = base.clone();
                for (int j = 0; j < cols; j++) {
                    without[j] -= cellCost[c][j];
                }
                far += Measure.distortion(inst.ratios(without, 1.0)) > 2.0 ? 1 : 0;
            }
            Clustering b = Seeding.kMeansPlusPlus(0).cluster(meansSet, Math.min(inst.k, k),
                    LabSeeds.rng(lab.seed, "F3/mean-split", s, 0), MetricsSink.NOOP);
            double[] sc = Sample.BOUND.sensitivities(meansSet, b);
            int[] split = new int[k];
            int splitMsgs = 0;
            for (int c = 0; c < k; c++) {
                split[c] = Math.max(1, (int) Math.ceil(sc[c] / ETA));
                splitMsgs += split[c];
            }
            rows.add(new Recorder.Row(cell("part", "mean", "instance", id, "rho", 0.0, "variant", "no-loss"), s,
                    metrics("distortion", baseDist, "meanAbsR", Measure.meanAbs(inst.ratios(base, 1.0)), "farCells",
                            far, "messages", k + 1, "bytes", (k + 1) * msgBytes), null));
            double[] recvShare = new double[k];
            for (double rho : rhos) {
                int replicas = Math.max(1, (int) Math.ceil(Math.log(DELTA) / Math.log(rho)));
                int fecTx = fecPackets(k, rho, DELTA);
                for (int t = 0; t < reps; t++) {
                    long fs = LabSeeds.of(lab.seed, "F3/mean-fault/" + id, s, (long) (rho * 1000) * 1000 + t);
                    boolean solve = t < 3;
                    // global offset and CF triples share one loss pattern over the k cell messages.
                    boolean[] lost = new boolean[Math.max(Math.max(k * replicas, splitMsgs), fecTx)];
                    Channel.bernoulli(rho).erase(lost, k, new SplittableRandom(fs));
                    for (int c = 0; c < k; c++) {
                        recvShare[c] = lost[c] ? 0.0 : 1.0;
                    }
                    double mass = receivedMass(mc, recvShare);
                    rows.add(meanRow(lab, inst, mc, cellCost, recvShare, 1.0, false, mc.offset, "global-none", rho, s,
                            t, far, baseDist, k + 1, msgBytes, solve));
                    rows.add(meanRow(lab, inst, mc, cellCost, recvShare, mass > 0 ? total / mass : 0.0, false,
                            mc.offset, "global-realized", rho, s, t, far, baseDist, k + 1, msgBytes, solve));
                    rows.add(meanRow(lab, inst, mc, cellCost, recvShare, mass > 0 ? total / mass : 0.0, true, 0.0,
                            "cf-realized", rho, s, t, far, baseDist, k, msgBytes, solve));
                    // split cells: each copy carries 1/r_c of the cell's count and cost.
                    Channel.bernoulli(rho).erase(lost, splitMsgs, new SplittableRandom(fs + 1));
                    for (int c = 0, at = 0; c < k; c++) {
                        int got = 0;
                        for (int x = 0; x < split[c]; x++) {
                            got += lost[at++] ? 0 : 1;
                        }
                        recvShare[c] = (double) got / split[c];
                    }
                    mass = receivedMass(mc, recvShare);
                    rows.add(meanRow(lab, inst, mc, cellCost, recvShare, mass > 0 ? total / mass : 0.0, true, 0.0,
                            "split", rho, s, t, far, baseDist, splitMsgs, msgBytes, solve));
                    Channel.bernoulli(rho).erase(lost, k * replicas, new SplittableRandom(fs + 2));
                    for (int c = 0; c < k; c++) {
                        boolean any = false;
                        for (int x = 0; x < replicas; x++) {
                            any |= !lost[c * replicas + x];
                        }
                        recvShare[c] = any ? 1.0 : 0.0;
                    }
                    mass = receivedMass(mc, recvShare);
                    rows.add(meanRow(lab, inst, mc, cellCost, recvShare, mass > 0 ? total / mass : 0.0, true, 0.0,
                            "replicate", rho, s, t, far, baseDist, k * replicas, msgBytes, solve));
                    // Ideal systematic MDS code: any k of the fecTx packets decode; else keep the source packets.
                    Channel.bernoulli(rho).erase(lost, fecTx, new SplittableRandom(fs + 3));
                    int arrived = 0;
                    for (int x = 0; x < fecTx; x++) {
                        arrived += lost[x] ? 0 : 1;
                    }
                    for (int c = 0; c < k; c++) {
                        recvShare[c] = arrived >= k || !lost[c] ? 1.0 : 0.0;
                    }
                    mass = receivedMass(mc, recvShare);
                    rows.add(meanRow(lab, inst, mc, cellCost, recvShare, mass > 0 ? total / mass : 0.0, true, 0.0,
                            "fec", rho, s, t, far, baseDist, fecTx, msgBytes, solve));
                }
            }
            return rows;
        });
        out.forEach(rec::addAll);
        lab.log("F3 mean %s done", id);
    }

    static double receivedMass(MeanCells mc, double[] share) {
        double m = 0.0;
        for (int c = 0; c < share.length; c++) {
            m += share[c] * mc.count(c);
        }
        return m;
    }

    /** Smallest t with P[Binomial(t, 1 - rho) < k] <= delta (exact tail, summed in log space). */
    static int fecPackets(int k, double rho, double delta) {
        for (int t = k; ; t++) {
            double fail = 0.0;
            for (int i = 0; i < k; i++) {
                fail += Math.exp(logChoose(t, i) + i * Math.log(1.0 - rho) + (t - i) * Math.log(rho));
            }
            if (fail <= delta) {
                return t;
            }
        }
    }

    static double logChoose(int n, int k) {
        double s = 0.0;
        for (int i = 1; i <= k; i++) {
            s += Math.log(n - k + i) - Math.log(i);
        }
        return s;
    }

    private static Recorder.Row meanRow(FaultLab lab, Instance inst, MeanCells mc, double[][] cellCost,
                                        double[] share, double scale, boolean cf, double globalOffset, String variant,
                                        double rho, int s, int t, int far, double baseDist, int messages, long msgBytes,
                                        boolean solve) {
        int cols = inst.candidates();
        double[] cost = new double[cols];
        double offset = globalOffset;
        for (int c = 0; c < share.length; c++) {
            if (share[c] > 0.0) {
                for (int j = 0; j < cols; j++) {
                    cost[j] += scale * share[c] * cellCost[c][j];
                }
                if (cf) {
                    offset += scale * share[c] * mc.phi[c];
                }
            }
        }
        for (int j = 0; j < cols; j++) {
            cost[j] += offset;
        }
        double[] r = inst.ratios(cost, 1.0);
        Double ds = null;
        if (solve) {
            int n = 0;
            for (double x : share) {
                n += x > 0 ? 1 : 0;
            }
            int[] rows = new int[n];
            double[] w = new double[n];
            for (int c = 0, at = 0; c < share.length; c++) {
                if (share[c] > 0) {
                    rows[at] = c;
                    w[at++] = scale * share[c] * mc.count(c);
                }
            }
            PointBlock blk = PointSet.of(mc.means).select(rows, w, n);
            ds = Measure.downstream(inst, PointSet.of(blk), LabSeeds.of(lab.seed, "solve/F3-mean", s, t));
        }
        int k = share.length;
        return new Recorder.Row(cell("part", "mean", "instance", inst.id, "rho", rho, "variant", variant), s * 1000 + t,
                metrics("distortion", Measure.distortion(r), "excess", Measure.distortion(r) - baseDist, "meanAbsR",
                        Measure.meanAbs(r), "farCells", far,
                        "predictedP2", 1.0 - Math.pow(1.0 - rho, far), "messages", messages, "overhead",
                        (double) messages / k - 1.0, "bytes", messages * msgBytes, "downstream", ds), null);
    }

    // ---------------------------------------------------------------- verdicts

    static Map<String, Object> sCell(String inst, int m, String ch, double rho, String pk, String dec) {
        return cell("part", "sample", "instance", inst, "m", m, "channel", ch, "rho", rho, "packetization", pk,
                "decoder", dec);
    }

    private static void verdicts(FaultLab lab, Recorder rec, int[] ms, int[] sweepMs) {
        StringBuilder md = new StringBuilder("## F3: packet erasure of coreset points\n\n");
        b1(lab, rec, ms, md);
        bursts(lab, rec, ms, md);
        b2(lab, rec, sweepMs, md);
        b4(lab, rec, md);
        lab.section(md.toString());
    }

    /** Slope of log mean(distortion - 1) against log 1/(1 - rho), with a seed-bootstrap 95% interval. */
    private static double[] b1Slope(Recorder rec, String inst, int m, String pk) {
        List<double[]> per = new ArrayList<>();
        for (double rho : RHOS) {
            per.add(rec.values(sCell(inst, m, "bernoulli", rho, pk, "ht-realized"), "distortion"));
        }
        int seeds = per.get(0).length;
        ToDoubleFunction<int[]> slope = idx -> {
            double[] x = new double[RHOS.length];
            double[] y = new double[RHOS.length];
            for (int i = 0; i < RHOS.length; i++) {
                double s = 0.0;
                for (int k : idx) {
                    s += per.get(i)[k] - 1.0;
                }
                x[i] = Math.log(1.0 / (1.0 - RHOS[i]));
                y[i] = Math.log(s / idx.length);
            }
            return Stats.ols(x, y).b();
        };
        double point = slope.applyAsDouble(Partitions.identity(seeds));
        double[] ci = Stats.bootstrap(seeds, slope, 1000, 7L);
        double ratio = Stats.mean(per.get(RHOS.length - 1)) - 1.0;
        ratio /= Stats.mean(per.get(0)) - 1.0;
        return new double[] {point, ci[0], ci[1], ratio};
    }

    private static void b1(FaultLab lab, Recorder rec, int[] ms, StringBuilder md) {
        md.append("### B1: erasure invariance (Bernoulli, `ht-realized`)\n\n| instance | m | packetization | "
                + "slope of log(distortion-1) vs log 1/(1-rho) | 95% CI | excess ratio rho=0.5 : 0 |\n"
                + "|---|---|---|---|---|---|\n");
        double[] main = null;
        for (String inst : List.of("zipf", "zipf-heavy", "esa22")) {
            for (int m : ms) {
                for (String pk : List.of("per-draw", "random-b64", "merged", "rateless")) {
                    double[] sl = b1Slope(rec, inst, m, pk);
                    if (inst.equals("zipf") && m == ms[0] && pk.equals("per-draw")) {
                        main = sl;
                    }
                    md.append(f("| %s | %d | %s | %.3f | [%.3f, %.3f] | %.3f |%n", inst, m, pk, sl[0], sl[1], sl[2],
                            sl[3]));
                }
            }
        }
        md.append('\n');
        md.append("Mean distortion by rho (zipf, m = ").append(ms[0]).append(", Bernoulli, `ht-realized`):\n\n| rho |");
        List<String> pks = List.of("per-draw", "merged", "random-b8", "random-b64", "sorted-b8", "rateless");
        for (String pk : pks) {
            md.append(' ').append(pk).append(" |");
        }
        md.append("\n|---|").append("---|".repeat(pks.size())).append('\n');
        for (double rho : RHOS) {
            md.append("| ").append(rho).append(" |");
            for (String pk : pks) {
                md.append(f(" %.4f |", Stats.mean(rec.values(sCell("zipf", ms[0], "bernoulli", rho, pk,
                        "ht-realized"), "distortion"))));
            }
            md.append('\n');
        }
        md.append("\nDecoders at rho = 0.3 (zipf, m = ").append(ms[0]).append(", per-draw, Bernoulli): ");
        for (String dec : DECODERS) {
            md.append(f("%s mean distortion %.4f; ", dec, Stats.mean(rec.values(sCell("zipf", ms[0], "bernoulli", 0.3,
                    "per-draw", dec), "distortion"))));
        }
        md.append("\n\n");
        boolean pass = main[1] <= 0.6 && main[2] >= 0.4 && Math.abs(main[0] - 0.5) <= 0.1;
        lab.verdict("F3/B1 slope", "split + realized: log-log slope of (distortion-1) vs 1/(1-rho) = 0.5 +/- 0.1; "
                        + "x1.41 at rho=0.5",
                f("zipf m=%d per-draw: slope %.3f (95%% CI %.3f..%.3f), excess ratio rho=0.5:0 = %.3f", ms[0], main[0],
                        main[1], main[2], main[3]),
                pass ? "PASS" : "FAIL");
        double[] rl = b1Slope(rec, "zipf", ms[0], "rateless");
        lab.verdict("F3/C3 rateless link", "rateless with one ACK: distortion independent of rho (m draws always "
                + "arrive), at ~m/(1-rho) transmissions",
                f("slope %.3f (CI %.3f..%.3f); packets sent at rho=0.5: %.0f for m=%d", rl[0], rl[1], rl[2],
                        Stats.mean(rec.values(sCell("zipf", ms[0], "bernoulli", 0.5, "rateless", "ht-realized"),
                                "packets")), ms[0]),
                rl[1] <= 0.0 && rl[2] >= 0.0 ? "PASS" : Math.abs(rl[0]) < 0.1 ? "PASS" : "FAIL");
    }

    private static void bursts(FaultLab lab, Recorder rec, int[] ms, StringBuilder md) {
        int m = ms[0];
        md.append("### Bursts (zipf, m = ").append(m).append(", rho = 0.3, `ht-realized`)\n\n| channel | "
                + "packetization | distortion mean | sd | p99 | max | sd(m') |\n|---|---|---|---|---|---|---|\n");
        for (String ch : CHANNELS) {
            for (String pk : List.of("per-draw", "merged", "random-b64", "sorted-b8")) {
                Recorder.Cell c = rec.get(sCell("zipf", m, ch, 0.3, pk, "ht-realized"));
                double[] d = c.values("distortion");
                md.append(f("| %s | %s | %.4f | %.4f | %.4f | %.4f | %.1f |%n", ch, pk, Stats.mean(d), Stats.sd(d),
                        Stats.quantile(d, 0.99), Stats.max(d), Stats.sd(c.values("mPrime"))));
            }
        }
        md.append('\n');
        double[] bern = rec.values(sCell("zipf", m, "bernoulli", 0.3, "per-draw", "ht-realized"), "distortion");
        double[] ge = rec.values(sCell("zipf", m, "ge-b20", 0.3, "per-draw", "ht-realized"), "distortion");
        double sdB = Stats.sd(rec.values(sCell("zipf", m, "bernoulli", 0.3, "per-draw", "ht-realized"), "mPrime"));
        double sdG = Stats.sd(rec.values(sCell("zipf", m, "ge-b20", 0.3, "per-draw", "ht-realized"), "mPrime"));
        double meanRel = (Stats.mean(ge) - 1.0) / (Stats.mean(bern) - 1.0);
        lab.verdict("F3/B1 bursts", "GE at the same mean rho with random packing: same mean as Bernoulli, wider spread",
                f("rho=0.3, burst 20: mean excess GE/Bernoulli = %.3f; sd(distortion) %.4f vs %.4f; sd(m') %.0f vs "
                        + "%.0f", meanRel, Stats.sd(ge), Stats.sd(bern), sdG, sdB),
                Math.abs(meanRel - 1.0) <= 0.15 && sdG > sdB && Stats.sd(ge) >= Stats.sd(bern) ? "PASS"
                        : Math.abs(meanRel - 1.0) <= 0.15 && sdG > sdB ? "PASS (m' spread) / FAIL (distortion spread)"
                        : "FAIL");
        double[] sorted = rec.values(sCell("zipf", m, "ge-b20", 0.3, "sorted-b8", "ht-realized"), "distortion");
        double p99s = Stats.quantile(sorted, 0.99) - 1.0;
        double p99r = Stats.quantile(ge, 0.99) - 1.0;
        lab.verdict("F3/B1 sorted+bursts", "sorted packing + bursts: large outliers (failure)",
                f("rho=0.3, burst 20: p99 excess sorted-b8 %.3f vs per-draw %.3f (x%.1f); max %.3f", p99s, p99r,
                        p99s / p99r, Stats.max(sorted)),
                p99s >= 3 * p99r ? "PASS" : "FAIL");
    }

    private static Map<String, Object> swCell(String inst, int m, double rho, String pk, String dec) {
        return cell("part", "sweep", "instance", inst, "m", m, "rho", rho, "packetization", pk, "decoder", dec);
    }

    /** Paired excess mean-square of r(Q_j) over rho = 0 (same sample), for the candidate {@code j}. */
    private static double excessMS(Recorder rec, String inst, int m, double rho, String pk, String dec, int j) {
        List<double[]> a = rec.get(swCell(inst, m, rho, pk, dec)).rs;
        List<double[]> b = rec.get(swCell(inst, m, 0.0, pk, dec)).rs;
        double s = 0.0;
        for (int i = 0; i < a.size(); i++) {
            s += a.get(i)[j] * a.get(i)[j] - b.get(i)[j] * b.get(i)[j];
        }
        return s / a.size();
    }

    private static void b2(FaultLab lab, Recorder rec, int[] ms, StringBuilder md) {
        md.append("### B2: merged duplicates versus split packets (Bernoulli, `ht-prior`)\n\n"
                + "Erasure RMS = sqrt of the paired excess mean-square of r(Q*) over rho = 0 for the candidate Q* with "
                + "the largest kappa_pt; floor = sqrt(rho/(1-rho) kappa_pt(Q*)).\n\n"
                + "| instance | kappa_pt(Q*) | rho | m | distinct/m | split RMS | merged RMS | floor |\n"
                + "|---|---|---|---|---|---|---|---|\n");
        String heavyInst = "zipf-heavy";
        double plateauMerged = 0.0;
        double plateauSplit = 0.0;
        double heavyFloor = 0.0;
        double heavyMerged = 0.0;
        double balancedRatio = Double.NaN;
        for (String id : List.of("zipf", "zipf-heavy", "esa22")) {
            Instance inst = lab.instance(id);
            int jStar = 0;
            for (int j = 1; j < inst.candidates(); j++) {
                jStar = inst.kappa[j] > inst.kappa[jStar] ? j : jStar;
            }
            for (double rho : new double[] {0.1, 0.3}) {
                double floor = Math.sqrt(rho / (1.0 - rho) * inst.kappa[jStar]);
                for (int m : ms) {
                    double sp = Math.sqrt(Math.max(0.0, excessMS(rec, id, m, rho, "per-draw", "ht-prior", jStar)));
                    double me = Math.sqrt(Math.max(0.0, excessMS(rec, id, m, rho, "merged", "ht-prior", jStar)));
                    double distinct = Stats.mean(rec.values(swCell(id, m, rho, "merged", "ht-prior"), "distinct")) / m;
                    md.append(f("| %s | %.3g | %s | %d | %.3f | %.4f | %.4f | %.4f |%n", id, inst.kappa[jStar], rho, m,
                            distinct, sp, me, floor));
                    if (id.equals(heavyInst) && rho == 0.3 && m == ms[ms.length - 1]) {
                        heavyFloor = floor;
                        heavyMerged = me;
                    }
                    if (id.equals("zipf") && rho == 0.3 && m == 1000) {
                        balancedRatio = me / sp;
                    }
                }
                if (id.equals(heavyInst) && rho == 0.3) {
                    int last = ms[ms.length - 1];
                    int mid = ms[ms.length - 3];
                    plateauMerged = Math.sqrt(excessMS(rec, id, last, rho, "merged", "ht-prior", jStar)
                            / excessMS(rec, id, mid, rho, "merged", "ht-prior", jStar));
                    plateauSplit = Math.sqrt(excessMS(rec, id, last, rho, "per-draw", "ht-prior", jStar)
                            / excessMS(rec, id, mid, rho, "per-draw", "ht-prior", jStar));
                }
            }
        }
        md.append('\n');
        int last = ms[ms.length - 1];
        int mid = ms[ms.length - 3];
        double expectSplit = Math.sqrt((double) mid / last);
        lab.verdict("F3/B2 plateau", "merged packets: excess plateaus as m grows at ~sqrt(rho/(1-rho) kappa_pt) "
                        + "(ESA'22 / high-s_max instance)",
                f("zipf-heavy rho=0.3, Q* = max kappa_pt: erasure RMS m=%d->%d shrinks x%.2f merged vs x%.2f split "
                        + "(sqrt(m ratio) = %.2f); merged RMS at m=%d %.4f vs floor %.4f", mid, last, plateauMerged,
                        plateauSplit, expectSplit, last, heavyMerged, heavyFloor),
                plateauMerged >= 0.8 && plateauSplit <= 1.25 * expectSplit
                        && Math.abs(heavyMerged / heavyFloor - 1.0) <= 0.35 ? "PASS" : "FAIL");
        lab.verdict("F3/B2 balanced", "balanced Zipf instance at large N: merged ~ split",
                f("zipf rho=0.3 m=1000: merged/split erasure RMS for Q* = %.3f", balancedRatio),
                Math.abs(balancedRatio - 1.0) <= 0.2 ? "PASS" : "FAIL");
    }

    private static Map<String, Object> mCell(String inst, double rho, String variant) {
        return cell("part", "mean", "instance", inst, "rho", rho, "variant", variant);
    }

    private static void b4(FaultLab lab, Recorder rec, StringBuilder md) {
        md.append("### B4: mean coreset (Bernoulli cell loss)\n\n| instance | rho | variant | distortion mean | "
                + "p50 | p99 | max | P[> 2] | predicted 1-(1-rho)^far | downstream mean | messages/cells - 1 |\n"
                + "|---|---|---|---|---|---|---|---|---|---|---|\n");
        double maxGap = 0.0;
        StringBuilder gaps = new StringBuilder();
        double worstSplit = 0.0;
        double worstRepl = 0.0;
        double fecExact = 1.0;
        double cfGlobal = Double.NaN;
        double cfP15 = Double.NaN;
        for (String id : List.of("zipf", "zipf-heavy")) {
            double base = Stats.mean(rec.values(mCell(id, 0.0, "no-loss"), "distortion"));
            md.append(f("| %s | 0 | no-loss | %.4f | | | | | | | 0 |%n", id, base));
            for (double rho : Arrays.copyOfRange(RHOS, 1, RHOS.length)) {
                for (String v : List.of("global-none", "global-realized", "cf-realized", "split", "replicate", "fec")) {
                    Recorder.Cell c = rec.get(mCell(id, rho, v));
                    double[] d = c.values("distortion");
                    double p2 = Stats.fraction(d, 2.0);
                    double pred = Stats.mean(c.values("predictedP2"));
                    double[] ds = c.values("downstream");
                    md.append(f("| %s | %s | %s | %.4f | %.4f | %.4f | %.3g | %.3f | %.3f | %.4f | %.2f |%n", id, rho,
                            v, Stats.mean(d), Stats.quantile(d, 0.5), Stats.quantile(d, 0.99), Stats.max(d), p2, pred,
                            Stats.mean(ds), Stats.mean(c.values("overhead"))));
                    if (v.equals("global-none")) {
                        double gap = Math.abs(p2 - pred);
                        if (gap > maxGap) {
                            maxGap = gap;
                        }
                        gaps.append(f("%s rho=%s: %.2f vs %.2f; ", id, rho, p2, pred));
                    }
                    if (rho <= 0.3 && v.equals("split")) {
                        double scaleRho = Math.sqrt(rho * ETA / (1.0 - rho));
                        worstSplit = Math.max(worstSplit, Stats.quantile(c.values("excess"), 0.99) / scaleRho);
                    }
                    if (v.equals("replicate")) {
                        worstRepl = Math.max(worstRepl, Stats.fraction(d, 2.0));
                    }
                    if (v.equals("fec")) {
                        int exact = 0;
                        for (double x : c.values("excess")) {
                            exact += Math.abs(x) <= 1e-6 ? 1 : 0;
                        }
                        fecExact = Math.min(fecExact, (double) exact / d.length);
                    }
                    if (id.equals("zipf") && rho == 0.3 && v.equals("cf-realized")) {
                        cfP15 = Stats.fraction(d, 1.5);
                        cfGlobal = Stats.mean(d) / Stats.mean(rec.values(mCell(id, rho, "global-realized"),
                                "distortion"));
                    }
                }
            }
        }
        md.append('\n');
        lab.verdict("F3/B4 global offset", "mean coreset, global offset: heavy-tailed; P[distortion > 2] = "
                + "1 - (1-rho)^{#far cells}", f("largest |measured - predicted| = %.2f (%s)", maxGap, gaps),
                maxGap <= 0.1 ? "PASS" : "FAIL");
        lab.verdict("F3/B4 CF triples", "CF triples fix the offset bias but not the missing mass",
                f("zipf rho=0.3: mean distortion cf/global-realized = %.3f; P[cf distortion > 1.5] = %.3f", cfGlobal,
                        cfP15),
                cfGlobal < 1.0 && cfP15 > 0.0 ? "PASS" : cfGlobal < 1.0 ? "PASS (bias) / FAIL (no residual tail)"
                        : "FAIL");
        lab.verdict("F3/B4 split", "split centroids restore distortion ~ 1 + O(sqrt(rho eta))",
                f("rho <= 0.3, eta = %s: worst (p99 distortion - no-loss) / sqrt(rho eta/(1-rho)) = %.2f", ETA,
                        worstSplit),
                worstSplit <= 3.0 ? "PASS" : "FAIL");
        lab.verdict("F3/B4 replicate", "replicated centroids (r = ceil(log delta / log rho), delta = 0.01) remove the "
                + "heavy tail", f("worst P[distortion > 2] over rho = %.3f", worstRepl),
                worstRepl <= 0.02 ? "PASS" : "FAIL");
        lab.verdict("F3/B4 FEC", "FEC restores exactness at overhead ~ rho",
                f("fraction of runs equal to no loss (to 1e-6) >= %.3f (target 1 - delta = %.2f); overhead in the "
                        + "table", fecExact, 1.0 - DELTA),
                fecExact >= 1.0 - 3 * DELTA ? "PASS" : "FAIL");
    }
}
