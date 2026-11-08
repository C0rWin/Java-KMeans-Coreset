package io.github.c0rwin.coreset.planner.validate;

import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import io.github.c0rwin.coreset.planner.Configuration;
import io.github.c0rwin.coreset.planner.Configuration.Family;
import io.github.c0rwin.coreset.planner.Configuration.Packetization;
import io.github.c0rwin.coreset.planner.Configuration.Redundancy;
import io.github.c0rwin.coreset.planner.Configuration.StragglerPolicy;
import io.github.c0rwin.coreset.planner.Configuration.Transport;
import io.github.c0rwin.coreset.planner.EnvironmentProfile;
import io.github.c0rwin.coreset.planner.EnvironmentProfile.GilbertElliott;
import io.github.c0rwin.coreset.planner.EnvironmentProfile.Partition;
import io.github.c0rwin.coreset.planner.FamilyScores;
import io.github.c0rwin.coreset.planner.Requirement;
import io.github.c0rwin.coreset.planner.model.NetworkCostModel;
import io.github.c0rwin.coreset.planner.model.ResilienceModel.Prediction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.stream.IntStream;

/**
 * Runs a configuration in-process on a full data set with injected faults and measures the worst relative cost error
 * over the workload's candidate queries. Faults: Gilbert-Elliott packet erasures per sender link (with systematic
 * block FEC when configured), independent F_pre/F_post node crashes, stragglers, and crashes of internal tree nodes.
 * Records are packed in uniformly random order. Decoders: realized-count reweighting for per-draw records, HT with
 * the realized packet-delivery rate for merged/split records, realized-mass rescaling for clustering-feature cells, and
 * survivor reweighting by the reliable per-share input weights.
 *
 * <p>Costs are evaluated from a precomputed table {@code dist[q][row]}, so a trial costs one coreset construction per
 * sender plus O(records) per candidate.
 */
public final class PlanSimulator {

    private final PointSet data;
    private final List<List<SparseVector>> candidates;
    private final double[][] dist;
    private final double[] fullCost;
    private final double totalWeight;
    private final int[] placementOrder;
    private final NetworkCostModel net = new NetworkCostModel();

    /** Measured errors of one validation run. */
    public record Result(int trials, double[] worstErrors, double quantile, double mean, double max,
                         double meanLostFraction, double meanErasure) {
    }

    public PlanSimulator(PointSet data, List<List<SparseVector>> candidates) {
        this.data = data;
        this.candidates = candidates;
        int n = data.size();
        this.dist = new double[candidates.size()][];
        this.fullCost = new double[candidates.size()];
        for (int q = 0; q < candidates.size(); q++) {
            List<SparseVector> cs = candidates.get(q);
            double[] dq = new double[n];
            IntStream.range(0, n).parallel().forEach(i -> {
                SparseVector v = data.vector(i);
                dq[i] = v.squaredDistance(cs.get(Cost.nearest(v, cs)));
            });
            dist[q] = dq;
            double c = 0.0;
            for (int i = 0; i < n; i++) {
                c += data.weight(i) * dq[i];
            }
            fullCost[q] = c;
        }
        this.totalWeight = data.totalWeight();
        // adversarial placement: rows sorted by their nearest center of the first candidate (cluster-correlated)
        Integer[] order = new Integer[n];
        int[] label = new int[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
            label[i] = Cost.nearest(data.vector(i), candidates.get(0));
        }
        Arrays.sort(order, (a, b) -> Integer.compare(label[a], label[b]));
        this.placementOrder = Arrays.stream(order).mapToInt(Integer::intValue).toArray();
    }

    public Result run(Configuration c, Prediction plan, EnvironmentProfile e, Requirement r, int trials, long seed) {
        double[] worst = new double[trials];
        double lost = 0.0;
        double erased = 0.0;
        SplittableRandom root = new SplittableRandom(seed);
        for (int t = 0; t < trials; t++) {
            Trial tr = new Trial(c, plan, e, r, root.split());
            worst[t] = tr.run();
            lost += tr.lostFraction;
            erased += tr.sentPackets == 0 ? 0.0 : (double) tr.erasedPackets / tr.sentPackets;
        }
        double[] sorted = worst.clone();
        Arrays.sort(sorted);
        int idx = Math.min(trials - 1, (int) Math.ceil((1.0 - r.delta()) * trials) - 1);
        double mean = Arrays.stream(worst).average().orElse(Double.NaN);
        return new Result(trials, worst, sorted[Math.max(0, idx)], mean, sorted[trials - 1], lost / trials,
                erased / trials);
    }

    /** A weighted multiset of input rows (sampling families) plus explicit cells (mean coreset). */
    private static final class Summary {
        final List<int[]> rowsParts = new ArrayList<>();
        final List<double[]> weightParts = new ArrayList<>();
        final List<SparseVector> cellMeans = new ArrayList<>();
        final List<Double> cellWeights = new ArrayList<>();
        double offset;
        double coveredWeight;

        void addRows(int[] rows, double[] weights) {
            rowsParts.add(rows);
            weightParts.add(weights);
        }

        void addAll(Summary s, double scale) {
            for (int i = 0; i < s.rowsParts.size(); i++) {
                double[] w = s.weightParts.get(i).clone();
                for (int j = 0; j < w.length; j++) {
                    w[j] *= scale;
                }
                addRows(s.rowsParts.get(i), w);
            }
            for (int i = 0; i < s.cellMeans.size(); i++) {
                cellMeans.add(s.cellMeans.get(i));
                cellWeights.add(s.cellWeights.get(i) * scale);
            }
            offset += s.offset * scale;
            coveredWeight += s.coveredWeight;
        }

        int[] rows() {
            return rowsParts.stream().flatMapToInt(Arrays::stream).toArray();
        }

        double[] weights() {
            return weightParts.stream().flatMapToDouble(Arrays::stream).toArray();
        }
    }

    private final class Trial {
        final Configuration c;
        final Prediction plan;
        final EnvironmentProfile e;
        final Requirement r;
        final SplittableRandom rnd;
        final GilbertElliott ge;
        final int rec;
        final int perPacket;
        double lostFraction;
        long sentPackets;
        long erasedPackets;
        int lastPackets;
        int lastLost;

        Trial(Configuration c, Prediction plan, EnvironmentProfile e, Requirement r, SplittableRandom rnd) {
            this.c = c;
            this.plan = plan;
            this.e = e;
            this.r = r;
            this.rnd = rnd;
            this.ge = e.gilbertElliott();
            this.rec = net.recordBytes((double) data.storedEntries() / ((double) data.size() * data.dim()), data.dim(),
                    c.packetization());
            this.perPacket = c.transport().datagram() ? Math.max(1, net.recordsPerPacket(e, rec)) : 1;
        }

        double run() {
            int n = e.nodes();
            int group = switch (c.redundancy()) {
                case NONE, POST_REPLICATION -> 1;
                case FRACTIONAL_REPETITION -> c.redundancyLevel() + 1;
                case RATELESS -> c.redundancyLevel() + c.ratelessSpare();
            };
            int units = n / group;
            // node states
            boolean[] pre = new boolean[n];
            boolean[] post = new boolean[n];
            boolean[] straggler = new boolean[n];
            for (int i = 0; i < n; i++) {
                pre[i] = rnd.nextDouble() < e.failPre();
                post[i] = !pre[i] && rnd.nextDouble() < e.failPost();
            }
            int[] perm = shuffled(n);
            for (int i = 0; i < e.stragglers(); i++) {
                straggler[perm[i]] = true;
            }
            // data shares
            int[][] shares = partition(units);

            List<Summary> leaves = new ArrayList<>(units);
            double coveredAll = 0.0;
            for (int u = 0; u < units; u++) {
                Summary s = leaf(u, shares[u], group, pre, post, straggler, n);
                leaves.add(s);
            }
            Summary result;
            if (c.arity() == 0) {
                result = new Summary();
                for (Summary s : leaves) {
                    if (s != null) {
                        result.addAll(s, 1.0);
                    }
                }
            } else {
                result = tree(leaves);
            }
            coveredAll = result == null ? 0.0 : result.coveredWeight;
            lostFraction = 1.0 - coveredAll / totalWeight;
            if (result == null || coveredAll <= 0.0) {
                return 1.0;
            }
            double scale = totalWeight / coveredAll;
            int[] rows = result.rows();
            double[] w = result.weights();
            double worst = 0.0;
            PointSet cells = cellSet(result);
            for (int q = 0; q < candidates.size(); q++) {
                double est = 0.0;
                for (int j = 0; j < rows.length; j++) {
                    est += w[j] * dist[q][rows[j]];
                }
                if (cells != null) {
                    est += Cost.cost(cells, candidates.get(q)) + result.offset;
                }
                est *= scale;
                worst = Math.max(worst, Math.abs(est / fullCost[q] - 1.0));
            }
            return worst;
        }

        private PointSet cellSet(Summary s) {
            if (s.cellMeans.isEmpty()) {
                return null;
            }
            List<io.github.c0rwin.coreset.core.WeightedPoint> pts = new ArrayList<>();
            for (int i = 0; i < s.cellMeans.size(); i++) {
                pts.add(new io.github.c0rwin.coreset.core.WeightedPoint(s.cellMeans.get(i), s.cellWeights.get(i)));
            }
            return PointSet.of(PointBlock.of(pts));
        }

        private int[][] partition(int units) {
            int n = data.size();
            int[][] out = new int[units][];
            if (e.partition() == Partition.ADVERSARIAL) {
                for (int u = 0; u < units; u++) {
                    int from = (int) ((long) n * u / units);
                    int to = (int) ((long) n * (u + 1) / units);
                    out[u] = Arrays.copyOfRange(placementOrder, from, to);
                }
                return out;
            }
            int[] count = new int[units];
            int[] assign = new int[n];
            for (int i = 0; i < n; i++) {
                assign[i] = rnd.nextInt(units);
                count[assign[i]]++;
            }
            for (int u = 0; u < units; u++) {
                out[u] = new int[count[u]];
                count[u] = 0;
            }
            for (int i = 0; i < n; i++) {
                out[assign[i]][count[assign[i]]++] = i;
            }
            return out;
        }

        /** The share's summary as received by the coordinator (or the parent), or null if lost. */
        private Summary leaf(int u, int[] rows, int group, boolean[] pre, boolean[] post, boolean[] straggler, int n) {
            double shareWeight = 0.0;
            for (int row : rows) {
                shareWeight += data.weight(row);
            }
            if (rows.length == 0) {
                Summary s = new Summary();
                s.coveredWeight = 0.0;
                return s;
            }
            int first = u * group;
            List<Integer> responders = new ArrayList<>();
            switch (c.redundancy()) {
                case NONE, POST_REPLICATION -> {
                    int node = first;
                    if (pre[node]) {
                        return null;
                    }
                    if (straggler[node] && c.stragglers() == StragglerPolicy.DROP) {
                        return null;
                    }
                    if (post[node]) {
                        if (c.redundancy() == Redundancy.NONE) {
                            return null;
                        }
                        boolean held = false;
                        int[] perm = shuffled(n);
                        int taken = 0;
                        for (int i = 0; i < n && taken < c.redundancyLevel(); i++) {
                            if (perm[i] == node) {
                                continue;
                            }
                            taken++;
                            held |= !pre[perm[i]] && !post[perm[i]];
                        }
                        if (!held) {
                            return null;
                        }
                    }
                    responders.add(node);
                }
                case FRACTIONAL_REPETITION -> {
                    Integer pick = null;
                    for (int i = first; i < first + group; i++) {
                        if (!pre[i] && !post[i] && (pick == null || (straggler[pick] && !straggler[i]))) {
                            pick = i;
                        }
                    }
                    if (pick == null) {
                        return null;
                    }
                    responders.add(pick);
                }
                case RATELESS -> {
                    int fast = 0;
                    for (int i = first; i < first + group; i++) {
                        if (!pre[i] && !post[i] && !straggler[i]) {
                            fast++;
                        }
                    }
                    boolean absorbed = e.stragglers() <= c.ratelessSpare();
                    for (int i = first; i < first + group; i++) {
                        if (!pre[i] && !post[i] && (!straggler[i] || !absorbed || fast == 0)) {
                            responders.add(i);
                        }
                    }
                    if (responders.isEmpty()) {
                        return null;
                    }
                }
                default -> throw new IllegalStateException();
            }
            double[] baseW = new double[rows.length];
            for (int j = 0; j < rows.length; j++) {
                baseW[j] = data.weight(rows[j]);
            }
            Summary s = c.family().sampling()
                    ? sampled(rows, baseW, plan.drawsPerSender(), responders.size())
                    : cells(rows, baseW, (int) plan.drawsPerSender());
            if (s != null) {
                s.coveredWeight = shareWeight;
            }
            return s;
        }

        /** Per-sender sampling plus channel plus decoder; pools {@code members} independent senders (rateless). */
        private Summary sampled(int[] rows, double[] weights, long draws, int members) {
            PointSet local = PointSet.of(data.select(rows, weights, rows.length));
            double[] q = c.family() == Family.SENSITIVITY && local.size() >= r.k()
                    ? FamilyScores.of(local, r.k(), rnd.split()).q(Family.SENSITIVITY)
                    : FamilyScores.of(local, 1, rnd.split()).q(c.family() == Family.SENSITIVITY ? Family.UNIFORM
                            : c.family());
            AliasSampler sampler = AliasSampler.of(q);
            List<int[]> recIdx = new ArrayList<>();
            List<int[]> recCount = new ArrayList<>();
            long received = 0;
            double sentTotal = 0;
            double deliveredRate = 0;
            for (int mbr = 0; mbr < members; mbr++) {
                int[] counts = new int[local.size()];
                List<int[]> records = new ArrayList<>(); // {local index, count}
                if (c.packetization() == Packetization.PER_DRAW && c.transport() == Transport.EARLY_CLOSE) {
                    // stream fresh draws packet by packet until the receiver has acknowledged the target count
                    long got = 0;
                    boolean bad = rnd.nextDouble() < ge.lossRate();
                    while (got < draws) {
                        bad = e.rho() > 0.0 && step(bad);
                        sentPackets++;
                        if (bad) {
                            erasedPackets++;
                            for (int j = 0; j < perPacket; j++) {
                                sampler.next(rnd);
                            }
                            continue;
                        }
                        for (int j = 0; j < perPacket && got < draws; j++, got++) {
                            recIdx.add(new int[] {sampler.next(rnd)});
                            recCount.add(new int[] {1});
                        }
                    }
                    received += got;
                    sentTotal += draws;
                    deliveredRate += 1.0;
                    continue;
                }
                if (c.packetization() == Packetization.PER_DRAW) {
                    for (long d = 0; d < draws; d++) {
                        records.add(new int[] {sampler.next(rnd), 1});
                    }
                } else {
                    for (long d = 0; d < draws; d++) {
                        counts[sampler.next(rnd)]++;
                    }
                    int beta = c.packetization() == Packetization.SPLIT ? c.splitBudget() : Integer.MAX_VALUE;
                    for (int i = 0; i < counts.length; i++) {
                        int left = counts[i];
                        while (left > 0) {
                            int part = Math.min(beta, left);
                            records.add(new int[] {i, part});
                            left -= part;
                        }
                    }
                }
                boolean[] ok = deliver(records.size());
                long got = 0;
                for (int j = 0; j < records.size(); j++) {
                    if (ok[j]) {
                        recIdx.add(new int[] {records.get(j)[0]});
                        recCount.add(new int[] {records.get(j)[1]});
                        got += records.get(j)[1];
                    }
                }
                received += got;
                sentTotal += draws;
                deliveredRate += lastPackets == 0 ? 0.0 : 1.0 - (double) lastLost / lastPackets;
            }
            if (recIdx.isEmpty()) {
                return null;
            }
            double factor = c.packetization() == Packetization.PER_DRAW
                    ? 1.0 / received
                    : 1.0 / (sentTotal * Math.max(1e-12, deliveredRate / members));
            int[] outRows = new int[recIdx.size()];
            double[] outW = new double[recIdx.size()];
            for (int j = 0; j < outRows.length; j++) {
                int i = recIdx.get(j)[0];
                outRows[j] = rows[i];
                outW[j] = weights[i] * recCount.get(j)[0] / q[i] * factor;
            }
            Summary s = new Summary();
            s.addRows(outRows, outW);
            return s;
        }

        /** Mean coreset of the share with plain D^2 cells; CF records rescale by realized mass. */
        private Summary cells(int[] rows, double[] weights, int cellsWanted) {
            PointSet local = PointSet.of(data.select(rows, weights, rows.length));
            int cells = Math.max(1, Math.min(cellsWanted, local.size()));
            List<SparseVector> seeds = KMeansPlusPlus.seed(local, cells, 1, rnd.split(), MetricsSink.NOOP);
            int g = seeds.size();
            int dim = data.dim();
            double[][] sum = new double[g][dim];
            double[] wt = new double[g];
            int[] assign = new int[local.size()];
            for (int i = 0; i < local.size(); i++) {
                SparseVector v = local.vector(i);
                assign[i] = Cost.nearest(v, seeds);
                v.addTo(sum[assign[i]], local.weight(i));
                wt[assign[i]] += local.weight(i);
            }
            SparseVector[] means = new SparseVector[g];
            double[] norms = new double[g];
            for (int a = 0; a < g; a++) {
                if (wt[a] > 0.0) {
                    for (int j = 0; j < dim; j++) {
                        sum[a][j] /= wt[a];
                        norms[a] += sum[a][j] * sum[a][j];
                    }
                    means[a] = SparseVector.dense(sum[a]);
                }
            }
            double[] off = new double[g];
            for (int i = 0; i < local.size(); i++) {
                off[assign[i]] += local.weight(i) * local.vector(i).squaredDistance(sum[assign[i]], norms[assign[i]]);
            }
            boolean[] ok = deliver(g);
            Summary s = new Summary();
            double wAll = 0.0;
            double wGot = 0.0;
            double offAll = 0.0;
            double offGot = 0.0;
            for (int a = 0; a < g; a++) {
                wAll += wt[a];
                offAll += off[a];
                if (ok[a] && wt[a] > 0.0) {
                    wGot += wt[a];
                    offGot += off[a];
                }
            }
            if (wGot == 0.0) {
                return null;
            }
            boolean cf = c.packetization() == Packetization.CELLS_CF;
            double scale = cf ? wAll / wGot : 1.0;
            for (int a = 0; a < g; a++) {
                if (ok[a] && wt[a] > 0.0) {
                    s.cellMeans.add(means[a]);
                    s.cellWeights.add(wt[a] * scale);
                }
            }
            s.offset = cf ? offGot * scale : offAll;
            return s;
        }

        /** Shuffles {@code records} into packets and applies the channel; returns per-record delivery. */
        private boolean[] deliver(int records) {
            boolean[] ok = new boolean[records];
            Arrays.fill(ok, true);
            if (c.transport() == Transport.RELIABLE || e.rho() == 0.0) {
                lastPackets = (records + perPacket - 1) / perPacket;
                lastLost = 0;
                sentPackets += lastPackets;
                return ok;
            }
            int[] order = shuffled(records);
            int packets = (records + perPacket - 1) / perPacket;
            boolean[] lostPk = new boolean[packets];
            boolean bad = rnd.nextDouble() < ge.lossRate();
            if (c.transport() == Transport.DATAGRAM_RAPTORQ) {
                long total = NetworkCostModel.raptorqPackets(e, packets, r.delta() / 4.0 / e.nodes());
                int arrived = 0;
                for (long j = 0; j < total; j++) {
                    bad = step(bad);
                    if (j < packets) {
                        lostPk[(int) j] = bad;
                    }
                    arrived += bad ? 0 : 1;
                }
                sentPackets += total - packets;
                if (arrived >= packets + 2) {
                    Arrays.fill(lostPk, false); // systematic source packets recovered from repair symbols
                }
            } else if (c.transport() == Transport.DATAGRAM_FEC) {
                int k = c.fecData();
                int p = c.fecParity();
                for (int b0 = 0; b0 < packets; b0 += k) {
                    int dataPk = Math.min(k, packets - b0);
                    boolean[] blk = new boolean[dataPk + p];
                    int lost = 0;
                    for (int j = 0; j < blk.length; j++) {
                        bad = step(bad);
                        blk[j] = bad;
                        lost += bad ? 1 : 0;
                    }
                    for (int j = 0; j < dataPk; j++) {
                        lostPk[b0 + j] = lost > p && blk[j];
                    }
                }
            } else {
                for (int j = 0; j < packets; j++) {
                    bad = step(bad);
                    lostPk[j] = bad;
                }
            }
            for (int j = 0; j < packets; j++) {
                sentPackets++;
                if (lostPk[j]) {
                    erasedPackets++;
                }
            }
            for (int pos = 0; pos < records; pos++) {
                if (lostPk[pos / perPacket]) {
                    ok[order[pos]] = false;
                }
            }
            lastPackets = packets;
            int lostCount = 0;
            for (boolean b : lostPk) {
                lostCount += b ? 1 : 0;
            }
            lastLost = lostCount;
            return ok;
        }

        private boolean step(boolean bad) {
            double u = rnd.nextDouble();
            return bad ? u >= ge.pBadToGood() : u < ge.pGoodToBad();
        }

        private int[] shuffled(int n) {
            int[] p = new int[n];
            for (int i = 0; i < n; i++) {
                p[i] = i;
            }
            for (int i = n - 1; i > 0; i--) {
                int j = rnd.nextInt(i + 1);
                int t = p[i];
                p[i] = p[j];
                p[j] = t;
            }
            return p;
        }

        /** Merge-and-reduce over the leaves: internal nodes may crash (checkpoint: and their peer); root is reliable. */
        private Summary tree(List<Summary> level) {
            double fT = e.failTotal();
            double pInternal = c.checkpoint() ? fT * fT : fT;
            List<Summary> cur = level;
            while (true) {
                int parents = (cur.size() + c.arity() - 1) / c.arity();
                boolean root = parents == 1;
                List<Summary> next = new ArrayList<>(parents);
                for (int p = 0; p < parents; p++) {
                    Summary merged = new Summary();
                    boolean any = false;
                    for (int j = p * c.arity(); j < Math.min(cur.size(), (p + 1) * c.arity()); j++) {
                        Summary ch = cur.get(j);
                        if (ch != null) {
                            merged.addAll(ch, 1.0);
                            any = true;
                        }
                    }
                    if (!any || (!root && rnd.nextDouble() < pInternal)) {
                        next.add(null);
                        continue;
                    }
                    int[] rows = merged.rows();
                    double[] w = merged.weights();
                    Summary red = reduce(rows, w, root ? c.m() : plan.internalDraws(), root);
                    if (red != null) {
                        red.coveredWeight = merged.coveredWeight;
                    }
                    next.add(red);
                }
                if (root) {
                    return next.get(0);
                }
                cur = next;
            }
        }

        /** Resamples a weighted multiset of rows to {@code draws} draws; at the root nothing is transmitted. */
        private Summary reduce(int[] rows, double[] w, long draws, boolean root) {
            if (rows.length == 0) {
                return null;
            }
            if (root) {
                return sampledReliable(rows, w, draws);
            }
            return sampled(rows, w, draws, 1);
        }

        private Summary sampledReliable(int[] rows, double[] w, long draws) {
            PointSet local = PointSet.of(data.select(rows, w, rows.length));
            double[] q = c.family() == Family.SENSITIVITY && local.size() >= r.k()
                    ? FamilyScores.of(local, r.k(), rnd.split()).q(Family.SENSITIVITY)
                    : FamilyScores.of(local, 1, rnd.split()).q(c.family() == Family.SENSITIVITY ? Family.UNIFORM
                            : c.family());
            AliasSampler sampler = AliasSampler.of(q);
            int[] outRows = new int[(int) draws];
            double[] outW = new double[(int) draws];
            for (int d = 0; d < draws; d++) {
                int i = sampler.next(rnd);
                outRows[d] = rows[i];
                outW[d] = w[i] / (q[i] * draws);
            }
            Summary s = new Summary();
            s.addRows(outRows, outW);
            return s;
        }
    }
}
