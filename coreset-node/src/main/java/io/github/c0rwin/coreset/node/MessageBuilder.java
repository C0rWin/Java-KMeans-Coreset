package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.algo.SampledCoreset;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.MeanCells;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.random.RandomGenerator;

/** Turns a snapshot into a {@link NodeMessage} per {@link MessageSpec}. */
final class MessageBuilder {

    private final MessageSpec spec;
    private final NodeSnapshot snap;
    private final long nodePoints;
    private final long stream;

    private MessageBuilder(MessageSpec spec, NodeSnapshot snap, long nodePoints, long stream) {
        this.spec = spec;
        this.snap = snap;
        this.nodePoints = nodePoints;
        this.stream = stream;
    }

    /**
     * @param partitionSeed shared by every replica holding the same summary: fixes q and the cells
     * @param stream        sampling stream: fixes the draws and the record order
     */
    static NodeMessage build(MessageSpec spec, NodeSnapshot snap, long nodePoints, long partitionSeed, long stream,
                             MetricsSink metrics) {
        MessageBuilder b = new MessageBuilder(spec, snap, nodePoints, stream);
        Coreset summary = snap.coreset();
        Packetization form = spec.packetization();
        if (form == Packetization.SUMMARY) {
            PointBlock records = summary.pointSet().compact();
            return b.message(records.size(), summary.offset(), records, null, null);
        }
        if (!(summary.totalWeight() > 0.0)) {
            return b.message(0, summary.offset(), PointBlock.EMPTY, null, null);
        }
        RandomGenerator shared = Seeds.messageShared(partitionSeed);
        RandomGenerator packing = Seeds.messagePacking(partitionSeed, stream);
        if (form.cells()) {
            return b.cells(summary, shared, packing, metrics);
        }
        SampledCoreset sc = new SensitivityCoreset(spec.k(), spec.size(), Seeding.kMeansPlusPlus(spec.lloydIterations()))
                .sample(summary, shared, Seeds.messageDraws(partitionSeed, stream), metrics);
        int[] counts = sc.drawCounts();
        PointSet distinct = sc.coreset().pointSet();
        return form == Packetization.MERGED
                ? b.message(sc.draws(), summary.offset(), distinct.compact(), counts, null)
                : b.perDraw(sc.draws(), counts, distinct, summary.offset(), packing);
    }

    private NodeMessage message(int draws, double offset, PointBlock records, int[] tags, double[] cellOffsets) {
        return new NodeMessage(spec.packetization(), stream, draws, spec.recordsPerPacket(), snap.appliedWeight(),
                nodePoints, snap.complete(), offset, records, tags, cellOffsets);
    }

    /** One record per draw, shuffled (thm:erasure-invariance (ii)); a draw of p weighs w_p / (m q_p). */
    private NodeMessage perDraw(int m, int[] counts, PointSet distinct, double offset, RandomGenerator packing) {
        int[] order = new int[m];
        for (int j = 0, t = 0; j < counts.length; j++) {
            for (int c = counts[j]; c > 0; c--) {
                order[t++] = j;
            }
        }
        shuffle(order, packing);
        double[] weights = new double[m];
        for (int t = 0; t < m; t++) {
            int j = order[t];
            weights[t] = distinct.weight(j) / counts[j];
        }
        return message(m, offset, distinct.select(order, weights, m), order, null);
    }

    /** Mean coreset over at most {@code spec.size()} cells, in shuffled order, offsets per cell or summed. */
    private NodeMessage cells(Coreset summary, RandomGenerator shared, RandomGenerator packing, MetricsSink metrics) {
        PointSet points = summary.pointSet();
        int n = points.size();
        int[] assignment;
        int parts;
        if (n <= spec.size()) {
            assignment = new int[n];
            for (int i = 0; i < n; i++) {
                assignment[i] = i;
            }
            parts = n;
        } else {
            // Plain D^2 seeding as in MeanCoreset: the per-cell identity holds for any partition.
            KMeansResult r = WeightedLloyd.cluster(points, spec.size(), spec.lloydIterations(), 1, shared, metrics);
            assignment = r.assignment();
            parts = r.centers().size();
        }
        MeanCells mc = WeightedLloyd.meanCells(points, assignment, parts);
        int cells = mc.means().size();
        int[] order = new int[cells];
        for (int i = 0; i < cells; i++) {
            order[i] = i;
        }
        shuffle(order, packing);
        double[] weights = new double[cells];
        double[] offsets = mc.offsets();
        double[] shuffled = new double[cells];
        double offsetSum = 0.0;
        for (int t = 0; t < cells; t++) {
            weights[t] = mc.means().weight(order[t]);
            shuffled[t] = offsets[order[t]];
            offsetSum += offsets[t];
        }
        PointBlock records = PointSet.of(mc.means()).select(order, weights, cells);
        return spec.packetization() == Packetization.MEAN_CF_TRIPLES
                ? message(cells, summary.offset(), records, null, shuffled)
                : message(cells, summary.offset() + offsetSum, records, null, null);
    }

    private static void shuffle(int[] a, RandomGenerator rng) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }
}
