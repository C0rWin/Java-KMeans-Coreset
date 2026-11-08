package io.github.c0rwin.coreset.grpc.resilience;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.grpc.proto.SparseVector;
import io.github.c0rwin.coreset.grpc.proto.WeightedPoint;
import io.github.c0rwin.coreset.node.Packetization;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Decodes what arrived of one message, or of the pooled messages of a rateless group, into a weighted set. Records go
 * from the packets straight into one exactly sized block (two passes, no per-point objects); per-draw records of the
 * same distinct draw (equal tag) are merged by adding weights.
 */
public final class MessageDecoder {

    /**
     * @param coreset          decoded weighted set, offsets included
     * @param decoderScale     factor applied to the received weights of the first source (before survivor scaling)
     * @param recordsReceived  delivered records over all sources
     * @param drawsReceived    delivered draws over all sources (draw counts for MERGED)
     */
    public record Decoded(Coreset coreset, double decoderScale, long recordsReceived, double drawsReceived) {
    }

    private MessageDecoder() {
    }

    /**
     * Decodes {@code sources} as one estimate: a single message, or the replicas of a rateless group sampling the
     * same q with independent streams (thm:rateless (a)). The reliable offset is taken once, from the first source.
     *
     * @param knownLoss  loss probability for {@link Decoder#HORVITZ_THOMPSON}
     * @param multiplier further factor on every weight and offset (survivor reweighting)
     * @throws IllegalArgumentException if the sources mix packetizations or a packet is malformed
     */
    public static Decoded decode(List<ReceivedMessage> sources, Decoder decoder, double knownLoss, double multiplier,
                                 int maxDim) {
        if (sources.isEmpty()) {
            throw new IllegalArgumentException("nothing to decode");
        }
        if (!(multiplier >= 0.0) || Double.isInfinite(multiplier)) {
            throw new IllegalArgumentException("multiplier must be finite and >= 0: " + multiplier);
        }
        Packetization form = sources.get(0).packetization();
        long records = 0;
        double draws = 0.0;
        double sentWeight = 0.0;
        double receivedWeight = 0.0;
        for (ReceivedMessage s : sources) {
            if (s.packetization() != form) {
                throw new IllegalArgumentException("cannot pool " + form.id() + " with " + s.packetization().id());
            }
            records += s.recordsReceived();
            draws += s.drawsReceived();
            sentWeight += s.header().getRecordWeight();
            receivedWeight += s.weightReceived();
        }
        double[] scale = new double[sources.size()];
        for (int i = 0; i < scale.length; i++) {
            scale[i] = switch (decoder) {
                case AS_SENT -> 1.0;
                case HORVITZ_THOMPSON -> {
                    if (!(knownLoss >= 0.0 && knownLoss < 1.0)) {
                        throw new IllegalArgumentException("known loss must be in [0, 1): " + knownLoss);
                    }
                    yield 1.0 / (1.0 - knownLoss);
                }
                case REALIZED_COUNT -> draws > 0 ? sources.get(i).drawsSent() / draws : 0.0;
                case REALIZED_WEIGHT -> receivedWeight > 0 ? sentWeight / receivedWeight : 0.0;
            };
        }
        double baseOffset = sources.get(0).header().getOffset();
        if (!(baseOffset >= 0.0) || Double.isInfinite(baseOffset)) {
            throw new IllegalArgumentException("offset must be finite and >= 0: " + baseOffset);
        }
        Rows rows = new Rows(sources, form == Packetization.PER_DRAW, maxDim);
        double cellOffsets = 0.0;
        double[] weights = new double[rows.count];
        int[] offsets = new int[rows.count + 1];
        int[] idx = new int[(int) rows.entries];
        double[] val = new double[(int) rows.entries];
        int row = 0;
        int e = 0;
        for (int s = 0; s < scale.length; s++) {
            int[] rowOfTag = rows.resetTags(sources.get(s));
            double factor = scale[s] * multiplier;
            for (Packet p : sources.get(s).packets()) {
                for (int i = 0, n = p.getRecordsCount(); i < n; i++) {
                    WeightedPoint r = p.getRecords(i);
                    double w = r.getWeight() * factor;
                    if (p.getCellOffsetsCount() > 0) {
                        cellOffsets += p.getCellOffsets(i) * factor;
                    }
                    if (rowOfTag != null) {
                        int tag = p.getTags(i);
                        if (rowOfTag[tag] >= 0) {
                            weights[rowOfTag[tag]] += w;
                            continue;
                        }
                        rowOfTag[tag] = row;
                    }
                    SparseVector v = r.getVector();
                    for (int k = 0, nnz = v.getIndicesCount(); k < nnz; k++, e++) {
                        idx[e] = v.getIndices(k);
                        val[e] = v.getValues(k);
                    }
                    weights[row] = w;
                    offsets[++row] = e;
                }
            }
        }
        PointBlock block = rows.count == 0 ? PointBlock.EMPTY : PointBlock.wrap(rows.dim, offsets, idx, val, weights);
        return new Decoded(new Coreset(block, baseOffset * multiplier + cellOffsets), scale[0], records, draws);
    }

    /**
     * The messages of a fractional-repetition group are identical, so each packet seq is needed from one member only:
     * the union, keyed by seq, of the members' delivered packets under the first member's header. Members whose header
     * differs (the summaries diverged) are left out.
     */
    public static ReceivedMessage unionOfIdentical(List<ReceivedMessage> members) {
        ReceivedMessage first = members.get(0);
        MessageHeader h = first.header();
        Packet[] bySeq = new Packet[h.getPackets()];
        for (ReceivedMessage m : members) {
            MessageHeader o = m.header();
            if (o.getRecords() != h.getRecords() || o.getDraws() != h.getDraws() || o.getStream() != h.getStream()
                    || o.getPacketization() != h.getPacketization() || o.getRecordWeight() != h.getRecordWeight()) {
                continue;
            }
            for (Packet p : m.packets()) {
                int seq = p.getSeq();
                if (seq >= 0 && seq < bySeq.length && bySeq[seq] == null) {
                    bySeq[seq] = p;
                }
            }
        }
        List<Packet> union = new ArrayList<>(bySeq.length);
        for (Packet p : bySeq) {
            if (p != null) {
                union.add(p);
            }
        }
        return new ReceivedMessage(first.node(), h, union);
    }

    /** First pass: output rows and entries, dimension and packet shape checks. */
    private static final class Rows {
        int count;
        long entries;
        int dim = -1;
        private final boolean mergeTags;
        private int[] tagRows;

        Rows(List<ReceivedMessage> sources, boolean mergeTags, int maxDim) {
            this.mergeTags = mergeTags;
            for (ReceivedMessage s : sources) {
                int[] seen = resetTags(s);
                for (Packet p : s.packets()) {
                    int n = p.getRecordsCount();
                    if (mergeTags && p.getTagsCount() != n
                            || p.getCellOffsetsCount() != 0 && p.getCellOffsetsCount() != n) {
                        throw new IllegalArgumentException("packet " + p.getSeq() + " has " + n + " records, "
                                + p.getTagsCount() + " tags and " + p.getCellOffsetsCount() + " cell offsets");
                    }
                    for (int i = 0; i < n; i++) {
                        if (seen != null) {
                            int tag = p.getTags(i);
                            if (tag < 0 || tag >= s.drawsSent()) {
                                throw new IllegalArgumentException("tag " + Integer.toUnsignedString(tag)
                                        + " outside the " + s.drawsSent() + " draws");
                            }
                            if (seen[tag] >= 0) {
                                continue;
                            }
                            seen[tag] = 0;
                        }
                        WeightedPoint r = p.getRecords(i);
                        SparseVector v = r.getVector();
                        int d = v.getDim();
                        if (!r.hasVector() || d < 0 || d > maxDim || dim >= 0 && d != dim) {
                            throw new IllegalArgumentException("record dim " + Integer.toUnsignedString(d)
                                    + " invalid (limit " + maxDim + ", message dim " + dim + ")");
                        }
                        if (v.getIndicesCount() != v.getValuesCount()) {
                            throw new IllegalArgumentException("indices and values lengths differ");
                        }
                        dim = d;
                        count++;
                        entries += v.getIndicesCount();
                    }
                }
            }
            if (entries > Integer.MAX_VALUE - 8) {
                throw new IllegalArgumentException("more than 2^31 entries: " + entries);
            }
        }

        /** Tag-to-row map whose first {@code s.drawsSent()} entries are -1, or null when records are not merged. */
        int[] resetTags(ReceivedMessage s) {
            if (!mergeTags) {
                return null;
            }
            long draws = s.drawsSent();
            if (draws > Integer.MAX_VALUE - 8) {
                throw new IllegalArgumentException("too many draws: " + draws);
            }
            if (tagRows == null || tagRows.length < draws) {
                tagRows = new int[(int) draws];
            }
            Arrays.fill(tagRows, 0, (int) draws, -1);
            return tagRows;
        }
    }
}
