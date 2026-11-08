package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.grpc.proto.SparseVector;
import io.github.c0rwin.coreset.grpc.proto.WeightedPoint;
import io.github.c0rwin.coreset.node.NodeMessage;
import java.util.ArrayList;
import java.util.List;

/** Wire form of a {@link NodeMessage}: one reliable {@link MessageHeader} and its {@link Packet}s. */
public final class MessageCodec {

    private MessageCodec() {
    }

    /** Header of {@code m} as served for {@code owner}. */
    public static MessageHeader.Builder header(NodeMessage m, long owner, boolean replica) {
        return MessageHeader.newBuilder()
                .setOwner(owner)
                .setReplica(replica)
                .setPacketization(ProtoCodec.toProto(m.packetization()))
                .setStream(m.stream())
                .setDraws(m.draws())
                .setRecords(m.recordCount())
                .setPackets(m.packets())
                .setRecordsPerPacket(m.recordsPerPacket())
                .setNodeWeight(m.nodeWeight())
                .setNodePoints(m.nodePoints())
                .setOffset(m.offset())
                .setRecordWeight(m.recordWeight())
                .setComplete(m.complete());
    }

    /** Packet {@code seq}: records {@code [seq * B, min((seq + 1) * B, records))}, encoded from the block's arrays. */
    public static Packet packet(NodeMessage m, int seq) {
        PointBlock r = m.records();
        int from = seq * m.recordsPerPacket();
        int to = Math.min(r.size(), from + m.recordsPerPacket());
        if (seq < 0 || from >= to) {
            throw new IllegalArgumentException("packet " + seq + " out of range: " + m.packets() + " packets");
        }
        Packet.Builder p = Packet.newBuilder().setSeq(seq);
        int[] tags = m.tags();
        double[] cells = m.cellOffsets();
        for (int i = from; i < to; i++) {
            SparseVector.Builder v = SparseVector.newBuilder().setDim(r.dim());
            for (int e = r.rowStart(i), end = r.rowEnd(i); e < end; e++) {
                v.addIndices(r.index(e));
                v.addValues(r.value(e));
            }
            p.addRecords(WeightedPoint.newBuilder().setVector(v).setWeight(r.weight(i)));
            if (tags.length != 0) {
                p.addTags(tags[i]);
            }
            if (cells.length != 0) {
                p.addCellOffsets(cells[i]);
            }
        }
        return p.build();
    }

    /**
     * Reassembles a complete message (a stored replica): every packet {@code 0 .. packets-1} exactly once, in any
     * order.
     *
     * @throws IllegalArgumentException if packets are missing, duplicated or inconsistent with the header
     */
    public static NodeMessage decode(MessageHeader h, List<Packet> packets, int maxDim, int maxRecords) {
        int records = h.getRecords();
        int b = h.getRecordsPerPacket();
        if (records < 0 || records > maxRecords || b < 1) {
            throw new IllegalArgumentException("header has " + Integer.toUnsignedString(records)
                    + " records (limit " + maxRecords + ") and " + Integer.toUnsignedString(b) + " per packet");
        }
        int expected = (records + b - 1) / b;
        if (packets.size() != expected || h.getPackets() != expected) {
            throw new IllegalArgumentException("expected " + expected + " packets, got " + packets.size());
        }
        Packet[] bySeq = new Packet[expected];
        for (Packet p : packets) {
            int seq = p.getSeq();
            if (seq < 0 || seq >= expected || bySeq[seq] != null) {
                throw new IllegalArgumentException("packet seq " + Integer.toUnsignedString(seq)
                        + " missing from or repeated in " + expected);
            }
            bySeq[seq] = p;
        }
        List<WeightedPoint> points = new ArrayList<>(records);
        boolean tagged = expected > 0 && bySeq[0].getTagsCount() > 0;
        boolean cells = expected > 0 && bySeq[0].getCellOffsetsCount() > 0;
        int[] tags = tagged ? new int[records] : null;
        double[] offsets = cells ? new double[records] : null;
        int at = 0;
        for (int seq = 0; seq < expected; seq++) {
            Packet p = bySeq[seq];
            int n = p.getRecordsCount();
            if (n != Math.min(b, records - seq * b) || tagged && p.getTagsCount() != n
                    || cells && p.getCellOffsetsCount() != n) {
                throw new IllegalArgumentException("packet " + seq + " has " + n + " records, "
                        + p.getTagsCount() + " tags and " + p.getCellOffsetsCount() + " cell offsets");
            }
            points.addAll(p.getRecordsList());
            for (int i = 0; i < n; i++, at++) {
                if (tagged) {
                    tags[at] = p.getTags(i);
                }
                if (cells) {
                    offsets[at] = p.getCellOffsets(i);
                }
            }
        }
        PointBlock block = ProtoCodec.blockFromProto(points, maxDim);
        return new NodeMessage(ProtoCodec.fromProto(h.getPacketization()), h.getStream(), h.getDraws(), b,
                h.getNodeWeight(), h.getNodePoints(), h.getComplete(), h.getOffset(), block, tags, offsets);
    }
}
