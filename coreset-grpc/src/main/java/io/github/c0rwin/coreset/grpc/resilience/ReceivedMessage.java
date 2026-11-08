package io.github.c0rwin.coreset.grpc.resilience;

import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.node.Packetization;
import java.util.List;
import java.util.Objects;

/**
 * What arrived of one message: its reliable header and the packets that were delivered, in arrival order.
 *
 * @param node    node that served it (the owner, or a replica holder)
 * @param header  reliable header
 * @param packets delivered packets
 */
public record ReceivedMessage(int node, MessageHeader header, List<Packet> packets) {

    public ReceivedMessage {
        Objects.requireNonNull(header, "header");
        packets = List.copyOf(packets);
    }

    public Packetization packetization() {
        return ProtoCodec.fromProto(header.getPacketization());
    }

    public long recordsReceived() {
        long n = 0;
        for (Packet p : packets) {
            n += p.getRecordsCount();
        }
        return n;
    }

    /** Draws carried by the delivered records: their draw counts for MERGED, one per record otherwise. */
    public double drawsReceived() {
        if (packetization() != Packetization.MERGED) {
            return recordsReceived();
        }
        double n = 0;
        for (Packet p : packets) {
            for (int i = 0, t = p.getTagsCount(); i < t; i++) {
                n += Integer.toUnsignedLong(p.getTags(i));
            }
        }
        return n;
    }

    /** Record weight delivered, as sent. */
    public double weightReceived() {
        double w = 0.0;
        for (Packet p : packets) {
            for (int i = 0, n = p.getRecordsCount(); i < n; i++) {
                w += p.getRecords(i).getWeight();
            }
        }
        return w;
    }

    /** Draws (or records) the message was sent with. */
    public long drawsSent() {
        return Integer.toUnsignedLong(header.getDraws());
    }
}
