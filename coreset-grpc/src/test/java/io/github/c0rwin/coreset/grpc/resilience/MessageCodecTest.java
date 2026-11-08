package io.github.c0rwin.coreset.grpc.resilience;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.c0rwin.coreset.grpc.MessageCodec;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.node.MessageSpec;
import io.github.c0rwin.coreset.node.NodeMessage;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.Test;

class MessageCodecTest {

    private static List<Packet> packets(NodeMessage m) {
        List<Packet> out = new ArrayList<>();
        for (int p = 0; p < m.packets(); p++) {
            out.add(MessageCodec.packet(m, p));
        }
        return out;
    }

    @Test
    void aMessageSurvivesTheWireInAnyPacketOrder() throws Exception {
        for (MessageSpec spec : List.of(MessageSpec.perDraw(100, 3, 7), MessageSpec.merged(100, 3, 7),
                MessageSpec.cfTriples(9, 2, 10), MessageSpec.SUMMARY)) {
            try (TestMessages tm = new TestMessages(TestMessages.mixture(500, 4), spec)) {
                NodeMessage m = tm.message(3);
                MessageHeader h = MessageCodec.header(m, 2, false).build();
                assertEquals((m.recordCount() + m.recordsPerPacket() - 1) / m.recordsPerPacket(), h.getPackets());
                List<Packet> shuffled = packets(m);
                Collections.reverse(shuffled);
                NodeMessage back = MessageCodec.decode(h, shuffled, TestMessages.MAX_DIM, 1 << 20);
                assertEquals(m.records(), back.records(), spec.toString());
                assertArrayEquals(m.tags(), back.tags());
                assertArrayEquals(m.cellOffsets(), back.cellOffsets());
                assertEquals(m.draws(), back.draws());
                assertEquals(m.offset(), back.offset());
                assertEquals(m.nodeWeight(), back.nodeWeight());
                assertEquals(m.stream(), back.stream());
            }
        }
    }

    @Test
    void incompleteOrInconsistentMessagesAreRefused() throws Exception {
        try (TestMessages tm = new TestMessages(TestMessages.mixture(500, 5), MessageSpec.perDraw(64, 3, 8))) {
            NodeMessage m = tm.message(0);
            MessageHeader h = MessageCodec.header(m, 0, false).build();
            List<Packet> all = packets(m);
            assertThrows(IllegalArgumentException.class,
                    () -> MessageCodec.decode(h, all.subList(1, all.size()), TestMessages.MAX_DIM, 1 << 20));
            List<Packet> dup = new ArrayList<>(all);
            dup.set(1, all.get(0));
            assertThrows(IllegalArgumentException.class,
                    () -> MessageCodec.decode(h, dup, TestMessages.MAX_DIM, 1 << 20));
            assertThrows(IllegalArgumentException.class,
                    () -> MessageCodec.decode(h, all, TestMessages.MAX_DIM, 10)); // over the record limit
            assertThrows(IllegalArgumentException.class, () -> MessageCodec.packet(m, m.packets()));
        }
    }
}
