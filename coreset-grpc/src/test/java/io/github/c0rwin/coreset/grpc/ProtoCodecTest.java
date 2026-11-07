package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.SessionState;
import io.github.c0rwin.coreset.node.TreeMode;
import io.github.c0rwin.coreset.grpc.proto.AckStatus;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ProtoCodecTest {

    @Test
    void roundTripsCoreTypes() {
        SparseVector v = SparseVector.of(10, new int[] {7, 2}, new double[] {-1.5, 3});
        assertEquals(v, ProtoCodec.fromProto(ProtoCodec.toProto(v)));
        Coreset c = new Coreset(List.of(new WeightedPoint(v, 2.5), new WeightedPoint(SparseVector.of(10,
                new int[0], new double[0]), 0)), 4.25);
        assertEquals(c, ProtoCodec.fromProto(ProtoCodec.toProto(c)));
        SessionSpec spec = new SessionSpec("x", Map.of("k", "3"), -5L, 3, 8, TreeMode.CAS);
        assertEquals(spec, ProtoCodec.fromProto(ProtoCodec.toProto(spec)));
        NodeSnapshot s = new NodeSnapshot(c, SessionState.FAILED, "boom", 5, 4, 10.0, 8.0, false);
        assertEquals(s, ProtoCodec.fromProto(ProtoCodec.toProto(s)));
        assertEquals(AckStatus.ACK_STATUS_BUSY, ProtoCodec.ack(3, IngestResult.busy("full")).getStatus());
    }

    @Test
    void wireSizeDependsOnNonZerosNotDimension() {
        SparseVector tiny = SparseVector.of(4, new int[] {1, 3}, new double[] {1, 2});
        SparseVector huge = SparseVector.of(1_000_000_000, new int[] {1, 3}, new double[] {1, 2});
        int a = ProtoCodec.toProto(tiny).getSerializedSize();
        int b = ProtoCodec.toProto(huge).getSerializedSize();
        assertTrue(b - a <= 4, a + " vs " + b); // only the varint of dim grows
        assertTrue(b < 32, "bytes " + b);
    }

    @Test
    void rejectsMalformedVectors() {
        io.github.c0rwin.coreset.grpc.proto.SparseVector mismatch = io.github.c0rwin.coreset.grpc.proto.SparseVector
                .newBuilder().setDim(4).addIndices(1).build();
        assertThrows(IllegalArgumentException.class, () -> ProtoCodec.fromProto(mismatch));
        io.github.c0rwin.coreset.grpc.proto.SparseVector outOfRange = io.github.c0rwin.coreset.grpc.proto.SparseVector
                .newBuilder().setDim(4).addIndices(4).addValues(1).build();
        assertThrows(IllegalArgumentException.class, () -> ProtoCodec.fromProto(outOfRange));
        io.github.c0rwin.coreset.grpc.proto.SparseVector hugeDim = io.github.c0rwin.coreset.grpc.proto.SparseVector
                .newBuilder().setDim(-1).build(); // uint32 0xFFFFFFFF
        assertThrows(IllegalArgumentException.class, () -> ProtoCodec.fromProto(hugeDim));
    }
}
