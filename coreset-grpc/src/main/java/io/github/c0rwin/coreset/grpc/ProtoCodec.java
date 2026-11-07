package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.AckStatus;
import io.github.c0rwin.coreset.grpc.proto.SnapshotResponse;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.SessionState;
import io.github.c0rwin.coreset.node.TreeMode;
import java.util.ArrayList;
import java.util.List;

/** Conversions between coreset-core/node types and protobuf messages; vectors travel as O(nnz) packed arrays. */
public final class ProtoCodec {

    private ProtoCodec() {
    }

    public static io.github.c0rwin.coreset.grpc.proto.SparseVector toProto(SparseVector v) {
        io.github.c0rwin.coreset.grpc.proto.SparseVector.Builder b =
                io.github.c0rwin.coreset.grpc.proto.SparseVector.newBuilder().setDim(v.dim());
        for (int k = 0, n = v.nnz(); k < n; k++) {
            b.addIndices(v.indexAt(k));
            b.addValues(v.valueAt(k));
        }
        return b.build();
    }

    /** @throws IllegalArgumentException if the message is not a valid vector */
    public static SparseVector fromProto(io.github.c0rwin.coreset.grpc.proto.SparseVector v) {
        int n = v.getIndicesCount();
        if (n != v.getValuesCount()) {
            throw new IllegalArgumentException("indices and values lengths differ: " + n + " vs " + v.getValuesCount());
        }
        int[] idx = new int[n];
        double[] val = new double[n];
        for (int k = 0; k < n; k++) {
            idx[k] = v.getIndices(k);
            val[k] = v.getValues(k);
        }
        return SparseVector.of(v.getDim(), idx, val); // uint32 above 2^31 arrives negative and is rejected here
    }

    public static io.github.c0rwin.coreset.grpc.proto.WeightedPoint toProto(WeightedPoint p) {
        return io.github.c0rwin.coreset.grpc.proto.WeightedPoint.newBuilder()
                .setVector(toProto(p.vector()))
                .setWeight(p.weight())
                .build();
    }

    public static WeightedPoint fromProto(io.github.c0rwin.coreset.grpc.proto.WeightedPoint p) {
        if (!p.hasVector()) {
            throw new IllegalArgumentException("point without vector");
        }
        return new WeightedPoint(fromProto(p.getVector()), p.getWeight());
    }

    public static List<io.github.c0rwin.coreset.grpc.proto.WeightedPoint> pointsToProto(List<WeightedPoint> points) {
        List<io.github.c0rwin.coreset.grpc.proto.WeightedPoint> out = new ArrayList<>(points.size());
        for (WeightedPoint p : points) {
            out.add(toProto(p));
        }
        return out;
    }

    public static List<WeightedPoint> pointsFromProto(List<io.github.c0rwin.coreset.grpc.proto.WeightedPoint> points) {
        List<WeightedPoint> out = new ArrayList<>(points.size());
        for (io.github.c0rwin.coreset.grpc.proto.WeightedPoint p : points) {
            out.add(fromProto(p));
        }
        return out;
    }

    public static List<io.github.c0rwin.coreset.grpc.proto.SparseVector> vectorsToProto(List<SparseVector> vectors) {
        List<io.github.c0rwin.coreset.grpc.proto.SparseVector> out = new ArrayList<>(vectors.size());
        for (SparseVector v : vectors) {
            out.add(toProto(v));
        }
        return out;
    }

    public static List<SparseVector> vectorsFromProto(List<io.github.c0rwin.coreset.grpc.proto.SparseVector> vectors) {
        List<SparseVector> out = new ArrayList<>(vectors.size());
        for (io.github.c0rwin.coreset.grpc.proto.SparseVector v : vectors) {
            out.add(fromProto(v));
        }
        return out;
    }

    public static io.github.c0rwin.coreset.grpc.proto.Coreset toProto(Coreset c) {
        return io.github.c0rwin.coreset.grpc.proto.Coreset.newBuilder()
                .addAllPoints(pointsToProto(c.points()))
                .setOffset(c.offset())
                .build();
    }

    public static Coreset fromProto(io.github.c0rwin.coreset.grpc.proto.Coreset c) {
        return new Coreset(pointsFromProto(c.getPointsList()), c.getOffset());
    }

    public static io.github.c0rwin.coreset.grpc.proto.SessionSpec toProto(SessionSpec s) {
        return io.github.c0rwin.coreset.grpc.proto.SessionSpec.newBuilder()
                .setAlgorithmId(s.algorithmId())
                .putAllParams(s.params())
                .setSeed(s.seed())
                .setWorkers(s.workers())
                .setQueueCapacity(s.queueCapacity())
                .setTreeMode(s.treeMode() == TreeMode.CAS
                        ? io.github.c0rwin.coreset.grpc.proto.TreeMode.TREE_MODE_CAS
                        : io.github.c0rwin.coreset.grpc.proto.TreeMode.TREE_MODE_SINGLE_WRITER)
                .build();
    }

    /** @throws IllegalArgumentException if the spec is invalid */
    public static SessionSpec fromProto(io.github.c0rwin.coreset.grpc.proto.SessionSpec s) {
        TreeMode mode = switch (s.getTreeMode()) {
            case TREE_MODE_UNSPECIFIED, TREE_MODE_SINGLE_WRITER -> TreeMode.SINGLE_WRITER;
            case TREE_MODE_CAS -> TreeMode.CAS;
            default -> throw new IllegalArgumentException("unknown tree mode: " + s.getTreeModeValue());
        };
        return new SessionSpec(s.getAlgorithmId(), s.getParamsMap(), s.getSeed(), s.getWorkers(),
                s.getQueueCapacity(), mode);
    }

    public static Ack ack(long seq, IngestResult r) {
        AckStatus status = switch (r.status()) {
            case ACCEPTED -> AckStatus.ACK_STATUS_ACCEPTED;
            case DUPLICATE -> AckStatus.ACK_STATUS_DUPLICATE;
            case BUSY -> AckStatus.ACK_STATUS_BUSY;
            case FAILED -> AckStatus.ACK_STATUS_FAILED;
        };
        return Ack.newBuilder()
                .setSeq(seq)
                .setStatus(status)
                .setMessage(r.reason())
                .setAcceptedPoints(r.acceptedPoints())
                .build();
    }

    public static SnapshotResponse toProto(NodeSnapshot s) {
        io.github.c0rwin.coreset.grpc.proto.SessionState state = switch (s.state()) {
            case ACTIVE -> io.github.c0rwin.coreset.grpc.proto.SessionState.SESSION_STATE_ACTIVE;
            case FAILED -> io.github.c0rwin.coreset.grpc.proto.SessionState.SESSION_STATE_FAILED;
            case CLOSED -> io.github.c0rwin.coreset.grpc.proto.SessionState.SESSION_STATE_CLOSED;
        };
        return SnapshotResponse.newBuilder()
                .setCoreset(toProto(s.coreset()))
                .setState(state)
                .setFailure(s.failure())
                .setAcceptedBatches(s.acceptedBatches())
                .setAppliedBatches(s.appliedBatches())
                .setAcceptedWeight(s.acceptedWeight())
                .setAppliedWeight(s.appliedWeight())
                .setPendingWeight(s.pendingWeight())
                .setConsistent(s.consistent())
                .setComplete(s.complete())
                .build();
    }

    public static NodeSnapshot fromProto(SnapshotResponse r) {
        SessionState state = switch (r.getState()) {
            case SESSION_STATE_FAILED -> SessionState.FAILED;
            case SESSION_STATE_CLOSED -> SessionState.CLOSED;
            case SESSION_STATE_ACTIVE -> SessionState.ACTIVE;
            default -> throw new IllegalArgumentException("unknown session state: " + r.getStateValue());
        };
        return new NodeSnapshot(fromProto(r.getCoreset()), state, r.getFailure(), r.getAcceptedBatches(),
                r.getAppliedBatches(), r.getAcceptedWeight(), r.getAppliedWeight(), r.getConsistent());
    }
}
