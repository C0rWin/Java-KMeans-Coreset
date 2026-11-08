package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.AckStatus;
import io.github.c0rwin.coreset.grpc.proto.SnapshotResponse;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.Messages;
import io.github.c0rwin.coreset.node.NodeLimits;
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

    /**
     * Decodes one vector. The dimension is checked against {@code maxDim} before anything is allocated; the packed
     * arrays are copied once, into arrays the vector then owns ({@code SparseVector.ofOwned}).
     *
     * @throws IllegalArgumentException if the message is not a valid vector or its dimension exceeds {@code maxDim}
     */
    public static SparseVector fromProto(io.github.c0rwin.coreset.grpc.proto.SparseVector v, int maxDim) {
        int dim = v.getDim(); // uint32 above 2^31 arrives negative and is rejected here
        if (dim < 0 || dim > maxDim) {
            throw new IllegalArgumentException("dim " + Integer.toUnsignedString(dim) + " exceeds the limit of "
                    + maxDim);
        }
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
        return SparseVector.ofOwned(dim, idx, val);
    }

    public static io.github.c0rwin.coreset.grpc.proto.WeightedPoint toProto(WeightedPoint p) {
        return io.github.c0rwin.coreset.grpc.proto.WeightedPoint.newBuilder()
                .setVector(toProto(p.vector()))
                .setWeight(p.weight())
                .build();
    }

    public static WeightedPoint fromProto(io.github.c0rwin.coreset.grpc.proto.WeightedPoint p, int maxDim) {
        if (!p.hasVector()) {
            throw new IllegalArgumentException("point without vector");
        }
        return new WeightedPoint(fromProto(p.getVector(), maxDim), p.getWeight());
    }

    public static List<io.github.c0rwin.coreset.grpc.proto.WeightedPoint> pointsToProto(List<WeightedPoint> points) {
        List<io.github.c0rwin.coreset.grpc.proto.WeightedPoint> out = new ArrayList<>(points.size());
        for (WeightedPoint p : points) {
            out.add(toProto(p));
        }
        return out;
    }

    /**
     * Decodes into an immutable list that {@code List.copyOf} (node ingest, {@code Coreset}) takes by reference: one
     * pass, one reference array plus the copy {@code List.of} makes.
     */
    public static List<WeightedPoint> pointsFromProto(List<io.github.c0rwin.coreset.grpc.proto.WeightedPoint> points,
                                                      int maxDim) {
        WeightedPoint[] out = new WeightedPoint[points.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = fromProto(points.get(i), maxDim);
        }
        return List.of(out);
    }

    public static List<io.github.c0rwin.coreset.grpc.proto.SparseVector> vectorsToProto(List<SparseVector> vectors) {
        List<io.github.c0rwin.coreset.grpc.proto.SparseVector> out = new ArrayList<>(vectors.size());
        for (SparseVector v : vectors) {
            out.add(toProto(v));
        }
        return out;
    }

    public static List<SparseVector> vectorsFromProto(List<io.github.c0rwin.coreset.grpc.proto.SparseVector> vectors,
                                                      int maxDim) {
        SparseVector[] out = new SparseVector[vectors.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = fromProto(vectors.get(i), maxDim);
        }
        return List.of(out);
    }

    /** Encodes straight from the segments' arrays: no per-point domain objects. */
    public static io.github.c0rwin.coreset.grpc.proto.Coreset toProto(Coreset c) {
        PointSet points = c.pointSet();
        io.github.c0rwin.coreset.grpc.proto.Coreset.Builder out =
                io.github.c0rwin.coreset.grpc.proto.Coreset.newBuilder().setOffset(c.offset());
        for (int t = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0, n = b.size(); i < n; i++) {
                io.github.c0rwin.coreset.grpc.proto.SparseVector.Builder v =
                        io.github.c0rwin.coreset.grpc.proto.SparseVector.newBuilder().setDim(b.dim());
                for (int e = b.rowStart(i), end = b.rowEnd(i); e < end; e++) {
                    v.addIndices(b.index(e));
                    v.addValues(b.value(e));
                }
                out.addPoints(io.github.c0rwin.coreset.grpc.proto.WeightedPoint.newBuilder()
                        .setVector(v).setWeight(b.weight(i)));
            }
        }
        return out.build();
    }

    /** Decodes straight into one exactly sized {@link PointBlock} ({@link PointBlock#wrap}): no per-point objects. */
    public static Coreset fromProto(io.github.c0rwin.coreset.grpc.proto.Coreset c, int maxDim) {
        return new Coreset(blockFromProto(c.getPointsList(), maxDim), c.getOffset());
    }

    /**
     * Decodes points into one block: a first pass checks dimensions and sizes the five arrays exactly, the second
     * fills them, and {@link PointBlock#wrap} validates and canonicalises in place.
     *
     * @throws IllegalArgumentException if a point has no vector, an invalid vector or weight, a dimension above
     *     {@code maxDim}, or the points differ in dimension
     */
    public static PointBlock blockFromProto(List<io.github.c0rwin.coreset.grpc.proto.WeightedPoint> points,
                                            int maxDim) {
        final int n = points.size();
        if (n == 0) {
            return PointBlock.EMPTY;
        }
        int dim = -1;
        long entries = 0;
        for (int i = 0; i < n; i++) {
            io.github.c0rwin.coreset.grpc.proto.WeightedPoint p = points.get(i);
            if (!p.hasVector()) {
                throw new IllegalArgumentException("point without vector");
            }
            io.github.c0rwin.coreset.grpc.proto.SparseVector v = p.getVector();
            int d = v.getDim(); // uint32 above 2^31 arrives negative and is rejected here
            if (d < 0 || d > maxDim) {
                throw new IllegalArgumentException("dim " + Integer.toUnsignedString(d) + " exceeds the limit of "
                        + maxDim);
            }
            if (dim >= 0 && d != dim) {
                throw new IllegalArgumentException("point " + i + " has dim " + d + ", expected " + dim);
            }
            dim = d;
            if (v.getIndicesCount() != v.getValuesCount()) {
                throw new IllegalArgumentException("indices and values lengths differ: " + v.getIndicesCount()
                        + " vs " + v.getValuesCount());
            }
            entries += v.getIndicesCount();
        }
        if (entries > Integer.MAX_VALUE - 8) {
            throw new IllegalArgumentException("more than 2^31 entries: " + entries);
        }
        int[] offsets = new int[n + 1];
        int[] idx = new int[(int) entries];
        double[] val = new double[(int) entries];
        double[] weights = new double[n];
        int e = 0;
        for (int i = 0; i < n; i++) {
            io.github.c0rwin.coreset.grpc.proto.WeightedPoint p = points.get(i);
            io.github.c0rwin.coreset.grpc.proto.SparseVector v = p.getVector();
            for (int k = 0, nnz = v.getIndicesCount(); k < nnz; k++, e++) {
                idx[e] = v.getIndices(k);
                val[e] = v.getValues(k);
            }
            offsets[i + 1] = e;
            weights[i] = p.getWeight();
        }
        return PointBlock.wrap(dim, offsets, idx, val, weights);
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

    /**
     * Checks the algorithm id and params against {@code limits} on the wire message, before the map is copied.
     *
     * @throws IllegalArgumentException if the spec is invalid or outside the limits
     */
    public static SessionSpec fromProto(io.github.c0rwin.coreset.grpc.proto.SessionSpec s, NodeLimits limits) {
        TreeMode mode = switch (s.getTreeMode()) {
            case TREE_MODE_UNSPECIFIED, TREE_MODE_SINGLE_WRITER -> TreeMode.SINGLE_WRITER;
            case TREE_MODE_CAS -> TreeMode.CAS;
            default -> throw new IllegalArgumentException("unknown tree mode: " + s.getTreeModeValue());
        };
        limits.checkAlgorithm(s.getAlgorithmId(), s.getParamsMap());
        if (Integer.compareUnsigned(s.getQueueCapacity(), limits.maxQueueCapacity()) > 0) {
            throw new IllegalArgumentException("queueCapacity " + Integer.toUnsignedString(s.getQueueCapacity())
                    + " exceeds the node limit of " + limits.maxQueueCapacity());
        }
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
                .setMessage(Messages.bounded(r.reason()))
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

    public static NodeSnapshot fromProto(SnapshotResponse r, int maxDim) {
        SessionState state = switch (r.getState()) {
            case SESSION_STATE_FAILED -> SessionState.FAILED;
            case SESSION_STATE_CLOSED -> SessionState.CLOSED;
            case SESSION_STATE_ACTIVE -> SessionState.ACTIVE;
            default -> throw new IllegalArgumentException("unknown session state: " + r.getStateValue());
        };
        return new NodeSnapshot(fromProto(r.getCoreset(), maxDim), state, r.getFailure(), r.getAcceptedBatches(),
                r.getAppliedBatches(), r.getAcceptedWeight(), r.getAppliedWeight(), r.getConsistent());
    }
}
