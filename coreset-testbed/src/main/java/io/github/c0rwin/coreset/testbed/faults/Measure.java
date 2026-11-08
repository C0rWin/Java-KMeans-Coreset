package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.testbed.problem.Centers;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.SplittableRandom;

/** Per-run metrics: distortion over the candidate pool, downstream ratio, and wire sizes. */
final class Measure {

    /** Lloyd budget of the deployed solution on a decoded coreset. */
    static final int LLOYD_ITERATIONS = 20;
    static final int RESTARTS = 3;
    /** Per-packet header bytes (IPv4 + UDP) in the packet model of F3. */
    static final int PACKET_HEADER = 28;

    private Measure() {
    }

    /** ESA'22 distortion over the pool: {@code max_j max(1 + r_j, 1 / (1 + r_j))}. */
    static double distortion(double[] r) {
        double d = 1.0;
        for (double x : r) {
            double ratio = 1.0 + x;
            d = Math.max(d, ratio > 0.0 ? Math.max(ratio, 1.0 / ratio) : Double.POSITIVE_INFINITY);
        }
        return d;
    }

    static double meanAbs(double[] r) {
        double s = 0.0;
        for (double x : r) {
            s += Math.abs(x);
        }
        return s / r.length;
    }

    /**
     * {@code cost(P, Q^) / cost(P, bestOf@P)} for the best-of-{@value #RESTARTS} Lloyd solution Q^ on the coreset;
     * NaN for an empty coreset.
     */
    static double downstream(Instance inst, PointSet coreset, long seed) {
        if (coreset.isEmpty() || !(coreset.totalWeight() > 0.0)) {
            return Double.NaN;
        }
        KMeansResult q = WeightedLloyd.bestOf(coreset, inst.k, LLOYD_ITERATIONS, RESTARTS, new SplittableRandom(seed),
                MetricsSink.NOOP);
        return fullCost(inst, Centers.of(q.centers())) / inst.referenceCost();
    }

    static double fullCost(Instance inst, Centers c) {
        Problem problem = Problem.kMeans();
        double[] scratch = new double[c.size()];
        PointSet pts = inst.points;
        double s = 0.0;
        for (int t = 0, g = 0; t < pts.segmentCount(); t++) {
            PointBlock b = pts.segment(t);
            for (int i = 0, rows = b.size(); i < rows; i++, g++) {
                double w = b.weight(i);
                if (w > 0.0) {
                    s += w * problem.pointCost(b.vector(i), c, scratch);
                }
            }
        }
        return s;
    }

    /** Serialized protobuf size of a coreset message, as the gRPC transport sends it. */
    static long messageBytes(Coreset c) {
        return ProtoCodec.toProto(c).getSerializedSize();
    }

    /** Payload bytes of one point in the packet model: (index, value) per entry, weight, item id. */
    static long pointBytes(int nnz) {
        return 12L * nnz + 12L;
    }
}
