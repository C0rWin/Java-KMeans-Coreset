package io.github.c0rwin.coreset.grpc.resilience;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.MessageCodec;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.IngestResult;
import io.github.c0rwin.coreset.node.MessageSpec;
import io.github.c0rwin.coreset.node.NodeMessage;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.github.c0rwin.coreset.node.testing.TestProvider;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/** Messages of a node whose summary is its raw input, and an erasure channel applied to their packets. */
final class TestMessages implements AutoCloseable {

    static final int MAX_DIM = 1 << 20;

    final CoresetNode node;
    final List<WeightedPoint> points;

    TestMessages(List<WeightedPoint> points, MessageSpec spec) throws InterruptedException {
        this.points = points;
        node = new CoresetNode(TestProvider.registry(), MetricsSink.NOOP, Duration.ofMillis(200));
        node.createSession("s", new SessionSpec("identity", Map.of(), 5L, 1, 16, TreeMode.SINGLE_WRITER, spec));
        for (int from = 0, seq = 0; from < points.size(); from += 500, seq++) {
            IngestResult r = node.ingest("s", seq, points.subList(from, Math.min(points.size(), from + 500)));
            if (r.status() != IngestResult.Status.ACCEPTED) {
                throw new IllegalStateException(r.toString());
            }
        }
        if (!node.awaitApplied("s", Duration.ofSeconds(30))) {
            throw new IllegalStateException("not applied");
        }
    }

    NodeMessage message(long stream) {
        return node.message("s", stream);
    }

    /** The message of {@code stream} after {@code channel} erased packets (its header is reliable). */
    ReceivedMessage received(long stream, ErasureChannel channel) {
        NodeMessage m = message(stream);
        List<Packet> delivered = new ArrayList<>();
        for (int p = 0; p < m.packets(); p++) {
            if (channel.deliver()) {
                delivered.add(MessageCodec.packet(m, p));
            }
        }
        return new ReceivedMessage(0, MessageCodec.header(m, 0, false).build(), delivered);
    }

    /** Unbalanced 2-d mixture with unit weights. */
    static List<WeightedPoint> mixture(int n, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        List<WeightedPoint> out = new ArrayList<>(n);
        double[][] centers = {{0, 0}, {12, 3}, {-6, 15}, {25, -10}};
        for (int i = 0; i < n; i++) {
            int c = rnd.nextInt(10) < 6 ? 0 : 1 + rnd.nextInt(3);
            out.add(new WeightedPoint(SparseVector.dense(new double[] {centers[c][0] + rnd.nextGaussian() * (1 + c),
                    centers[c][1] + rnd.nextGaussian()}), 1.0));
        }
        return out;
    }

    @Override
    public void close() {
        node.close();
    }
}
