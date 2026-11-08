package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.MessageCodec;
import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.grpc.resilience.Decoder;
import io.github.c0rwin.coreset.grpc.resilience.MessageDecoder;
import io.github.c0rwin.coreset.grpc.resilience.ReceivedMessage;
import io.github.c0rwin.coreset.node.CoresetNode;
import io.github.c0rwin.coreset.node.MessageSpec;
import io.github.c0rwin.coreset.node.NodeMessage;
import io.github.c0rwin.coreset.node.Packetization;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.TearDown;
import org.openjdk.jmh.annotations.Warmup;

/**
 * CPU cost of the loss-resilient message path on one node's summary of {@link #SUMMARY} dense points in d = 16:
 * building the message per packetization (snapshot union included), encoding its packets, and decoding a delivered
 * subset. Wire bytes per form are printed once per trial.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 1, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class ResilienceBenchmark {

    public static final int SUMMARY = 4000;
    public static final int DIM = 16;
    public static final int RECORDS_PER_PACKET = 16;

    @Param({"summary", "merged", "per-draw", "mean-cf"})
    public String form;

    @Param({"1000"})
    public int size;

    private CoresetNode node;
    private NodeMessage message;
    private MessageHeader header;
    private List<Packet> delivered;
    private long stream;

    @Setup(Level.Trial)
    public void setup() throws InterruptedException {
        Packetization p = Packetization.fromId(form);
        MessageSpec spec = p == Packetization.SUMMARY ? MessageSpec.SUMMARY
                : new MessageSpec(p, size, p.sampled() ? 10 : 0, RECORDS_PER_PACKET, p.cells() ? 10 : 0);
        node = new CoresetNode(CoresetAlgorithms.load(), MetricsSink.NOOP, Duration.ofSeconds(5));
        // uniform with m above the batch size keeps every point: the summary is the raw input
        node.createSession("b", new SessionSpec("uniform", Map.of("m", Integer.toString(SUMMARY)), 1L, 1, 16,
                TreeMode.SINGLE_WRITER, spec));
        SplittableRandom rnd = new SplittableRandom(7);
        List<WeightedPoint> points = new ArrayList<>(SUMMARY);
        for (int i = 0; i < SUMMARY; i++) {
            double[] v = new double[DIM];
            int c = rnd.nextInt(10);
            for (int j = 0; j < DIM; j++) {
                v[j] = c * 10 + rnd.nextGaussian();
            }
            points.add(new WeightedPoint(SparseVector.dense(v), 1.0));
        }
        node.ingest("b", 0, points);
        node.awaitApplied("b", Duration.ofSeconds(30));
        message = node.message("b", 0);
        header = MessageCodec.header(message, 0, false).build();
        List<Packet> all = new ArrayList<>(message.packets());
        long bytes = header.getSerializedSize();
        for (int j = 0; j < message.packets(); j++) {
            Packet packet = MessageCodec.packet(message, j);
            all.add(packet);
            bytes += packet.getSerializedSize();
        }
        delivered = new ArrayList<>();
        SplittableRandom loss = new SplittableRandom(3);
        for (Packet packet : all) {
            if (loss.nextDouble() >= 0.3) {
                delivered.add(packet);
            }
        }
        long unary = ProtoCodec.toProto(node.snapshot("b").coreset()).getSerializedSize();
        System.out.printf("%n# form=%s records=%d packets=%d message_bytes=%d unary_snapshot_bytes=%d%n", form,
                message.recordCount(), message.packets(), bytes, unary);
    }

    @TearDown(Level.Trial)
    public void tearDown() {
        node.close();
    }

    /** A fresh message per call (new sampling stream): snapshot union plus the packetization's work. */
    @Benchmark
    public NodeMessage build() {
        return node.message("b", ++stream);
    }

    @Benchmark
    public long encodePackets() {
        long bytes = 0;
        for (int j = 0; j < message.packets(); j++) {
            bytes += MessageCodec.packet(message, j).getSerializedSize();
        }
        return bytes;
    }

    /** Realized-count (sampled forms) or realized-weight decode of the 70% of packets a Bernoulli(0.3) delivered. */
    @Benchmark
    public MessageDecoder.Decoded decode() {
        Decoder d = Packetization.fromId(form).sampled() ? Decoder.REALIZED_COUNT : Decoder.REALIZED_WEIGHT;
        return MessageDecoder.decode(List.of(new ReceivedMessage(0, header, delivered)), d, 0.3, 1.0, 1 << 20);
    }
}
