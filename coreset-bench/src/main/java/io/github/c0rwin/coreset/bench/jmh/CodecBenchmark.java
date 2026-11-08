package io.github.c0rwin.coreset.bench.jmh;

import com.google.protobuf.InvalidProtocolBufferException;
import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Protobuf encode and decode of one ingest batch ({@link ProtoCodec} + {@link Batch}, packed O(nnz) arrays), from and
 * to the {@link WeightedPoint}s the node computes on. Serialized sizes per batch are printed by
 * {@code GrpcEndToEndHarness}.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class CodecBenchmark {

    public static final int BATCH = 1000;
    public static final int DIM = 1 << 17;

    @Param({"16", "128"})
    public int nnz;

    private List<WeightedPoint> points;
    private byte[] protoBytes;
    private long seq;

    @Setup
    public void setup() {
        points = batch(nnz, 13L);
        protoBytes = protoEncode();
    }

    public static List<WeightedPoint> batch(int nnz, long seed) {
        SplittableRandom rng = new SplittableRandom(seed);
        List<WeightedPoint> out = new ArrayList<>(BATCH);
        for (int i = 0; i < BATCH; i++) {
            out.add(new WeightedPoint(Workloads.sparse(DIM, nnz, rng), 1.0));
        }
        return out;
    }

    @Benchmark
    public byte[] protoEncode() {
        return Batch.newBuilder()
                .setSessionId("bench")
                .setSeq(seq++)
                .addAllPoints(ProtoCodec.pointsToProto(points))
                .build()
                .toByteArray();
    }

    @Benchmark
    public List<WeightedPoint> protoDecode() throws InvalidProtocolBufferException {
        return ProtoCodec.pointsFromProto(Batch.parseFrom(protoBytes).getPointsList(),
                io.github.c0rwin.coreset.node.NodeLimits.defaults().maxDim());
    }
}
