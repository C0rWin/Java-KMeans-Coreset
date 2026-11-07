package io.github.c0rwin.coreset.bench.jmh;

import com.google.protobuf.InvalidProtocolBufferException;
import io.github.c0rwin.coreset.bench.LegacyConvert;
import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.apache.thrift.TDeserializer;
import org.apache.thrift.TException;
import org.apache.thrift.TSerializer;
import org.apache.thrift.protocol.TBinaryProtocol;
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
import univ.ml.distributed.coreset.CoresetPoints;
import univ.ml.distributed.coreset.CoresetUtils;
import univ.ml.sparse.SparseWeightableVector;

/**
 * Encode and decode of one ingest batch, from and to the domain types each stack computes on: protobuf
 * ({@link ProtoCodec} + {@link Batch}, packed O(nnz) arrays) vs legacy Thrift ({@code TBinaryProtocol}, a
 * {@code map<i32,double>} per point, decoded into commons-math {@link SparseWeightableVector}s as the legacy server
 * does). Sizes per batch are printed by {@code CodecSizes}.
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
    private byte[] thriftBytes;
    private TSerializer serializer;
    private TDeserializer deserializer;
    private long seq;

    @Setup
    public void setup() throws TException {
        points = batch(nnz, 13L);
        serializer = new TSerializer(new TBinaryProtocol.Factory());
        deserializer = new TDeserializer(new TBinaryProtocol.Factory());
        protoBytes = protoEncode();
        thriftBytes = thriftEncode();
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
        return ProtoCodec.pointsFromProto(Batch.parseFrom(protoBytes).getPointsList());
    }

    @Benchmark
    public byte[] thriftEncode() throws TException {
        return serializer.serialize(new CoresetPoints(seq++, LegacyConvert.toThrift(points)));
    }

    @Benchmark
    public List<SparseWeightableVector> thriftDecode() throws TException {
        CoresetPoints msg = new CoresetPoints();
        deserializer.deserialize(msg, thriftBytes);
        return CoresetUtils.toSparseWeightableVectors(msg);
    }
}
