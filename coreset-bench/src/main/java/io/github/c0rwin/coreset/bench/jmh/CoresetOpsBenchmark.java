package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import io.github.c0rwin.coreset.grpc.ProtoCodec;
import io.github.c0rwin.coreset.spark.CoresetCodec;
import java.util.ArrayList;
import java.util.List;
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
 * Coreset plumbing that runs on every merge, snapshot and gather: two-way union (merge-and-reduce), a left fold of
 * {@value #PARTS} unions (how a coordinator gathers node snapshots), and decoding a protobuf coreset.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class CoresetOpsBenchmark {

    static final int PARTS = 8;

    @Param({"1000"})
    public int m;

    private Coreset a;
    private Coreset b;
    private Coreset[] parts;
    private io.github.c0rwin.coreset.grpc.proto.Coreset proto;
    private byte[] sparkBytes;
    private List<WeightedPoint> list;
    private List<SparseVector> centers;
    private MergeReduceTree tree;

    @Setup
    public void setup() {
        List<WeightedPoint> pts = Workloads.gaussianMixture((PARTS + 2) * m, CompressBenchmark.D, CompressBenchmark.K,
                17L);
        a = new Coreset(pts.subList(0, m), 1.0);
        b = new Coreset(pts.subList(m, 2 * m), 2.0);
        parts = new Coreset[PARTS];
        for (int i = 0; i < PARTS; i++) {
            parts[i] = new Coreset(pts.subList((i + 2) * m, (i + 3) * m), 0.5);
        }
        proto = ProtoCodec.toProto(a);
        sparkBytes = CoresetCodec.encode(a);
        list = List.copyOf(pts.subList(0, m));
        centers = new ArrayList<>();
        for (int i = 0; i < CompressBenchmark.K; i++) {
            centers.add(pts.get(pts.size() - 1 - i).vector());
        }
        tree = MergeReduceTree.restore(new UniformCoreset(m), MergeRandomness.derived(1L, 0L), MetricsSink.NOOP,
                List.of(parts), (1L << PARTS) - 1);
    }

    @Benchmark
    public Coreset union2() {
        return Coreset.union(a, b);
    }

    @Benchmark
    public Coreset unionFold8() {
        Coreset acc = Coreset.of(List.of());
        for (Coreset p : parts) {
            acc = Coreset.union(acc, p);
        }
        return acc;
    }

    @Benchmark
    public Coreset protoDecodeCoreset() {
        return ProtoCodec.fromProto(proto, io.github.c0rwin.coreset.node.NodeLimits.defaults().maxDim());
    }

    @Benchmark
    public Coreset sparkDecodeCoreset() {
        return CoresetCodec.decode(sparkBytes);
    }

    /** Builds a coreset from a list of points (validation plus the internal copy). */
    @Benchmark
    public Coreset coresetOf() {
        return Coreset.of(list);
    }

    /** Union of {@value #PARTS} occupied levels. */
    @Benchmark
    public Coreset treeSnapshot() {
        return tree.snapshot();
    }

    /** Cost of {@code m} dense points (d = 16) against k = 10 centers. */
    @Benchmark
    public double cost() {
        return Cost.cost(a, centers);
    }
}
