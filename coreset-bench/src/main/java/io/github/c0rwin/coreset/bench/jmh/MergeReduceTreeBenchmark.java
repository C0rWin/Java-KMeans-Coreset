package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.util.List;
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
import org.openjdk.jmh.annotations.Warmup;

/**
 * Amortised {@link MergeReduceTree#add} of m-point batch coresets: on average one merge, i.e. one compress of 2m
 * points, per add. The tree is rebuilt every iteration. {@code metrics=recording} measures the instrumentation cost
 * against {@link MetricsSink#NOOP}.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class MergeReduceTreeBenchmark {

    private static final int POOL = 256;

    @Param({"uniform", "sensitivity"})
    public String algorithm;

    @Param({"200", "1000"})
    public int m;

    @Param({"noop", "recording"})
    public String metrics;

    private Coreset[] batches;
    private MergeReduceTree tree;
    private int next;

    @Setup(Level.Trial)
    public void data() {
        List<WeightedPoint> points = Workloads.gaussianMixture(POOL * m, CompressBenchmark.D, CompressBenchmark.K, 9L);
        batches = new Coreset[POOL];
        for (int i = 0; i < POOL; i++) {
            batches[i] = Coreset.of(points.subList(i * m, (i + 1) * m));
        }
    }

    @Setup(Level.Iteration)
    public void freshTree() {
        MetricsSink sink = metrics.equals("recording") ? new RecordingMetrics() : MetricsSink.NOOP;
        tree = new MergeReduceTree(CompressBenchmark.create(algorithm, m), new SplittableRandom(10), sink);
        next = 0;
    }

    @Benchmark
    public MergeReduceTree add() {
        tree.add(batches[next]);
        next = (next + 1) & (POOL - 1);
        return tree;
    }
}
