package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.Clustering;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.validate.IntegrityCheck;
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
 * Sampling coresets, bicriteria seeding and the integrity check on {@code n} dense points (d = 16, k = 10 Gaussian
 * clusters) compressed to m = 1000. The {@code weighted} input has weights {@code 1 + i % 7}, as merged coresets do;
 * the plain input has unit weights. Use {@code -prof gc} for the per-point scratch (B/op divided by n).
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"-Xms4g", "-Xmx4g"})
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 4, time = 1)
public class SamplingBenchmark {

    static final int K = 10;
    static final int D = 16;
    static final int M = 1000;

    @Param({"100000", "1000000"})
    public int n;

    private Coreset input;
    private Coreset weighted;
    private Coreset sample;
    private UniformCoreset uniform;
    private SensitivityCoreset sensitivity;
    private SensitivityCoreset sensitivityBicriteria;
    private SplittableRandom rng;

    @Setup
    public void setup() {
        List<WeightedPoint> points = Workloads.gaussianMixture(n, D, K, 21L);
        input = Coreset.of(points);
        List<WeightedPoint> w = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            w.add(new WeightedPoint(points.get(i).vector(), 1.0 + i % 7));
        }
        weighted = Coreset.of(w);
        uniform = new UniformCoreset(M);
        sensitivity = new SensitivityCoreset(K, M, Seeding.kMeansPlusPlus(0));
        sensitivityBicriteria = new SensitivityCoreset(K, M, Seeding.bicriteria());
        rng = new SplittableRandom(22);
        sample = uniform.compress(weighted, new SplittableRandom(23));
    }

    @Benchmark
    public Coreset uniform() {
        return uniform.compress(input, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Coreset uniformWeighted() {
        return uniform.compress(weighted, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Coreset sensitivity() {
        return sensitivity.compress(input, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Coreset sensitivityWeighted() {
        return sensitivity.compress(weighted, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Coreset sensitivityBicriteria() {
        return sensitivityBicriteria.compress(input, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Clustering bicriteria() {
        return Seeding.bicriteria().cluster(input.pointSet(), K, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public IntegrityCheck.Report integrity() {
        return IntegrityCheck.check(weighted, sample);
    }
}
