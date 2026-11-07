package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.core.sampling.AliasSampler;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OperationsPerInvocation;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Param;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/** Vose alias table construction (O(n)) and single draws (O(1)) over heavy-tailed sensitivity-like weights. */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class AliasSamplerBenchmark {

    private static final int DRAWS = 1024;

    @Param({"10000", "1000000"})
    public int n;

    private double[] weights;
    private AliasSampler sampler;
    private SplittableRandom rng;

    @Setup
    public void setup() {
        SplittableRandom init = new SplittableRandom(7);
        weights = new double[n];
        for (int i = 0; i < n; i++) {
            weights[i] = Math.pow(1.0 - init.nextDouble(), -1.5); // Pareto tail like real sensitivities
        }
        sampler = AliasSampler.of(weights);
        rng = new SplittableRandom(11);
    }

    @Benchmark
    public AliasSampler build() {
        return AliasSampler.of(weights);
    }

    @Benchmark
    @OperationsPerInvocation(DRAWS)
    public void draw(Blackhole bh) {
        for (int i = 0; i < DRAWS; i++) {
            bh.consume(sampler.next(rng));
        }
    }
}
