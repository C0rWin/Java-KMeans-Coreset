package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.Map;
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
 * One {@link CoresetAlgorithm#compress} of {@code n} points (k = 10 Gaussian clusters, d = 16) down to {@code m}.
 * {@code mean} runs 3 Lloyd iterations instead of the default 20 to keep the grid affordable; its cost is linear in
 * the iteration count.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class CompressBenchmark {

    static final int K = 10;
    static final int D = 16;

    @Param({"uniform", "sensitivity", "mean"})
    public String algorithm;

    @Param({"10000", "100000"})
    public int n;

    @Param({"200", "1000"})
    public int m;

    private Coreset input;
    private CoresetAlgorithm algo;
    private SplittableRandom rng;

    @Setup
    public void setup() {
        input = Coreset.of(Workloads.gaussianMixture(n, D, K, 5L));
        algo = create(algorithm, m);
        rng = new SplittableRandom(6);
    }

    static CoresetAlgorithm create(String id, int m) {
        Map<String, String> params = switch (id) {
            case "uniform" -> Map.of("m", Integer.toString(m));
            case "sensitivity" -> Map.of("k", Integer.toString(K), "m", Integer.toString(m));
            case "mean" -> Map.of("m", Integer.toString(m), "lloydIterations", "3");
            default -> throw new IllegalArgumentException(id);
        };
        return CoresetAlgorithms.load().create(id, params);
    }

    @Benchmark
    public Coreset compress() {
        return algo.compress(input, rng, MetricsSink.NOOP);
    }
}
