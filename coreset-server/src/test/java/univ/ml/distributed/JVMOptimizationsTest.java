package univ.ml.distributed;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.util.FastMath;
import org.junit.Test;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(1)
public class JVMOptimizationsTest {

    private List<Integer> list = new ArrayList<>();

    @Test
    public void executeBecnhmarkTest() throws RunnerException {
        final Options opt = new OptionsBuilder()
                .include(this.getClass().getName() + ".*")
                .addProfiler("gc")
                .addProfiler("cl")
                .build();

        new Runner(opt).run();
    }

    @Setup
    public void initializeBanchmark() {
        final Random rnd = new Random(System.currentTimeMillis());
        for (int i = 0; i < 10_000; i++) {
            list.add(rnd.nextInt(10_000));
        }
    }

    @Benchmark
    public void jdkMathPow(Blackhole bh) {
        for (Integer x : list) {
            bh.consume(Math.pow(x, 10));
        }
    }

    @Benchmark
    public void variableMult(Blackhole bh) {
        for (Integer x : list) {
            Integer res = 1;
            for (int i = 0; i < 10; i++) {
                res *= x;
            }
            bh.consume(res);
        }
    }

    @Benchmark
    public void fastMathPow(Blackhole bh) {
        for (Integer x : list) {
            bh.consume(FastMath.pow(x, 10));
        }
    }
}
