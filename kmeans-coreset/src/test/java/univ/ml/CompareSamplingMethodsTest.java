package univ.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.junit.Test;
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
import org.openjdk.jmh.infra.Blackhole;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import univ.ml.sparse.BinarySearchRandomSample;
import univ.ml.sparse.CTRandomSample;
import univ.ml.sparse.RandomSampleAlgorithm;
import univ.ml.sparse.SparseRandomSample;
import univ.ml.sparse.SparseWeightableVector;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(1)
public class CompareSamplingMethodsTest {

    private List<SparseWeightableVector> dataset;

    @Param("10000")
    public int N;

    @Param({"10"})
    public int D;

    @Param({"100", "1000", "5000"})
    public int sampleSize;

    @Setup
    public void setup() {
        dataset = new ArrayList<>();
        final Random rng = new Random(System.currentTimeMillis());
        for (int i = 0; i < N; i++) {
            final Map<Integer, Double> coords = new HashMap<>();
            for (int j = 0; j < D; j++) {
                coords.put(j, rng.nextDouble());
            }
            dataset.add(new SparseWeightableVector(coords, rng.nextDouble(), D));
        }
    }

    @Test
    public void benchmark() throws RunnerException {
        final Options opt = new OptionsBuilder()
                .include(this.getClass().getName() + ".*")
                .build();

        new Runner(opt).run();
    }

    @Benchmark
    public void randomSample(Blackhole bh) {
        final RandomSampleAlgorithm randomSample = new SparseRandomSample();
        // Sample points and consume by black hole object
        bh.consume(randomSample.getSampleOfSize(dataset, sampleSize));
    }

    @Benchmark
    public void constantTimeSample(Blackhole bh) {
        final RandomSampleAlgorithm randomSample = new CTRandomSample();
        // Sample points and consume by black hole object
        bh.consume(randomSample.getSampleOfSize(dataset, sampleSize));
    }

    @Benchmark
    public void polylogTimeSample(Blackhole bh) {
        final RandomSampleAlgorithm randomSample = new BinarySearchRandomSample();
        // Sample points and consume by black hole object
        bh.consume(randomSample.getSampleOfSize(dataset, sampleSize));
    }

}
