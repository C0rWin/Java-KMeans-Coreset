package univ.ml;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import org.apache.commons.math3.util.MathArrays;
import org.junit.Before;
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

import univ.ml.sparse.SparseWeightableVector;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 5, time = 1, timeUnit = TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Fork(1)
public class VectorComputationTest {

    private List<SparseWeightableVector> sparseVectors;

    private List<SparseWeightableVector> sparseCenters;

    private double[][] vectors;

    private double[][] centers;

    @Setup
    @Before
    public void setup() {
        final int N = 1_000;
        final int D = 10;
        final int C = 10;

        vectors = new double[N][];
        sparseVectors = new ArrayList<>();

        centers = new double[C][];
        sparseCenters = new ArrayList<>();

        fillVectors(sparseVectors, vectors, N, D);
        fillVectors(sparseCenters, centers, C, D);
    }

    public static void fillVectors(final List<SparseWeightableVector> sparse, final double[][] vectors, int cnt, int dim) {
        final Random rng = new Random(System.currentTimeMillis());
        for (int i = 0; i < cnt; ++i) {
            vectors[i] = new double[dim];
            final Map<Integer, Double> coords = new HashMap<>();
            for (int j = 0; j < dim; j++) {
                vectors[i][j] = rng.nextDouble();
                coords.put(j, rng.nextDouble());
            }
            sparse.add(new SparseWeightableVector(coords, 1d, dim));
        }
    }

    @Benchmark
    public void vectorAsArray(final Blackhole bh) {
        double result = 0d;

        for (double[] vector : vectors ) {
            double minDist = Double.MAX_VALUE;
            for (double[] center : centers) {
                final double dist = MathArrays.distance(vector, center);
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            result += minDist;
        }

        bh.consume(result);
    }

    @Benchmark
    public void sparseVectors(final Blackhole bh) {
        double result = 0d;

        for (final SparseWeightableVector vector : sparseVectors) {
            double minDist = Double.MAX_VALUE;
            for (final SparseWeightableVector center : sparseCenters) {
                final double dist = vector.getDistance(center.getVector());
                if (dist < minDist) {
                    minDist = dist;
                }
            }
            result += minDist;
        }

        bh.consume(result);
    }

    @Test
    public void benchmark() throws RunnerException {
        final Options opt = new OptionsBuilder()
                .include(this.getClass().getName() + ".*")
                .build();

        new Runner(opt).run();
    }

    @Test
    public void sparseVectors() {
        for (final SparseWeightableVector vector : sparseVectors) {
            for (final SparseWeightableVector center : sparseCenters) {
                vector.getDistance(center.getVector());
            }
        }

    }
}
