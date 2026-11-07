package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansPlusPlus;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.kmeans.WeightedLloyd;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
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
 * Weighted k-means++ seeding ({@code n * (k - 1)} sparse-sparse distances) and one Lloyd iteration from fixed seeds
 * ({@code refine(.., 1)}: initial assignment, one centroid update, re-assignment, i.e. {@code 2 n k} sparse-dense
 * distances) over a Gaussian mixture stored as sparse vectors with {@code nnz = d}.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms4g", "-Xmx4g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class KMeansBenchmark {

    @Param({"20000", "100000"})
    public int n;

    @Param({"10", "50"})
    public int k;

    @Param({"16", "128"})
    public int d;

    private List<WeightedPoint> points;
    private List<SparseVector> seeds;
    private SplittableRandom rng;

    @Setup
    public void setup() {
        points = Workloads.gaussianMixture(n, d, k, 1L);
        seeds = KMeansPlusPlus.seed(points, k, new SplittableRandom(2));
        rng = new SplittableRandom(3);
    }

    @Benchmark
    public List<SparseVector> seed() {
        return KMeansPlusPlus.seed(points, k, rng);
    }

    @Benchmark
    public KMeansResult lloydIteration() {
        return WeightedLloyd.refine(points, seeds, 1, MetricsSink.NOOP);
    }
}
