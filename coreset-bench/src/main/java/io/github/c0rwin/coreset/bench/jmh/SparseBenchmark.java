package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.algo.Clustering;
import io.github.c0rwin.coreset.core.algo.MeanCoreset;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
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
 * Seeding, Lloyd, cost and the three compressions on high-dimensional sparse data ({@link Workloads#sparseMixture}:
 * {@code n} points, {@code dim = 10^6}, about 10 non-zeros, k clusters), where centers are far denser than points and
 * Lloyd uses its sparse center layout.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"-Xms4g", "-Xmx4g"})
@Warmup(iterations = 2, time = 2)
@Measurement(iterations = 4, time = 2)
public class SparseBenchmark {

    static final int DIM = 1_000_000;
    static final int NNZ = 10;

    @Param({"100000"})
    public int n;

    @Param({"10"})
    public int k;

    private Coreset input;
    private List<SparseVector> seeds;
    private SensitivityCoreset sensitivity;
    private UniformCoreset uniform;
    private MeanCoreset mean;
    private SplittableRandom rng;

    @Setup
    public void setup() {
        input = Coreset.of(Workloads.sparseMixture(n, DIM, NNZ, k, 11L));
        seeds = KMeansPlusPlus.seed(input.pointSet(), k, 1, new SplittableRandom(12), MetricsSink.NOOP);
        sensitivity = new SensitivityCoreset(k, 1000, Seeding.kMeansPlusPlus(0));
        uniform = new UniformCoreset(1000);
        mean = new MeanCoreset(100, 3);
        rng = new SplittableRandom(13);
    }

    /** Default (greedy) seeding. */
    @Benchmark
    public List<SparseVector> kmeansppSeed() {
        return KMeansPlusPlus.seed(input.pointSet(), k, rng, MetricsSink.NOOP);
    }

    /** Plain D^2 seeding (one trial per step). */
    @Benchmark
    public List<SparseVector> kmeansppSeedPlain() {
        return KMeansPlusPlus.seed(input.pointSet(), k, 1, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Clustering bicriteria() {
        return Seeding.bicriteria().cluster(input.pointSet(), k, rng, MetricsSink.NOOP);
    }

    /** Twenty Lloyd iterations from fixed seeds (stops early on convergence, deterministic). */
    @Benchmark
    public KMeansResult lloyd20() {
        return WeightedLloyd.refine(input.pointSet(), seeds, 20, MetricsSink.NOOP);
    }

    @Benchmark
    public double cost() {
        return Cost.cost(input, seeds);
    }

    @Benchmark
    public Coreset sensitivityCompress() {
        return sensitivity.compress(input, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Coreset uniformCompress() {
        return uniform.compress(input, rng, MetricsSink.NOOP);
    }

    @Benchmark
    public Coreset meanCompress() {
        return mean.compress(input, rng, MetricsSink.NOOP);
    }
}
