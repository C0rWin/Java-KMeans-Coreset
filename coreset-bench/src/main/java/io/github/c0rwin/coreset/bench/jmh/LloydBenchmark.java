package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
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
 * Twenty Lloyd iterations (stopping early on convergence) with k centers from fixed plain k-means++ seeds over a dense
 * Gaussian mixture of {@value #CLUSTERS} clusters ({@code n} points, {@code d} stored coordinates each) read from one
 * block: the dense center layout's assignment and update steps. With fewer clusters than centers, several centers
 * share a cluster and keep moving, so all twenty iterations run (k matching well-separated clusters converges at once).
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Fork(value = 1, jvmArgsAppend = {"-Xms4g", "-Xmx4g"})
@Warmup(iterations = 2, time = 2)
@Measurement(iterations = 4, time = 2)
public class LloydBenchmark {

    static final int CLUSTERS = 5;

    @Param({"200000"})
    public int n;

    @Param({"10"})
    public int d;

    @Param({"10", "50"})
    public int k;

    private PointSet points;
    private List<SparseVector> seeds;

    @Setup
    public void setup() {
        points = PointSet.of(PointBlock.of(Workloads.gaussianMixture(n, d, CLUSTERS, 21L)));
        seeds = KMeansPlusPlus.seed(points, k, 1, new SplittableRandom(22), MetricsSink.NOOP);
    }

    @Benchmark
    public KMeansResult lloyd20() {
        return WeightedLloyd.refine(points, seeds, 20, MetricsSink.NOOP);
    }
}
