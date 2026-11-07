package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.bench.Workloads;
import io.github.c0rwin.coreset.core.SparseVector;
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

/**
 * Squared distance kernels: sparse-sparse (support merge) and sparse-dense (O(nnz) gather against a dense centroid,
 * as in Lloyd's assignment step). Each invocation walks {@value #PAIRS} distinct pairs so branch predictors and
 * caches cannot memorise a single pair; scores are per distance.
 */
@State(Scope.Benchmark)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class DistanceBenchmark {

    static final int PAIRS = 256;
    private static final int CENTERS = 8;

    @Param({"16", "128", "1024"})
    public int nnz;

    @Param({"4096", "1048576"})
    public int dim;

    private SparseVector[] a;
    private SparseVector[] b;
    private double[][] dense;
    private double[] denseNorms;

    @Setup
    public void setup() {
        SplittableRandom rng = new SplittableRandom(42);
        a = new SparseVector[PAIRS];
        b = new SparseVector[PAIRS];
        for (int i = 0; i < PAIRS; i++) {
            a[i] = Workloads.sparse(dim, nnz, rng);
            b[i] = Workloads.sparse(dim, nnz, rng);
        }
        // A few dense centroids, like Lloyd's k centers; at dim 2^20 each is 8 MiB, so gathers miss cache.
        dense = new double[CENTERS][];
        denseNorms = new double[CENTERS];
        for (int j = 0; j < CENTERS; j++) {
            double[] c = new double[dim];
            for (int t = 0; t < dim; t++) {
                c[t] = rng.nextDouble();
            }
            dense[j] = c;
            double s = 0.0;
            for (double v : c) {
                s += v * v;
            }
            denseNorms[j] = s;
        }
    }

    @Benchmark
    @OperationsPerInvocation(PAIRS)
    public void sparseSparse(Blackhole bh) {
        for (int i = 0; i < PAIRS; i++) {
            bh.consume(a[i].squaredDistance(b[i]));
        }
    }

    @Benchmark
    @OperationsPerInvocation(PAIRS)
    public void sparseDense(Blackhole bh) {
        for (int i = 0; i < PAIRS; i++) {
            int j = i & (CENTERS - 1);
            bh.consume(a[i].squaredDistance(dense[j], denseNorms[j]));
        }
    }
}
