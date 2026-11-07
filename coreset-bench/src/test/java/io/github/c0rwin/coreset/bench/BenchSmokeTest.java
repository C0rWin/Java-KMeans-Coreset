package io.github.c0rwin.coreset.bench;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.bench.jmh.DistanceBenchmark;
import io.github.c0rwin.coreset.bench.perf.NodeIngestHarness;
import io.github.c0rwin.coreset.bench.perf.ParityHarness;
import io.github.c0rwin.coreset.core.SparseVector;
import java.util.Collection;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.openjdk.jmh.results.RunResult;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

/** Build-time smoke runs: one tiny JMH benchmark in process, and both harnesses on toy sizes (they self-check). */
class BenchSmokeTest {

    @Test
    void jmhBenchmarkRuns() throws Exception {
        Options opt = new OptionsBuilder()
                .include(DistanceBenchmark.class.getName() + ".sparseSparse")
                .param("nnz", "16")
                .param("dim", "4096")
                .forks(0)
                .warmupIterations(0)
                .measurementIterations(1)
                .measurementTime(TimeValue.milliseconds(200))
                .timeUnit(TimeUnit.NANOSECONDS)
                .shouldFailOnError(true)
                .build();
        Collection<RunResult> results = new Runner(opt).run();
        assertEquals(1, results.size());
        double score = results.iterator().next().getPrimaryResult().getScore();
        assertTrue(score > 0.0 && Double.isFinite(score), "score " + score);
    }

    @Test
    void sparseGeneratorHasExactDistinctSupport() {
        SplittableRandom rng = new SplittableRandom(1);
        for (int i = 0; i < 100; i++) {
            SparseVector v = Workloads.sparse(64, 40, rng);
            assertEquals(40, v.nnz());
        }
    }

    @Test
    void nodeIngestHarnessRunsOnToySizes() throws Exception {
        NodeIngestHarness.main(new String[] {"--points", "20000", "--batch", "500", "--pool-batches", "8",
                "--workers", "1,2", "--grpc-nodes", "1,2", "--repeats", "1"});
    }

    @Test
    void parityHarnessRunsOnToySizes() throws Exception {
        ParityHarness.main(new String[] {"--points", "4000", "--batch", "500", "--repeats", "1", "--candidates", "1",
                "--grpc-workers", "1"});
    }
}
