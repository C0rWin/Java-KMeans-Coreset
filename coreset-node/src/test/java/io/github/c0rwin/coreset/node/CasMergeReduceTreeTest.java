package io.github.c0rwin.coreset.node;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.node.testing.TestData;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class CasMergeReduceTreeTest {

    @Test
    void sequentialInsertsFormABinaryCounter() {
        CasMergeReduceTree tree = new CasMergeReduceTree(TestAlgorithms.IDENTITY, MetricsSink.NOOP);
        SplittableRandom rng = new SplittableRandom(1);
        for (int i = 1; i <= 11; i++) {
            tree.add(new Summary(Coreset.of(TestData.batch(rng, 1, 4)), 1, 1.0), rng);
            CasMergeReduceTree.Read read = tree.read(0L, () -> false);
            assertTrue(read.consistent());
            assertEquals(Integer.bitCount(i), read.summaries().size());
            assertEquals(32 - Integer.numberOfLeadingZeros(i), tree.height());
        }
    }

    @Test
    void concurrentWritersLoseNothing() throws Exception {
        CasMergeReduceTree tree = new CasMergeReduceTree(TestAlgorithms.MERGE_EQUAL, MetricsSink.NOOP);
        int threads = 8;
        int perThread = 2_000;
        CyclicBarrier start = new CyclicBarrier(threads);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<Double>> weights = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int seed = t;
                weights.add(pool.submit(() -> {
                    SplittableRandom rng = new SplittableRandom(seed);
                    double w = 0;
                    start.await();
                    for (int i = 0; i < perThread; i++) {
                        List<WeightedPoint> b = TestData.batch(rng, 2, 4);
                        double bw = TestData.weight(b);
                        w += bw;
                        tree.add(new Summary(Coreset.of(b), 1, bw), rng);
                    }
                    return w;
                }));
            }
            double expected = 0;
            for (Future<Double> f : weights) {
                expected += f.get(60, TimeUnit.SECONDS);
            }
            CasMergeReduceTree.Read read = tree.read(0L, () -> false);
            assertTrue(read.consistent());
            assertEquals(0, tree.inFlightBatches());
            long batches = read.summaries().stream().mapToLong(Summary::batches).sum();
            double inputWeight = read.summaries().stream().mapToDouble(Summary::inputWeight).sum();
            double coresetWeight = read.summaries().stream().mapToDouble(s -> s.coreset().totalWeight()).sum();
            assertEquals((long) threads * perThread, batches);
            assertEquals(expected, inputWeight);
            assertEquals(expected, coresetWeight);
            assertEquals(Long.bitCount(batches), read.summaries().size());
        } finally {
            pool.shutdownNow();
        }
    }
}
