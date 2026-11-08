package io.github.c0rwin.coreset.core.algo;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;

/** One instance of each algorithm shared by 8 threads, each with its own seeded generator, must match a serial run. */
class AlgorithmConcurrencyTest {

    private static final int THREADS = 8;
    private static final int TASKS = 32;

    @Test
    void sharedInstancesGiveSerialResults() {
        Coreset input = TestData.mixture(1500, 4, 3, true, 1);
        List<CoresetAlgorithm> algorithms = List.of(
                new UniformCoreset(100),
                new SensitivityCoreset(4, 100, Seeding.kMeansPlusPlus(2)),
                new SensitivityCoreset(4, 100, Seeding.bicriteria()),
                new MeanCoreset(30, 5),
                MeanCoreset.adaptive(4, 0.3, 64, 5));
        RecordingMetrics metrics = new RecordingMetrics();
        for (CoresetAlgorithm algo : algorithms) {
            List<Coreset> serial = new ArrayList<>(TASKS);
            for (int t = 0; t < TASKS; t++) {
                serial.add(algo.compress(input, new SplittableRandom(100 + t)));
            }
            List<Coreset> parallel = assertTimeoutPreemptively(Duration.ofMinutes(2), () -> {
                ExecutorService pool = Executors.newFixedThreadPool(THREADS);
                try {
                    List<Callable<Coreset>> calls = new ArrayList<>(TASKS);
                    for (int t = 0; t < TASKS; t++) {
                        long seed = 100 + t;
                        calls.add(() -> algo.compress(input, new SplittableRandom(seed), metrics));
                    }
                    List<Coreset> out = new ArrayList<>(TASKS);
                    for (Future<Coreset> f : pool.invokeAll(calls)) {
                        out.add(f.get());
                    }
                    return out;
                } finally {
                    pool.shutdownNow();
                }
            });
            assertEquals(serial, parallel, algo.toString());
        }
    }
}
