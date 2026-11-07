package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.core.metrics.Span;
import java.lang.management.ManagementFactory;
import java.util.SplittableRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

/**
 * {@link RecordingMetrics} record paths under 1, 4 and 12 threads hitting one shared metric name, against a bare
 * shared {@link LongAdder} (the floor for any lock-free counter). {@code record} samples log-normal latencies so
 * histogram buckets vary as in production. {@code span} is one open/close of a {@link Span} (clock, CPU and
 * allocation probes plus three timings).
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class MetricsBenchmark {

    static final String NAME = "bench.metric";

    @State(Scope.Benchmark)
    public static class Shared {
        final RecordingMetrics metrics = new RecordingMetrics();
        final LongAdder adder = new LongAdder();
    }

    @State(Scope.Thread)
    public static class Samples {
        final double[] values = new double[1024];
        int next;

        @Setup
        public void setup() {
            SplittableRandom rng = new SplittableRandom(Thread.currentThread().getId());
            for (int i = 0; i < values.length; i++) {
                values[i] = Math.exp(10.0 + 1.5 * rng.nextGaussian()); // ~22 us median, heavy right tail
            }
        }

        double next() {
            double v = values[next];
            next = (next + 1) & (values.length - 1);
            return v;
        }
    }

    @Benchmark
    @Threads(1)
    public void record1(Shared s, Samples v) {
        s.metrics.record(NAME, v.next());
    }

    @Benchmark
    @Threads(4)
    public void record4(Shared s, Samples v) {
        s.metrics.record(NAME, v.next());
    }

    @Benchmark
    @Threads(12)
    public void record12(Shared s, Samples v) {
        s.metrics.record(NAME, v.next());
    }

    @Benchmark
    @Threads(1)
    public void count1(Shared s) {
        s.metrics.count(NAME, 1);
    }

    @Benchmark
    @Threads(4)
    public void count4(Shared s) {
        s.metrics.count(NAME, 1);
    }

    @Benchmark
    @Threads(12)
    public void count12(Shared s) {
        s.metrics.count(NAME, 1);
    }

    @Benchmark
    @Threads(1)
    public void longAdder1(Shared s) {
        s.adder.increment();
    }

    @Benchmark
    @Threads(4)
    public void longAdder4(Shared s) {
        s.adder.increment();
    }

    @Benchmark
    @Threads(12)
    public void longAdder12(Shared s) {
        s.adder.increment();
    }

    @Benchmark
    @Threads(1)
    public void span(Shared s) {
        try (Span ignored = s.metrics.span(NAME)) {
            // empty body: measures the probe and record overhead alone
        }
    }

    /** The thread CPU clock a span reads twice; on macOS this is a Mach {@code thread_info} call. */
    @Benchmark
    @Threads(1)
    public long threadCpuTimeProbe() {
        return THREADS.getCurrentThreadCpuTime();
    }

    /** The allocation counter a span reads twice. */
    @Benchmark
    @Threads(1)
    public long allocatedBytesProbe() {
        return THREADS.getCurrentThreadAllocatedBytes();
    }

    private static final com.sun.management.ThreadMXBean THREADS =
            (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();
}
