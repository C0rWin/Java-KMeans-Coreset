package io.github.c0rwin.coreset.core.metrics;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * In-memory, thread-safe {@link MetricsSink}. Counters are {@link LongAdder}s. Each distribution keeps exact count,
 * sum, min and max plus a uniform reservoir of {@link #RESERVOIR_SIZE} samples (Vitter's Algorithm R): quantiles are
 * exact while a distribution has at most {@code RESERVOIR_SIZE} samples and estimated from the reservoir afterwards.
 * Timings ({@link #timeNanos}) are stored as distributions in nanoseconds.
 */
public final class RecordingMetrics implements MetricsSink {

    /** Samples retained per distribution for quantile computation. */
    public static final int RESERVOIR_SIZE = 4096;

    private final ConcurrentHashMap<String, LongAdder> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Recorder> distributions = new ConcurrentHashMap<>();

    @Override
    public void count(String name, long delta) {
        LongAdder a = counters.get(name);
        if (a == null) {
            a = counters.computeIfAbsent(name, k -> new LongAdder());
        }
        a.add(delta);
    }

    /** @throws IllegalArgumentException if {@code value} is NaN */
    @Override
    public void record(String name, double value) {
        if (Double.isNaN(value)) {
            throw new IllegalArgumentException("NaN sample for " + name);
        }
        Recorder r = distributions.get(name);
        if (r == null) {
            r = distributions.computeIfAbsent(name, Recorder::new);
        }
        r.add(value);
    }

    @Override
    public void timeNanos(String name, long nanos) {
        record(name, nanos);
    }

    /** Consistent per metric; metrics updated concurrently may be captured at slightly different moments. */
    public MetricsSnapshot snapshot() {
        Map<String, Long> c = new HashMap<>();
        counters.forEach((k, v) -> c.put(k, v.sum()));
        Map<String, Distribution> d = new HashMap<>();
        distributions.forEach((k, v) -> d.put(k, v.summary()));
        return new MetricsSnapshot(c, d);
    }

    /** Drops all metrics; updates racing with reset may land on either side of it. */
    public void reset() {
        counters.clear();
        distributions.clear();
    }

    private static final class Recorder {
        private final SplittableRandom random;
        private double[] reservoir = new double[16];
        private long count;
        private double sum;
        private double min = Double.POSITIVE_INFINITY;
        private double max = Double.NEGATIVE_INFINITY;

        Recorder(String name) {
            random = new SplittableRandom(name.hashCode());
        }

        synchronized void add(double v) {
            long n = ++count;
            sum += v;
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            if (n <= RESERVOIR_SIZE) {
                int i = (int) (n - 1);
                if (i == reservoir.length) {
                    reservoir = Arrays.copyOf(reservoir, Math.min(2 * i, RESERVOIR_SIZE));
                }
                reservoir[i] = v;
            } else {
                long j = random.nextLong(n);
                if (j < RESERVOIR_SIZE) {
                    reservoir[(int) j] = v;
                }
            }
        }

        synchronized Distribution summary() {
            int m = (int) Math.min(count, RESERVOIR_SIZE);
            double[] s = Arrays.copyOf(reservoir, m);
            Arrays.sort(s);
            return new Distribution(count, sum, min, max, sum / count,
                    quantile(s, 0.50), quantile(s, 0.95), quantile(s, 0.99));
        }
    }

    static double quantile(double[] sorted, double p) {
        double h = (sorted.length - 1) * p;
        int lo = (int) Math.floor(h);
        int hi = Math.min(lo + 1, sorted.length - 1);
        return sorted[lo] + (h - lo) * (sorted[hi] - sorted[lo]);
    }
}
