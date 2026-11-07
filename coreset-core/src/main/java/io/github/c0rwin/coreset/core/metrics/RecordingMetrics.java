package io.github.c0rwin.coreset.core.metrics;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.DoubleAccumulator;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.concurrent.atomic.LongAdder;

/**
 * In-memory {@link MetricsSink} whose record paths are lock-free: counters are {@link LongAdder}s; each distribution
 * keeps count ({@link LongAdder}), sum ({@link DoubleAdder}), exact min/max ({@link DoubleAccumulator}) and a
 * log-linear histogram of {@link AtomicLongArray} buckets. A map entry is created via
 * {@link ConcurrentHashMap#computeIfAbsent} only the first time a name is seen.
 *
 * <p>Histogram: {@value #SUB_BUCKETS} linear sub-buckets per power of two for magnitudes in
 * {@code [2^MIN_EXPONENT, 2^(MAX_EXPONENT + 1))}, mirrored for negative values, plus one bucket for magnitudes below
 * {@code 2^MIN_EXPONENT} (reported as 0). Larger magnitudes, including infinities, go to the outermost bucket.
 * Quantiles use the nearest-rank definition (the {@code ceil(p * n)}-th smallest sample), are reported at the bucket
 * midpoint and clamped to {@code [min, max]}; within the covered range the relative error is at most
 * {@link #QUANTILE_RELATIVE_ERROR}. Timings ({@link #timeNanos}) are stored as distributions in nanoseconds.
 */
public final class RecordingMetrics implements MetricsSink {

    /** Linear sub-buckets per power of two. */
    public static final int SUB_BUCKETS = 64;
    /** Smallest magnitude exponent tracked; {@code |v| < 2^MIN_EXPONENT} (about 2.3e-10) counts as 0. */
    public static final int MIN_EXPONENT = -32;
    /** Largest magnitude exponent tracked; covers {@code Long.MAX_VALUE} nanoseconds and bytes. */
    public static final int MAX_EXPONENT = 63;
    /** Bound on {@code |reported - true| / |true|} for quantiles of samples within the covered range. */
    public static final double QUANTILE_RELATIVE_ERROR = 1.0 / (2 * SUB_BUCKETS);

    private static final int SUB_BITS = Integer.numberOfTrailingZeros(SUB_BUCKETS);
    private static final int BUCKETS = (MAX_EXPONENT - MIN_EXPONENT + 1) * SUB_BUCKETS;
    private static final double MIN_MAGNITUDE = Math.scalb(1.0, MIN_EXPONENT);

    private final ConcurrentHashMap<String, LongAdder> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Histogram> distributions = new ConcurrentHashMap<>();

    @Override
    public void count(String name, long delta) {
        LongAdder a = counters.get(name);
        if (a == null) {
            a = counters.computeIfAbsent(name, k -> new LongAdder());
        }
        a.add(delta);
    }

    /**
     * Adds a sample. A NaN sample is not added to the distribution: it increments the counter
     * {@code name + }{@value MetricNames#NAN_SAMPLES_SUFFIX} instead, so instrumentation never fails the data path.
     */
    @Override
    public void record(String name, double value) {
        if (Double.isNaN(value)) {
            count(name + MetricNames.NAN_SAMPLES_SUFFIX, 1);
            return;
        }
        Histogram h = distributions.get(name);
        if (h == null) {
            h = distributions.computeIfAbsent(name, k -> new Histogram());
        }
        h.add(value);
    }

    @Override
    public void timeNanos(String name, long nanos) {
        record(name, nanos);
    }

    /**
     * Copies all metrics. Not atomic across concurrent updates: a distribution's count, sum and buckets may each
     * include or miss an in-flight sample.
     */
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

    /** Bucket of a magnitude {@code a >= MIN_MAGNITUDE}: (exponent, top mantissa bits), clamped to the last bucket. */
    static int bucketOf(double a) {
        int e = Math.getExponent(a);
        if (e > MAX_EXPONENT) {
            return BUCKETS - 1;
        }
        int sub = (int) ((Double.doubleToRawLongBits(a) >>> (52 - SUB_BITS)) & (SUB_BUCKETS - 1));
        return (e - MIN_EXPONENT) * SUB_BUCKETS + sub;
    }

    static double midpointOf(int bucket) {
        int e = bucket / SUB_BUCKETS + MIN_EXPONENT;
        int sub = bucket % SUB_BUCKETS;
        return Math.scalb(1.0 + (sub + 0.5) / SUB_BUCKETS, e);
    }

    private static final class Histogram {
        private final LongAdder count = new LongAdder();
        private final DoubleAdder sum = new DoubleAdder();
        private final DoubleAccumulator min = new DoubleAccumulator(Math::min, Double.POSITIVE_INFINITY);
        private final DoubleAccumulator max = new DoubleAccumulator(Math::max, Double.NEGATIVE_INFINITY);
        private final LongAdder zeros = new LongAdder();
        private final AtomicLongArray positive = new AtomicLongArray(BUCKETS);
        // Negative samples are rare (e.g. drift), so their buckets are allocated on first use.
        private final AtomicReference<AtomicLongArray> negative = new AtomicReference<>();

        void add(double v) {
            count.increment();
            sum.add(v);
            min.accumulate(v);
            max.accumulate(v);
            double a = Math.abs(v);
            if (a < MIN_MAGNITUDE) {
                zeros.increment();
            } else if (v > 0) {
                positive.incrementAndGet(bucketOf(a));
            } else {
                AtomicLongArray neg = negative.get();
                if (neg == null) {
                    negative.compareAndSet(null, new AtomicLongArray(BUCKETS));
                    neg = negative.get();
                }
                neg.incrementAndGet(bucketOf(a));
            }
        }

        Distribution summary() {
            AtomicLongArray neg = negative.get();
            long[] negCounts = new long[neg == null ? 0 : BUCKETS];
            long[] posCounts = new long[BUCKETS];
            long total = zeros.sum();
            long zeroCount = total;
            for (int i = 0; i < negCounts.length; i++) {
                negCounts[i] = neg.get(i);
                total += negCounts[i];
            }
            for (int i = 0; i < BUCKETS; i++) {
                posCounts[i] = positive.get(i);
                total += posCounts[i];
            }
            long n = count.sum();
            double s = sum.sum();
            double lo = min.get();
            double hi = max.get();
            return new Distribution(n, s, lo, hi, n == 0 ? 0.0 : s / n,
                    quantile(negCounts, zeroCount, posCounts, total, 0.50, lo, hi),
                    quantile(negCounts, zeroCount, posCounts, total, 0.95, lo, hi),
                    quantile(negCounts, zeroCount, posCounts, total, 0.99, lo, hi));
        }

        private static double quantile(long[] neg, long zeros, long[] pos, long total, double p, double lo,
                                       double hi) {
            if (total == 0) {
                return 0.0;
            }
            double v = valueAtRank(neg, zeros, pos, Math.max(1, (long) Math.ceil(p * total)), hi);
            return Math.min(hi, Math.max(lo, v));
        }

        /** Midpoint of the bucket holding the {@code rank}-th smallest sample (1-based), in ascending value order. */
        private static double valueAtRank(long[] neg, long zeros, long[] pos, long rank, double fallback) {
            long seen = 0;
            for (int i = neg.length - 1; i >= 0; i--) {
                seen += neg[i];
                if (seen >= rank) {
                    return -midpointOf(i);
                }
            }
            seen += zeros;
            if (seen >= rank) {
                return 0.0;
            }
            for (int i = 0; i < pos.length; i++) {
                seen += pos[i];
                if (seen >= rank) {
                    return midpointOf(i);
                }
            }
            return fallback;
        }
    }
}
