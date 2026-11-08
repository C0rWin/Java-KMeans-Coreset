package io.github.c0rwin.coreset.core.metrics;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.concurrent.atomic.LongAdder;

/**
 * In-memory {@link MetricsSink} whose record paths are lock-free: counters are {@link LongAdder}s; each distribution
 * keeps count, sum, exact min/max and a log-linear histogram, striped like {@link LongAdder}: threads that collide on
 * a stripe make the distribution grow more stripes (up to the next power of two of the processor count), so
 * concurrent recording does not serialise on one cache line. A map entry is created via
 * {@link ConcurrentHashMap#computeIfAbsent} only the first time a name is seen.
 *
 * <p>Histogram: {@value #SUB_BUCKETS} linear sub-buckets per power of two for magnitudes in
 * {@code [2^MIN_EXPONENT, 2^(MAX_EXPONENT + 1))}, mirrored for negative values, plus one bucket for magnitudes below
 * {@code 2^MIN_EXPONENT} (reported as 0). Larger magnitudes, including infinities, go to the outermost bucket.
 * Quantiles use the nearest-rank definition (the {@code ceil(p * n)}-th smallest sample), are reported at the bucket
 * midpoint and clamped to {@code [min, max]}; within the covered range the relative error is at most
 * {@link #QUANTILE_RELATIVE_ERROR}. Timings ({@link #timeNanos}) are stored as distributions in nanoseconds.
 *
 * <p>Spans opened on a default instance record wall time only; {@link #RecordingMetrics(boolean)} with {@code true}
 * adds the thread CPU and allocation probes (see {@link MetricsSink#threadProbes()}).
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
    private static final int STRIPE_LIMIT =
            Math.min(64, Integer.highestOneBit(Math.max(1, Runtime.getRuntime().availableProcessors() * 2 - 1)));

    private final ConcurrentHashMap<String, LongAdder> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Histogram> distributions = new ConcurrentHashMap<>();
    private final boolean threadProbes;

    /** Spans record wall time only. */
    public RecordingMetrics() {
        this(false);
    }

    /** @param threadProbes whether spans also record thread CPU time and allocated bytes (about 0.5-1.5 us each) */
    public RecordingMetrics(boolean threadProbes) {
        this.threadProbes = threadProbes;
    }

    @Override
    public boolean threadProbes() {
        return threadProbes;
    }

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

    /**
     * One distribution as a table of {@link Stripe}s: a sample goes to the stripe its thread's probe selects. When the
     * stripe's sum CAS fails, i.e. only after threads actually collided, the table doubles (up to
     * {@link #STRIPE_LIMIT}) or, once full, the thread moves to another stripe. An uncontended distribution keeps one
     * stripe; snapshots merge them, and the count is the number of bucketed samples.
     */
    private static final class Histogram {
        private static final VarHandle STRIPES;

        static {
            try {
                STRIPES = MethodHandles.lookup().findVarHandle(Histogram.class, "stripes", Stripe[].class);
            } catch (ReflectiveOperationException e) {
                throw new ExceptionInInitializerError(e);
            }
        }

        private volatile Stripe[] stripes = {new Stripe()};

        void add(double v) {
            int[] probe = PROBE.get();
            Stripe[] table = stripes;
            Stripe s = table[probe[0] & (table.length - 1)];
            if (!s.trySum(v)) {
                s = contended(table, probe);
                s.addSum(v);
            }
            s.addRest(v);
        }

        private Stripe contended(Stripe[] table, int[] probe) {
            if (table.length < STRIPE_LIMIT) {
                Stripe[] bigger = Arrays.copyOf(table, table.length * 2);
                for (int i = table.length; i < bigger.length; i++) {
                    bigger[i] = new Stripe();
                }
                STRIPES.compareAndSet(this, table, bigger); // losing the race means another thread grew it
            } else {
                int h = probe[0]; // xorshift: move this thread to another stripe
                h ^= h << 13;
                h ^= h >>> 17;
                h ^= h << 5;
                probe[0] = h;
            }
            Stripe[] now = stripes;
            return now[probe[0] & (now.length - 1)];
        }

        Distribution summary() {
            long[] negCounts = null;
            long[] posCounts = new long[BUCKETS];
            long zeroCount = 0;
            double s = 0.0;
            double lo = Double.POSITIVE_INFINITY;
            double hi = Double.NEGATIVE_INFINITY;
            for (Stripe st : stripes) {
                s += (double) Stripe.SUM.getVolatile(st);
                lo = Math.min(lo, (double) Stripe.MIN.getVolatile(st));
                hi = Math.max(hi, (double) Stripe.MAX.getVolatile(st));
                zeroCount += (long) Stripe.ZEROS.getVolatile(st);
                st.positive.addTo(posCounts);
                Chunks neg = st.negative;
                if (neg != null) {
                    if (negCounts == null) {
                        negCounts = new long[BUCKETS];
                    }
                    neg.addTo(negCounts);
                }
            }
            if (negCounts == null) {
                negCounts = new long[0];
            }
            long total = zeroCount;
            for (long c : negCounts) {
                total += c;
            }
            for (long c : posCounts) {
                total += c;
            }
            long n = total;
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

    /** Per-thread stripe selector, seeded from the thread id and rehashed when its stripe is contended. */
    private static final ThreadLocal<int[]> PROBE = ThreadLocal.withInitial(
            () -> new int[] {(int) ((Thread.currentThread().getId() * 0x9E3779B97F4A7C15L) >>> 32) | 1});

    /**
     * Sum, min, max and zero count of one stripe, updated with CAS (uncontended unless two threads share the stripe),
     * plus its buckets. Padding keeps neighbouring stripes' hot fields off a shared cache line.
     */
    @SuppressWarnings("unused")
    private static final class Stripe {
        static final VarHandle SUM;
        static final VarHandle MIN;
        static final VarHandle MAX;
        static final VarHandle ZEROS;
        static final VarHandle NEGATIVE;

        static {
            try {
                MethodHandles.Lookup l = MethodHandles.lookup();
                SUM = l.findVarHandle(Stripe.class, "sum", double.class);
                MIN = l.findVarHandle(Stripe.class, "min", double.class);
                MAX = l.findVarHandle(Stripe.class, "max", double.class);
                ZEROS = l.findVarHandle(Stripe.class, "zeros", long.class);
                NEGATIVE = l.findVarHandle(Stripe.class, "negative", Chunks.class);
            } catch (ReflectiveOperationException e) {
                throw new ExceptionInInitializerError(e);
            }
        }

        private long p0, p1, p2, p3, p4, p5, p6;
        private volatile double sum;
        private volatile double min = Double.POSITIVE_INFINITY;
        private volatile double max = Double.NEGATIVE_INFINITY;
        private volatile long zeros;
        final Chunks positive = new Chunks();
        // Negative samples are rare (e.g. drift), so their buckets are allocated on first use.
        volatile Chunks negative;
        private long q0, q1, q2, q3, q4, q5, q6;

        /** One strong CAS: failure means another thread updated this stripe in between. */
        boolean trySum(double v) {
            double s = sum;
            return SUM.compareAndSet(this, s, s + v);
        }

        void addSum(double v) {
            double s = sum;
            while (!SUM.weakCompareAndSet(this, s, s + v)) {
                s = sum;
            }
        }

        /** Everything but the sum. */
        void addRest(double v) {
            double lo = min;
            while (Double.compare(Math.min(lo, v), lo) != 0 && !MIN.weakCompareAndSet(this, lo, Math.min(lo, v))) {
                lo = min;
            }
            double hi = max;
            while (Double.compare(Math.max(hi, v), hi) != 0 && !MAX.weakCompareAndSet(this, hi, Math.max(hi, v))) {
                hi = max;
            }
            double a = Math.abs(v);
            if (a < MIN_MAGNITUDE) {
                ZEROS.getAndAdd(this, 1L);
            } else if (v > 0) {
                positive.increment(bucketOf(a));
            } else {
                Chunks neg = negative;
                if (neg == null) {
                    NEGATIVE.compareAndSet(this, null, new Chunks());
                    neg = negative;
                }
                neg.increment(bucketOf(a));
            }
        }
    }

    /** Buckets in one chunk of {@value #SUB_BUCKETS} per power of two, each allocated when first hit. */
    private static final class Chunks {
        private final AtomicReferenceArray<AtomicLongArray> chunks = new AtomicReferenceArray<>(BUCKETS / SUB_BUCKETS);

        void increment(int bucket) {
            int e = bucket >>> SUB_BITS;
            AtomicLongArray chunk = chunks.get(e);
            if (chunk == null) {
                chunks.compareAndSet(e, null, new AtomicLongArray(SUB_BUCKETS));
                chunk = chunks.get(e);
            }
            chunk.getAndIncrement(bucket & (SUB_BUCKETS - 1));
        }

        void addTo(long[] counts) {
            for (int e = 0; e < chunks.length(); e++) {
                AtomicLongArray chunk = chunks.get(e);
                if (chunk != null) {
                    for (int i = 0; i < SUB_BUCKETS; i++) {
                        counts[e * SUB_BUCKETS + i] += chunk.get(i);
                    }
                }
            }
        }
    }
}
