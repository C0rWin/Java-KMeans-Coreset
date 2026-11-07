package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.DistributionSummary;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * Bridges {@link MetricsSink} to Micrometer: counts become {@link Counter}s, samples {@link DistributionSummary}s and
 * timings {@link Timer}s, registered on first use and cached so the hot path is one map lookup. Micrometer drops
 * negative distribution samples (e.g. weight drift below zero); use the in-memory recorder for those.
 */
public final class MicrometerMetricsSink implements MetricsSink {

    private final MeterRegistry registry;
    private final ConcurrentHashMap<String, Counter> counters = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, DistributionSummary> summaries = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Timer> timers = new ConcurrentHashMap<>();

    public MicrometerMetricsSink(MeterRegistry registry) {
        this.registry = Objects.requireNonNull(registry, "registry");
    }

    @Override
    public void count(String name, long delta) {
        Counter c = counters.get(name);
        if (c == null) {
            c = counters.computeIfAbsent(name, n -> Counter.builder(n).register(registry));
        }
        c.increment(delta);
    }

    @Override
    public void record(String name, double value) {
        DistributionSummary s = summaries.get(name);
        if (s == null) {
            s = summaries.computeIfAbsent(name, n -> DistributionSummary.builder(n).register(registry));
        }
        s.record(value);
    }

    @Override
    public void timeNanos(String name, long nanos) {
        Timer t = timers.get(name);
        if (t == null) {
            t = timers.computeIfAbsent(name, n -> Timer.builder(n).register(registry));
        }
        t.record(nanos, TimeUnit.NANOSECONDS);
    }
}
