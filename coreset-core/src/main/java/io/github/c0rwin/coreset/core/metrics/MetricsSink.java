package io.github.c0rwin.coreset.core.metrics;

/**
 * Receives counters, distribution samples and timings. Implementations must be thread-safe.
 *
 * <p>Instrumented code emits coarse-grained events only (per call, per iteration, per merge), never per distance
 * evaluation, so {@link #NOOP} costs nothing measurable on hot paths. Metric names are listed in {@link MetricNames}.
 */
public interface MetricsSink {

    /** Discards everything; spans opened on it skip all clock and MXBean reads. */
    MetricsSink NOOP = NoopMetricsSink.INSTANCE;

    void count(String name, long delta);

    /** Adds one sample to the distribution {@code name}. */
    void record(String name, double value);

    /** Adds one duration sample, in nanoseconds, to the distribution {@code name}. */
    void timeNanos(String name, long nanos);

    default Span span(String name) {
        return Span.start(this, name);
    }
}
