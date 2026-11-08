package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.List;

/** Forwards every event to several sinks (e.g. an in-memory recorder for JSON stats plus a Prometheus registry). */
public final class TeeMetricsSink implements MetricsSink {

    private final MetricsSink[] sinks;

    private TeeMetricsSink(MetricsSink[] sinks) {
        this.sinks = sinks;
    }

    /** True if any sink wants thread probes: they are read once per span and forwarded to all. */
    @Override
    public boolean threadProbes() {
        for (MetricsSink s : sinks) {
            if (s.threadProbes()) {
                return true;
            }
        }
        return false;
    }

    /** Returns {@link MetricsSink#NOOP} for no real sinks and the sink itself for exactly one. */
    public static MetricsSink of(MetricsSink... sinks) {
        MetricsSink[] real = List.of(sinks).stream().filter(s -> s != MetricsSink.NOOP).toArray(MetricsSink[]::new);
        if (real.length == 0) {
            return MetricsSink.NOOP;
        }
        return real.length == 1 ? real[0] : new TeeMetricsSink(real);
    }

    @Override
    public void count(String name, long delta) {
        for (MetricsSink s : sinks) {
            s.count(name, delta);
        }
    }

    @Override
    public void record(String name, double value) {
        for (MetricsSink s : sinks) {
            s.record(name, value);
        }
    }

    @Override
    public void timeNanos(String name, long nanos) {
        for (MetricsSink s : sinks) {
            s.timeNanos(name, nanos);
        }
    }
}
