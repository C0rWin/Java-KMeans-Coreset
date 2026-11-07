package io.github.c0rwin.coreset.core.metrics;

final class NoopMetricsSink implements MetricsSink {

    static final NoopMetricsSink INSTANCE = new NoopMetricsSink();

    private NoopMetricsSink() {
    }

    @Override
    public void count(String name, long delta) {
    }

    @Override
    public void record(String name, double value) {
    }

    @Override
    public void timeNanos(String name, long nanos) {
    }

    @Override
    public String toString() {
        return "MetricsSink.NOOP";
    }
}
