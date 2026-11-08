package io.github.c0rwin.coreset.core.metrics;

/**
 * Summary of a sample distribution. Count, sum, min, max and mean are exact; quantiles follow the nearest-rank
 * definition and may be approximate (see {@link RecordingMetrics} for the error bound).
 */
public record Distribution(
        long count, double sum, double min, double max, double mean, double p50, double p95, double p99) {
}
