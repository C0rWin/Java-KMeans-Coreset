package io.github.c0rwin.coreset.core.metrics;

/**
 * Summary of a sample distribution. Quantiles use linear interpolation between closest ranks
 * ({@code h = (n - 1) * p}); see {@link RecordingMetrics} for when they are exact.
 */
public record Distribution(
        long count, double sum, double min, double max, double mean, double p50, double p95, double p99) {
}
