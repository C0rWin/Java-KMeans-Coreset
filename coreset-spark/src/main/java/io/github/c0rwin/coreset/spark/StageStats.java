package io.github.c0rwin.coreset.spark;

/**
 * Totals over the partial coresets one stage of the aggregation tree produced. Stage 0 holds one partial per input
 * partition, stage {@code r} the outputs of reduce round {@code r}, and the last stage the final coreset on the driver.
 *
 * @param stage          0-based stage index
 * @param name           {@code partition}, {@code round-r} or {@code final}
 * @param partials       number of partial coresets
 * @param points         weighted points summed over the partials
 * @param encodedBytes   {@link CoresetCodec} bytes summed over the partials (what crosses the network)
 * @param estimatedBytes heap estimate ({@code Coreset.estimatedBytes()}) summed over the partials
 */
public record StageStats(int stage, String name, long partials, long points, long encodedBytes, long estimatedBytes) {
}
