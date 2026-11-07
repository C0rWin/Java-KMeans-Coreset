package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.validate.ErrorBudget;
import java.util.List;

/**
 * What one distributed build did. All counters are carried inside the partial coresets and summed on the driver, so
 * they are exact and unaffected by task retries (unlike Spark accumulators updated in transformations).
 *
 * @param partitions          input partitions (= partition coresets)
 * @param chunkSize           input points per level-0 batch of each partition's merge-reduce tree
 * @param fanIn               partials merged per reduce operation in the aggregation tree
 * @param reduceRounds        executor-side reduce rounds (shuffles) before the final merge on the driver
 * @param partitionTreeLevels largest number of levels any partition's merge-reduce tree reached
 * @param compressionDepth    most compressions any input point passed through, partition tree to final coreset; this
 *                            is the L that per-level errors compound over
 * @param reduceOps           compressions performed in total
 * @param inputPoints         input points consumed
 * @param inputWeight         input weight consumed, summed per partition then in partition order
 * @param stages              per-stage totals, partition stage first, final coreset last
 * @param shippedEncodedBytes encoded bytes of every partial that left its task (all stages but the final one)
 * @param shippedEstimatedBytes heap estimate of the same partials
 * @param wallNanos           driver wall time of the build
 */
public record BuildMetrics(int partitions, int chunkSize, int fanIn, int reduceRounds, int partitionTreeLevels,
                           int compressionDepth, long reduceOps, long inputPoints, double inputWeight,
                           List<StageStats> stages, long shippedEncodedBytes, long shippedEstimatedBytes,
                           long wallNanos) {

    public BuildMetrics {
        stages = List.copyOf(stages);
    }

    /**
     * Splits an end-to-end {@code (eps, delta)} target over this build's tree (research notes §6a item 6): each of the
     * {@link #compressionDepth} levels may use {@code eps_l = (1 + eps)^(1/L) - 1} and each of the {@link #reduceOps}
     * compressions may fail with probability {@code delta / reduceOps}.
     */
    public ErrorBudget errorBudget(double eps, double delta) {
        if (!(eps > 0.0) || Double.isInfinite(eps)) {
            throw new IllegalArgumentException("eps must be finite and > 0: " + eps);
        }
        if (!(delta > 0.0 && delta < 1.0)) {
            throw new IllegalArgumentException("delta must be in (0, 1): " + delta);
        }
        double epsPerLevel = compressionDepth == 0 ? eps : Math.expm1(Math.log1p(eps) / compressionDepth);
        double deltaPerReduce = reduceOps == 0 ? delta : delta / reduceOps;
        return new ErrorBudget(eps, delta, compressionDepth, reduceOps, epsPerLevel, deltaPerReduce);
    }
}
