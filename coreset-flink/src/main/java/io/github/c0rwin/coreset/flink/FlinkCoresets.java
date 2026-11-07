package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.WeightedPoint;
import java.time.Duration;
import java.util.Objects;
import org.apache.flink.api.common.typeinfo.Types;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;

/**
 * Coreset construction on Flink DataStreams.
 *
 * <p>Both pipelines start with a local stage running at the input's parallelism and chained to it (forward
 * partitioning): each subtask buffers records into chunks of {@link CoresetConfig#chunkSize()}, merge-reduces them,
 * and emits partial coresets. Results are bit-identical across runs and failure recoveries as long as every local
 * subtask receives the same records in the same order on replay, i.e. the input comes from a source with
 * deterministic split assignment and no {@code rebalance}/{@code shuffle} sits in front of the coreset stage.
 *
 * <p>The input must be typed {@link CoresetTypes#POINT}; a Kryo or POJO type for {@link WeightedPoint} is rejected.
 */
public final class FlinkCoresets {

    private FlinkCoresets() {
    }

    /**
     * Continuously updated coreset of an unbounded or bounded stream. Local subtasks emit a {@link PartialCoreset}
     * after every {@link CoresetConfig#chunksPerPartial()} chunks (plus the optional time triggers and at end of
     * input);
     * a merge stage with parallelism 1 merge-reduces them and emits {@link CoresetSnapshot}s per its triggers, and a
     * final one with {@code last = true} when the input ends.
     *
     * <p>Exactly once: both stages keep trees, buffers and sequence counters in checkpointed operator state, and every
     * compression is seeded from its position, never from the attempt. With count triggers only, the final snapshot of
     * a run that failed and recovered equals the undisturbed run's bit for bit. Time triggers keep exactly-once
     * accounting (no record lost or counted twice) but cut partials where the clock or watermark happened to be.
     */
    public static DataStream<CoresetSnapshot> stream(DataStream<WeightedPoint> points, CoresetConfig config) {
        Objects.requireNonNull(config, "config");
        requirePointType(points);
        SingleOutputStreamOperator<PartialCoreset> partials = points
                .transform("coreset-local", CoresetTypes.PARTIAL, new LocalCoresetOperator(config))
                .uid(config.uidPrefix() + "-local");
        if (points.getParallelism() > 0) {
            partials.setParallelism(points.getParallelism());
        }
        return partials.global()
                .transform("coreset-merge", CoresetTypes.SNAPSHOT, new GlobalCoresetOperator(config))
                .uid(config.uidPrefix() + "-merge")
                .setParallelism(1)
                .setMaxParallelism(1);
    }

    /**
     * One coreset per tumbling event-time window of length {@code size} (aligned to the epoch). Each local subtask
     * summarises its share of a window with the configured compressor; when the watermark passes the window's end, a
     * keyed window function merges the subtask partials (ordered by subtask) and compresses them once. Records need
     * event-time timestamps; records behind the watermark are dropped and counted.
     */
    public static DataStream<WindowCoreset> tumblingWindows(DataStream<WeightedPoint> points, CoresetConfig config,
                                                            Duration size) {
        Objects.requireNonNull(config, "config");
        Objects.requireNonNull(size, "size");
        if (size.toMillis() < 1) {
            throw new IllegalArgumentException("window size must be at least 1 ms: " + size);
        }
        requirePointType(points);
        SingleOutputStreamOperator<WindowPartial> partials = points
                .transform("coreset-window-local", CoresetTypes.WINDOW_PARTIAL,
                        new WindowCoresetOperator(config, size.toMillis()))
                .uid(config.uidPrefix() + "-window-local");
        if (points.getParallelism() > 0) {
            partials.setParallelism(points.getParallelism());
        }
        return partials
                .keyBy(WindowPartial::windowStart, Types.LONG)
                .window(TumblingEventTimeWindows.of(Duration.ofMillis(size.toMillis())))
                .process(new WindowMergeFunction(config), CoresetTypes.WINDOW_CORESET)
                .name("coreset-window-merge")
                .uid(config.uidPrefix() + "-window-merge");
    }

    private static void requirePointType(DataStream<WeightedPoint> points) {
        Objects.requireNonNull(points, "points");
        if (!CoresetTypes.POINT.equals(points.getType())) {
            throw new IllegalArgumentException("input stream has type " + points.getType() + "; declare it as"
                    + " CoresetTypes.POINT (e.g. .returns(CoresetTypes.POINT)) or register"
                    + " CoresetTypes.serializationConfig(), so points are not serialized with Kryo");
        }
    }
}
