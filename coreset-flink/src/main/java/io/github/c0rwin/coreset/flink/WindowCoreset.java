package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.Objects;

/**
 * Coreset of one tumbling event-time window, emitted by {@link FlinkCoresets#tumblingWindows}.
 *
 * @param windowStart start of the window, inclusive, in event-time milliseconds
 * @param windowEnd   end of the window, exclusive
 * @param coreset     the summary
 * @param records     input records summarised
 * @param weight      their total input weight
 * @param partials    subtask partials merged into it
 * @param depth       compressions on the deepest point's path inside {@code coreset}
 */
public record WindowCoreset(long windowStart, long windowEnd, Coreset coreset, long records, double weight,
                            int partials, int depth) {

    public WindowCoreset {
        Objects.requireNonNull(coreset, "coreset");
    }
}
