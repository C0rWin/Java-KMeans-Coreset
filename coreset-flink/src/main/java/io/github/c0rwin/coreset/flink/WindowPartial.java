package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.Objects;

/**
 * Coreset of one tumbling window's records seen by one local subtask.
 *
 * @param windowStart start of the window, inclusive, in event-time milliseconds
 * @param subtask     index of the emitting subtask; the merge orders partials by it
 * @param coreset     the summary
 * @param records     input records summarised
 * @param weight      their total input weight
 * @param depth       compressions on the deepest point's path inside {@code coreset}
 */
public record WindowPartial(long windowStart, int subtask, Coreset coreset, long records, double weight, int depth) {

    public WindowPartial {
        Objects.requireNonNull(coreset, "coreset");
    }
}
