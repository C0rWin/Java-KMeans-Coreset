package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.Objects;

/**
 * Global coreset emitted by the merge stage of {@link FlinkCoresets#stream}.
 *
 * @param coreset    union of every level the merge stage holds, uncompressed
 * @param compressed {@code coreset} compressed once, or null when snapshot compression is off
 * @param records    input records summarised
 * @param weight     their total input weight
 * @param partials   partial coresets merged so far
 * @param levels     occupied tree levels behind {@code coreset}
 * @param depth      compressions on the deepest point's path inside {@code coreset}; {@code compressed} adds one
 * @param last       true for the snapshot emitted when the input ended
 */
public record CoresetSnapshot(Coreset coreset, Coreset compressed, long records, double weight, long partials,
                              int levels, int depth, boolean last) {

    public CoresetSnapshot {
        Objects.requireNonNull(coreset, "coreset");
    }
}
