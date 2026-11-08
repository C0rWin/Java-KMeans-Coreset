package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.Objects;

/**
 * Coreset of the records one local subtask received since its previous partial, sent to the merge stage.
 *
 * @param subtask     index of the emitting subtask
 * @param parallelism parallelism of the local stage, checked by the merge stage to reject a rescaled restore
 * @param sequence    0-based index of this partial among the subtask's partials
 * @param coreset     the summary
 * @param records     input records summarised
 * @param weight      their total input weight
 * @param depth       compressions on the deepest point's path inside {@code coreset}
 */
public record PartialCoreset(int subtask, int parallelism, long sequence, Coreset coreset, long records,
                             double weight, int depth) {

    public PartialCoreset {
        Objects.requireNonNull(coreset, "coreset");
    }
}
