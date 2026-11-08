package io.github.c0rwin.coreset.core.validate;

import io.github.c0rwin.coreset.core.SparseVector;
import java.util.List;
import java.util.Objects;

/**
 * A candidate center set Q used to probe a coreset, labelled by the generator that produced it.
 *
 * @param generator label such as {@code kmeans++@P#0}; reported as the argmax of a distortion run
 * @param centers   non-empty, immutable copy
 */
public record Candidate(String generator, List<SparseVector> centers) {

    public Candidate {
        Objects.requireNonNull(generator, "generator");
        centers = List.copyOf(centers);
        if (centers.isEmpty()) {
            throw new IllegalArgumentException("centers must not be empty");
        }
    }
}
