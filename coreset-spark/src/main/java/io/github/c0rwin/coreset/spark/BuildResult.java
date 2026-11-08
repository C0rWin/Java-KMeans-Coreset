package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.Objects;

/** Final coreset of a distributed build plus what the build did. */
public record BuildResult(Coreset coreset, BuildMetrics metrics) {

    public BuildResult {
        Objects.requireNonNull(coreset, "coreset");
        Objects.requireNonNull(metrics, "metrics");
    }
}
