package io.github.c0rwin.coreset.core;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.random.RandomGenerator;

/**
 * Reduces a weighted point set to a smaller one with approximately the same k-means cost for every set of centers.
 *
 * <p>Implementations must be stateless and thread-safe; all randomness must come from {@code rng} so that results are
 * reproducible for a seeded generator.
 */
public interface CoresetAlgorithm {

    Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics);

    default Coreset compress(Coreset input, RandomGenerator rng) {
        return compress(input, rng, MetricsSink.NOOP);
    }
}
