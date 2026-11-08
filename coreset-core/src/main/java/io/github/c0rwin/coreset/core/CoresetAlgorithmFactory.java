package io.github.c0rwin.coreset.core;

import java.io.Serializable;

/**
 * Serializable recipe for a {@link CoresetAlgorithm}, shipped to the JVMs that compress (Spark executors, Flink task
 * managers) and called there. Use {@link RegisteredAlgorithm} to resolve a registered id on each JVM, or implement
 * this interface for an algorithm supplied in code; the implementation and everything it references must then be on
 * the job's class path and serializable.
 */
@FunctionalInterface
public interface CoresetAlgorithmFactory extends Serializable {

    /**
     * May be called many times, on any JVM of the job.
     *
     * @return a non-null, stateless, thread-safe algorithm
     * @throws IllegalArgumentException if the recipe is invalid
     */
    CoresetAlgorithm create();
}
