package io.github.c0rwin.coreset.examples.lightweight;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithmFactory;

/**
 * {@link LightweightCoreset} as a code-supplied engine recipe, for Spark and Flink jobs that ship the algorithm in the
 * job jar instead of registering it. A named class rather than a lambda, so its identity (and so engine checkpoint
 * compatibility) depends only on {@code m}.
 */
public final class LightweightCoresetFactory implements CoresetAlgorithmFactory {

    private static final long serialVersionUID = 1L;

    private final int m;

    /** @throws IllegalArgumentException if {@code m < 1} */
    public LightweightCoresetFactory(int m) {
        new LightweightCoreset(m); // validates on the driver, not first on a worker
        this.m = m;
    }

    @Override
    public CoresetAlgorithm create() {
        return new LightweightCoreset(m);
    }

    @Override
    public String identity() {
        return "lightweight-factory(m=" + m + ")";
    }
}
