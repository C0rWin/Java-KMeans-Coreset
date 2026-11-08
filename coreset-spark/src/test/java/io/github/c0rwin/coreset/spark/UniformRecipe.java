package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithmFactory;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;

/** Code-supplied recipe for the built-in uniform coreset; a named class, so tasks deserialize it by name. */
final class UniformRecipe implements CoresetAlgorithmFactory {

    private static final long serialVersionUID = 1L;

    private final int m;

    UniformRecipe(int m) {
        this.m = m;
    }

    @Override
    public CoresetAlgorithm create() {
        return new UniformCoreset(m);
    }
}
