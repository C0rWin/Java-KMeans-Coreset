package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithmProvider;
import java.util.Map;
import java.util.Set;

/**
 * Provider of the algorithms shipped in coreset-core ({@code uniform}, {@code sensitivity}, {@code mean}), registered
 * as a {@link java.util.ServiceLoader} service; parameters and limits as in {@link Algorithms#create(String, Map)}.
 */
public final class BuiltInAlgorithms implements CoresetAlgorithmProvider {

    private static final Set<String> IDS = Set.of(Algorithms.UNIFORM, Algorithms.SENSITIVITY, Algorithms.MEAN);

    @Override
    public Set<String> ids() {
        return IDS;
    }

    @Override
    public CoresetAlgorithm create(String id, Map<String, String> params) {
        return Algorithms.create(id, params);
    }
}
