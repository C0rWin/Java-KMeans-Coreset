package io.github.c0rwin.coreset.node;

import java.util.Map;
import java.util.Set;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.algo.Algorithms;

/** {@link AlgorithmResolver} provider for the algorithms shipped in coreset-core ({@link Algorithms}). */
public final class CoreAlgorithmsResolver implements AlgorithmResolver {

    private static final Set<String> IDS = Set.of(Algorithms.UNIFORM, Algorithms.SENSITIVITY, Algorithms.MEAN);

    @Override
    public CoresetAlgorithm resolve(String id, Map<String, String> params) {
        // Unknown ids must be UnknownAlgorithmException so ServiceLoaderAlgorithmResolver tries other providers.
        if (!IDS.contains(id)) {
            throw new UnknownAlgorithmException(id);
        }
        return Algorithms.create(id, params);
    }
}
