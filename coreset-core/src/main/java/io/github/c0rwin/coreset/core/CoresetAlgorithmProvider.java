package io.github.c0rwin.coreset.core;

import java.util.Map;
import java.util.Set;

/**
 * Contributes coreset algorithms by id; the {@link java.util.ServiceLoader} service behind {@link CoresetAlgorithms}.
 *
 * <p>Register an implementation in {@code META-INF/services/io.github.c0rwin.coreset.core.CoresetAlgorithmProvider}
 * (class path) or with {@code provides ... with ...} (module path). It needs a public no-argument constructor and must
 * be thread-safe. Each id may be owned by one provider only: {@link CoresetAlgorithms#load()} fails if two providers
 * claim the same id. coreset-core registers {@link io.github.c0rwin.coreset.core.algo.BuiltInAlgorithms} for
 * {@code uniform}, {@code sensitivity} and {@code mean}.
 */
public interface CoresetAlgorithmProvider {

    /** Ids this provider owns: non-empty, no blank id, the same on every call. */
    Set<String> ids();

    /**
     * Builds the algorithm {@code id} from string parameters (as carried by RPC or configuration). Called only with an
     * id from {@link #ids()}.
     *
     * @return a non-null, stateless, thread-safe algorithm
     * @throws IllegalArgumentException if {@code params} are missing, unknown, unparsable or out of range for {@code id}
     */
    CoresetAlgorithm create(String id, Map<String, String> params);
}
