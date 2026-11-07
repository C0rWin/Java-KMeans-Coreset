package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import java.util.Map;

/**
 * Maps an algorithm id and its string parameters to a {@link CoresetAlgorithm}. Implementations must be thread-safe.
 *
 * <p>This is also the {@link java.util.ServiceLoader} service type used by the CLI (see
 * {@link ServiceLoaderAlgorithmResolver}): a provider that does not own {@code id} must throw
 * {@link UnknownAlgorithmException} so that other providers are consulted.
 */
@FunctionalInterface
public interface AlgorithmResolver {

    /**
     * @throws UnknownAlgorithmException if {@code id} is not known to this resolver
     * @throws IllegalArgumentException  if {@code params} are invalid for {@code id}
     */
    CoresetAlgorithm resolve(String id, Map<String, String> params);
}
