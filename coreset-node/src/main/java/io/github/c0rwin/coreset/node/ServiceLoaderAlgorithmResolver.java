package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import java.util.List;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Resolves algorithms through every {@link AlgorithmResolver} registered under
 * {@code META-INF/services/io.github.c0rwin.coreset.node.AlgorithmResolver}, in classpath order; the first provider
 * that does not throw {@link UnknownAlgorithmException} wins.
 */
public final class ServiceLoaderAlgorithmResolver implements AlgorithmResolver {

    private final List<AlgorithmResolver> providers;

    private ServiceLoaderAlgorithmResolver(List<AlgorithmResolver> providers) {
        this.providers = List.copyOf(providers);
    }

    /** Loads providers visible to the context class loader. */
    public static ServiceLoaderAlgorithmResolver load() {
        return load(Thread.currentThread().getContextClassLoader());
    }

    /** @throws IllegalStateException if no provider is registered */
    public static ServiceLoaderAlgorithmResolver load(ClassLoader loader) {
        List<AlgorithmResolver> found = ServiceLoader.load(AlgorithmResolver.class, loader).stream()
                .map(ServiceLoader.Provider::get)
                .toList();
        if (found.isEmpty()) {
            throw new IllegalStateException("no AlgorithmResolver provider on the classpath; add a jar that declares "
                    + "META-INF/services/" + AlgorithmResolver.class.getName());
        }
        return new ServiceLoaderAlgorithmResolver(found);
    }

    public List<AlgorithmResolver> providers() {
        return providers;
    }

    @Override
    public CoresetAlgorithm resolve(String id, Map<String, String> params) {
        for (AlgorithmResolver provider : providers) {
            CoresetAlgorithm algorithm;
            try {
                algorithm = provider.resolve(id, params);
            } catch (UnknownAlgorithmException e) {
                continue;
            }
            if (algorithm == null) {
                throw new IllegalStateException(provider.getClass().getName() + " returned null for " + id);
            }
            return algorithm;
        }
        throw new UnknownAlgorithmException(id);
    }
}
