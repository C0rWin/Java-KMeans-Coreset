package io.github.c0rwin.coreset.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * Resolves algorithm ids to {@link CoresetAlgorithm}s across {@link CoresetAlgorithmProvider}s; the single entry point
 * every layer (node, gRPC CLIs, Spark, Flink, testbed) uses to turn {@code (id, params)} into an algorithm.
 *
 * <p>{@link #load()} discovers providers with {@link ServiceLoader}, so an algorithm packaged in its own jar with a
 * {@code META-INF/services/io.github.c0rwin.coreset.core.CoresetAlgorithmProvider} entry is available wherever that
 * jar is on the class path. Conflicts are reported when the registry is built, not on first use: two providers
 * claiming one id, a provider with no or blank ids, or no provider at all is an {@link IllegalStateException}.
 *
 * <p>Immutable and thread-safe.
 */
public final class CoresetAlgorithms {

    private static final String SERVICE_FILE = "META-INF/services/" + CoresetAlgorithmProvider.class.getName();

    private final Map<String, CoresetAlgorithmProvider> byId;
    private final SortedSet<String> ids;

    private CoresetAlgorithms(TreeMap<String, CoresetAlgorithmProvider> byId) {
        this.byId = Collections.unmodifiableMap(byId);
        this.ids = Collections.unmodifiableSortedSet(new TreeSet<>(byId.keySet()));
    }

    /** Providers visible to the thread's context class loader, or to coreset-core's loader if there is none. */
    public static CoresetAlgorithms load() {
        ClassLoader context = Thread.currentThread().getContextClassLoader();
        return load(context != null ? context : CoresetAlgorithms.class.getClassLoader());
    }

    /**
     * Providers visible to {@code loader}: class-path {@code META-INF/services} entries and, for named modules,
     * {@code provides} clauses.
     *
     * @throws IllegalStateException if no provider is found, or the providers conflict (see the class comment)
     * @throws java.util.ServiceConfigurationError if a registered provider cannot be instantiated
     */
    public static CoresetAlgorithms load(ClassLoader loader) {
        Objects.requireNonNull(loader, "loader");
        List<CoresetAlgorithmProvider> found = new ArrayList<>();
        ServiceLoader.load(CoresetAlgorithmProvider.class, loader).forEach(found::add);
        if (found.isEmpty()) {
            throw new IllegalStateException("no " + CoresetAlgorithmProvider.class.getSimpleName() + " visible to "
                    + loader + "; coreset-core registers the built-in algorithms in " + SERVICE_FILE
                    + ", is coreset-core on this class path?");
        }
        return of(found);
    }

    /** {@link #of(List)} over {@code providers}. */
    public static CoresetAlgorithms of(CoresetAlgorithmProvider... providers) {
        return of(List.of(providers));
    }

    /**
     * A registry over exactly {@code providers}, without service discovery (embedding, tests).
     *
     * @throws IllegalStateException if the providers conflict (see the class comment)
     */
    public static CoresetAlgorithms of(List<? extends CoresetAlgorithmProvider> providers) {
        TreeMap<String, CoresetAlgorithmProvider> byId = new TreeMap<>();
        for (CoresetAlgorithmProvider p : providers) {
            Objects.requireNonNull(p, "provider");
            Set<String> declared = p.ids();
            if (declared == null || declared.isEmpty()) {
                throw new IllegalStateException(name(p) + " declares no algorithm ids");
            }
            for (String id : declared) {
                if (id == null || id.isBlank()) {
                    throw new IllegalStateException(name(p) + " declares a blank algorithm id");
                }
                CoresetAlgorithmProvider previous = byId.putIfAbsent(id, p);
                if (previous != null) {
                    throw new IllegalStateException("algorithm id '" + id + "' is provided by both " + name(previous)
                            + " and " + name(p) + "; remove one of them from the class path");
                }
            }
        }
        return new CoresetAlgorithms(byId);
    }

    /** Every resolvable id, sorted. */
    public SortedSet<String> ids() {
        return ids;
    }

    /**
     * Builds algorithm {@code id} with its provider.
     *
     * @throws UnknownAlgorithmException if no provider owns {@code id}
     * @throws IllegalArgumentException  if the provider rejects {@code params}
     * @throws IllegalStateException     if the provider returns null
     */
    public CoresetAlgorithm create(String id, Map<String, String> params) {
        Objects.requireNonNull(id, "id");
        Map<String, String> copy = Map.copyOf(Objects.requireNonNull(params, "params"));
        CoresetAlgorithmProvider provider = byId.get(id);
        if (provider == null) {
            throw new UnknownAlgorithmException(id, ids);
        }
        CoresetAlgorithm algorithm = provider.create(id, copy);
        if (algorithm == null) {
            throw new IllegalStateException(name(provider) + " returned null for algorithm '" + id + "'");
        }
        return algorithm;
    }

    @Override
    public String toString() {
        Map<String, String> owners = new TreeMap<>();
        byId.forEach((id, p) -> owners.put(id, name(p)));
        return "CoresetAlgorithms" + owners;
    }

    private static String name(CoresetAlgorithmProvider p) {
        return p.getClass().getName();
    }
}
