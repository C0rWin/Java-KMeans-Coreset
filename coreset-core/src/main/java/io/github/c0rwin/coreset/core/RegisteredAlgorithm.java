package io.github.c0rwin.coreset.core;

import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Factory for a registered algorithm: {@link #create()} resolves {@code (id, params)} with
 * {@link CoresetAlgorithms#load()} in the JVM where it runs, so the provider's jar must be on that JVM's class path.
 *
 * @param id     algorithm id, e.g. {@code sensitivity}
 * @param params algorithm parameters, copied into an unmodifiable map sorted by key
 */
public record RegisteredAlgorithm(String id, Map<String, String> params) implements CoresetAlgorithmFactory {

    private static final long serialVersionUID = 1L;

    public RegisteredAlgorithm {
        Objects.requireNonNull(id, "id");
        params = Collections.unmodifiableSortedMap(new TreeMap<>(Objects.requireNonNull(params, "params")));
    }

    @Override
    public CoresetAlgorithm create() {
        return CoresetAlgorithms.load().create(id, params);
    }
}
