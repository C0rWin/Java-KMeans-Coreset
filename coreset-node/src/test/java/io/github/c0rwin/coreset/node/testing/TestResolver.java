package io.github.c0rwin.coreset.node.testing;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.node.AlgorithmResolver;
import io.github.c0rwin.coreset.node.UnknownAlgorithmException;
import java.util.HashMap;
import java.util.Map;

/**
 * Resolver over {@link TestAlgorithms}: {@code identity}, {@code merge-equal}, {@code sample} (param {@code m}),
 * {@code failing} (param {@code poison}); extra fixed instances can be added per test. Registered as a test-scope
 * {@link java.util.ServiceLoader} provider.
 */
public final class TestResolver implements AlgorithmResolver {

    private final Map<String, CoresetAlgorithm> extra;

    public TestResolver() {
        this(Map.of());
    }

    public TestResolver(Map<String, CoresetAlgorithm> extra) {
        this.extra = new HashMap<>(extra);
    }

    @Override
    public CoresetAlgorithm resolve(String id, Map<String, String> params) {
        CoresetAlgorithm fixed = extra.get(id);
        if (fixed != null) {
            return fixed;
        }
        return switch (id) {
            case "identity" -> TestAlgorithms.IDENTITY;
            case "merge-equal" -> TestAlgorithms.MERGE_EQUAL;
            case "sample" -> TestAlgorithms.sample(Integer.parseInt(params.getOrDefault("m", "4")));
            case "failing" -> TestAlgorithms.failingOn(Double.parseDouble(params.getOrDefault("poison", "13")));
            default -> throw new UnknownAlgorithmException(id);
        };
    }
}
