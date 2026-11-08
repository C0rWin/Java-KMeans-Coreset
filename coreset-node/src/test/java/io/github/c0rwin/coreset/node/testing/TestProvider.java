package io.github.c0rwin.coreset.node.testing;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithmProvider;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Provider over {@link TestAlgorithms}: {@code identity}, {@code merge-equal}, {@code sample} (param {@code m}),
 * {@code failing} (param {@code poison}); extra fixed instances can be added per test. Registered as a test-scope
 * {@link java.util.ServiceLoader} provider, which the coreset-grpc CLI tests rely on.
 */
public final class TestProvider implements CoresetAlgorithmProvider {

    private static final Set<String> FIXED = Set.of("identity", "merge-equal", "sample", "failing");

    private final Map<String, CoresetAlgorithm> extra;
    private final Set<String> ids;

    public TestProvider() {
        this(Map.of());
    }

    public TestProvider(Map<String, CoresetAlgorithm> extra) {
        this.extra = new HashMap<>(extra);
        TreeSet<String> all = new TreeSet<>(FIXED);
        all.addAll(extra.keySet());
        this.ids = Set.copyOf(all);
    }

    /** A registry over the test algorithms only. */
    public static CoresetAlgorithms registry() {
        return registry(Map.of());
    }

    /** A registry over the test algorithms plus {@code extra}. */
    public static CoresetAlgorithms registry(Map<String, CoresetAlgorithm> extra) {
        return CoresetAlgorithms.of(new TestProvider(extra));
    }

    @Override
    public Set<String> ids() {
        return ids;
    }

    @Override
    public CoresetAlgorithm create(String id, Map<String, String> params) {
        CoresetAlgorithm fixed = extra.get(id);
        if (fixed != null) {
            return fixed;
        }
        return switch (id) {
            case "identity" -> TestAlgorithms.IDENTITY;
            case "merge-equal" -> TestAlgorithms.MERGE_EQUAL;
            case "sample" -> TestAlgorithms.sample(Integer.parseInt(params.getOrDefault("m", "4")));
            case "failing" -> TestAlgorithms.failingOn(Double.parseDouble(params.getOrDefault("poison", "13")));
            default -> throw new IllegalArgumentException("not a test algorithm: " + id);
        };
    }
}
