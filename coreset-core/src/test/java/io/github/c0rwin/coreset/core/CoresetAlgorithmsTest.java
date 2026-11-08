package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.algo.BuiltInAlgorithms;
import io.github.c0rwin.coreset.core.algo.MeanCoreset;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CoresetAlgorithmsTest {

    /** Provider over fixed ids, each answered with the identity algorithm. */
    public static final class FixedIds implements CoresetAlgorithmProvider {
        private final Set<String> ids;

        public FixedIds() {
            this(Set.of("fixed"));
        }

        FixedIds(Set<String> ids) {
            this.ids = ids;
        }

        @Override
        public Set<String> ids() {
            return ids;
        }

        @Override
        public CoresetAlgorithm create(String id, Map<String, String> params) {
            return (input, rng, metrics) -> input;
        }
    }

    /** Claims {@code uniform}, which coreset-core already registers. */
    public static final class ClaimsUniform implements CoresetAlgorithmProvider {
        @Override
        public Set<String> ids() {
            return Set.of("uniform");
        }

        @Override
        public CoresetAlgorithm create(String id, Map<String, String> params) {
            throw new AssertionError("never resolved");
        }
    }

    @Test
    void serviceLoaderFindsTheBuiltIns() {
        CoresetAlgorithms registry = CoresetAlgorithms.load();
        assertEquals(List.of("mean", "sensitivity", "uniform"), List.copyOf(registry.ids()));
        assertInstanceOf(UniformCoreset.class, registry.create("uniform", Map.of("m", "4")));
        assertInstanceOf(SensitivityCoreset.class, registry.create("sensitivity", Map.of("k", "2", "m", "4")));
        assertInstanceOf(MeanCoreset.class, registry.create("mean", Map.of("m", "4")));
        assertTrue(registry.toString().contains("uniform=" + BuiltInAlgorithms.class.getName()), registry.toString());
    }

    @Test
    void unknownIdListsTheAvailableOnes() {
        UnknownAlgorithmException e = assertThrows(UnknownAlgorithmException.class,
                () -> CoresetAlgorithms.load().create("kmeans", Map.of()));
        assertEquals("unknown coreset algorithm 'kmeans', available: [mean, sensitivity, uniform]", e.getMessage());
        assertEquals("kmeans", e.algorithmId());
        assertEquals(Set.of("mean", "sensitivity", "uniform"), e.availableIds());
    }

    @Test
    void badParametersAreTheProvidersErrorNotAnUnknownId() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> CoresetAlgorithms.load().create("uniform", Map.of("m", "4", "bogus", "1")));
        assertFalse(e instanceof UnknownAlgorithmException);
        assertTrue(e.getMessage().contains("unknown parameter(s) [bogus]"), e.getMessage());
    }

    @Test
    void conflictingProvidersFailWhenTheRegistryIsBuilt() {
        IllegalStateException dup = assertThrows(IllegalStateException.class,
                () -> CoresetAlgorithms.of(new BuiltInAlgorithms(), new ClaimsUniform()));
        assertEquals("algorithm id 'uniform' is provided by both " + BuiltInAlgorithms.class.getName() + " and "
                + ClaimsUniform.class.getName() + "; remove one of them from the class path", dup.getMessage());
        assertTrue(assertThrows(IllegalStateException.class, () -> CoresetAlgorithms.of(new FixedIds(Set.of())))
                .getMessage().endsWith("declares no algorithm ids"));
        assertTrue(assertThrows(IllegalStateException.class, () -> CoresetAlgorithms.of(new FixedIds(Set.of(" "))))
                .getMessage().endsWith("declares a blank algorithm id"));
    }

    @Test
    void aProviderReturningNullIsReported() {
        CoresetAlgorithmProvider broken = new CoresetAlgorithmProvider() {
            @Override
            public Set<String> ids() {
                return Set.of("fixed");
            }

            @Override
            public CoresetAlgorithm create(String id, Map<String, String> params) {
                return null;
            }
        };
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> CoresetAlgorithms.of(broken).create("fixed", Map.of()));
        assertTrue(e.getMessage().endsWith("returned null for algorithm 'fixed'"), e.getMessage());
    }

    @Test
    void providersInAnExtraJarAreDiscoveredAndCheckedForConflicts(@TempDir Path dir) throws IOException {
        Path services = Files.createDirectories(dir.resolve("META-INF/services"));
        Path file = services.resolve(CoresetAlgorithmProvider.class.getName());
        Files.writeString(file, FixedIds.class.getName() + "\n");
        try (URLClassLoader jar = new URLClassLoader(new URL[] {dir.toUri().toURL()}, getClass().getClassLoader())) {
            CoresetAlgorithms registry = CoresetAlgorithms.load(jar);
            assertEquals(List.of("fixed", "mean", "sensitivity", "uniform"), List.copyOf(registry.ids()));
            Coreset in = Coreset.of(List.of(new WeightedPoint(SparseVector.dense(new double[] {1, 2}), 3)));
            assertEquals(in, registry.create("fixed", Map.of()).compress(in, new java.util.SplittableRandom(1)));
        }
        Files.writeString(file, ClaimsUniform.class.getName() + "\n");
        try (URLClassLoader jar = new URLClassLoader(new URL[] {dir.toUri().toURL()}, getClass().getClassLoader())) {
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> CoresetAlgorithms.load(jar));
            assertTrue(e.getMessage().startsWith("algorithm id 'uniform' is provided by both"), e.getMessage());
        }
    }

    @Test
    void noProviderAtAllIsAStartupError() throws IOException {
        try (URLClassLoader empty = new URLClassLoader(new URL[0], ClassLoader.getPlatformClassLoader())) {
            IllegalStateException e = assertThrows(IllegalStateException.class, () -> CoresetAlgorithms.load(empty));
            assertTrue(e.getMessage().contains("META-INF/services/" + CoresetAlgorithmProvider.class.getName()),
                    e.getMessage());
        }
    }

    @Test
    void registeredAlgorithmIsASortedSerializableRecipe() throws Exception {
        Map<String, String> params = new TreeMap<>(Map.of("m", "8", "k", "2"));
        RegisteredAlgorithm recipe = new RegisteredAlgorithm("sensitivity", params);
        params.put("m", "9"); // copied, not aliased
        assertEquals(List.of("k", "m"), List.copyOf(recipe.params().keySet()));
        assertThrows(UnsupportedOperationException.class, () -> recipe.params().put("x", "1"));
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(recipe);
        }
        Object copy;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            copy = in.readObject();
        }
        assertEquals(recipe, copy);
        assertEquals("sensitivity(k=2, m=8, seeding=kmeans++(lloydIterations=0), bound=standard(alpha=2.0))",
                ((CoresetAlgorithmFactory) copy).create().toString());
        assertThrows(UnknownAlgorithmException.class, () -> new RegisteredAlgorithm("nope", Map.of()).create());
    }
}
