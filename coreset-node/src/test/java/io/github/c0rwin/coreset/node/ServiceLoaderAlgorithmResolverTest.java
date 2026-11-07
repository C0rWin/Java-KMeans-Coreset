package io.github.c0rwin.coreset.node;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.node.testing.TestResolver;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ServiceLoaderAlgorithmResolverTest {

    @Test
    void findsRegisteredProvidersAndFallsThroughUnknownIds() {
        ServiceLoaderAlgorithmResolver r = ServiceLoaderAlgorithmResolver.load(getClass().getClassLoader());
        assertEquals(1, r.providers().size());
        assertTrue(r.providers().get(0) instanceof TestResolver);
        assertSame(TestAlgorithms.IDENTITY, r.resolve("identity", Map.of()));
        UnknownAlgorithmException e = assertThrows(UnknownAlgorithmException.class, () -> r.resolve("x", Map.of()));
        assertEquals("x", e.algorithmId());
    }

    @Test
    void failsLoudlyWithoutProviders() {
        ClassLoader empty = new URLClassLoader(new URL[0], ClassLoader.getPlatformClassLoader());
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ServiceLoaderAlgorithmResolver.load(empty));
        assertTrue(e.getMessage().contains("META-INF/services/io.github.c0rwin.coreset.node.AlgorithmResolver"));
    }
}
