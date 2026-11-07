package io.github.c0rwin.coreset.node;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.core.algo.MeanCoreset;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import java.net.URL;
import java.net.URLClassLoader;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ServiceLoaderAlgorithmResolverTest {

    @Test
    void routesEachIdToTheProviderThatOwnsItAndFallsThroughUnknownIds() {
        ServiceLoaderAlgorithmResolver r = ServiceLoaderAlgorithmResolver.load(getClass().getClassLoader());
        assertEquals(2, r.providers().size());
        assertSame(TestAlgorithms.IDENTITY, r.resolve("identity", Map.of()));
        assertTrue(r.resolve("mean", Map.of("m", "4")) instanceof MeanCoreset);
        assertTrue(r.resolve("uniform", Map.of("m", "4")) instanceof UniformCoreset);
        UnknownAlgorithmException e = assertThrows(UnknownAlgorithmException.class, () -> r.resolve("x", Map.of()));
        assertEquals("x", e.algorithmId());
    }

    @Test
    void coreProviderRejectsBadParametersInsteadOfFallingThrough() {
        CoreAlgorithmsResolver core = new CoreAlgorithmsResolver();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> core.resolve("uniform", Map.of("m", "4", "bogus", "1")));
        assertTrue(!(e instanceof UnknownAlgorithmException));
    }

    @Test
    void failsLoudlyWithoutProviders() {
        ClassLoader empty = new URLClassLoader(new URL[0], ClassLoader.getPlatformClassLoader());
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> ServiceLoaderAlgorithmResolver.load(empty));
        assertTrue(e.getMessage().contains("META-INF/services/io.github.c0rwin.coreset.node.AlgorithmResolver"));
    }
}
