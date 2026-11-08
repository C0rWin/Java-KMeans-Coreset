package io.github.c0rwin.coreset.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.RegisteredAlgorithm;
import io.github.c0rwin.coreset.core.UnknownAlgorithmException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.Map;
import org.junit.jupiter.api.Test;

class CoresetConfigTest {

    @Test
    void registeredIdsAreResolvedAndValidatedWhenBuilding() {
        CoresetConfig c = CoresetConfig.builder("uniform", Map.of("m", "50"), 3).build();
        assertEquals(new RegisteredAlgorithm("uniform", Map.of("m", "50")), c.algorithmFactory());
        assertEquals(100, c.chunkSize());
        assertTrue(c.toString().startsWith("CoresetConfig{uniform{m=50}, seed=3"), c.toString());
        assertThrows(UnknownAlgorithmException.class, () -> CoresetConfig.builder("nope", Map.of("m", "5"), 3));
        assertThrows(IllegalArgumentException.class, () -> CoresetConfig.builder("uniform", Map.of("m", "0"), 3));
    }

    @Test
    void codeSuppliedFactoriesNeedAChunkSizeAndAreValidatedWhenBuilding() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> CoresetConfig.builder(new UniformRecipe(50), 3).build());
        assertTrue(e.getMessage().startsWith("set chunkSize"), e.getMessage());
        assertThrows(IllegalArgumentException.class, () -> CoresetConfig.builder(new UniformRecipe(0), 3));
        assertEquals(64, CoresetConfig.builder(new UniformRecipe(50), 3).chunkSize(64).build().chunkSize());
    }

    @Test
    void fingerprintFollowsTheAlgorithmIdentitySeedAndChunkSize() throws Exception {
        long a = CoresetConfig.builder(new UniformRecipe(50), 3).chunkSize(100).build().fingerprint();
        assertEquals(a, CoresetConfig.builder(new UniformRecipe(50), 3).chunkSize(100).build().fingerprint());
        assertNotEquals(a, CoresetConfig.builder(new UniformRecipe(51), 3).chunkSize(100).build().fingerprint());
        assertNotEquals(a, CoresetConfig.builder(new UniformRecipe(50), 4).chunkSize(100).build().fingerprint());
        assertNotEquals(a, CoresetConfig.builder(new UniformRecipe(50), 3).chunkSize(101).build().fingerprint());
        assertNotEquals(a, CoresetConfig.builder("uniform", Map.of("m", "50"), 3).build().fingerprint());

        CoresetConfig original = CoresetConfig.builder(new UniformRecipe(50), 3).chunkSize(100).build();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(original);
        }
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            CoresetConfig shipped = (CoresetConfig) in.readObject();
            assertEquals(a, shipped.fingerprint());
            assertEquals(original.algorithmFactory().identity(), shipped.algorithmFactory().identity());
        }
    }
}
