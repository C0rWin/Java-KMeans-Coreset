package io.github.c0rwin.coreset.testbed.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.WeightedPoint;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class SyntheticDataTest {

    @Test
    void zipfSizesSumToNKeepEveryClusterAndDecrease() {
        int[] balanced = SyntheticData.zipfSizes(100, 4, 0.0);
        assertArrayEquals(new int[] {25, 25, 25, 25}, balanced);
        int[] skewed = SyntheticData.zipfSizes(1000, 10, 2.0);
        assertEquals(1000, Arrays.stream(skewed).sum());
        for (int i = 1; i < skewed.length; i++) {
            assertTrue(skewed[i] <= skewed[i - 1] && skewed[i] >= 1, Arrays.toString(skewed));
        }
        assertTrue(skewed[0] > 50 * skewed[9] / 10, Arrays.toString(skewed)); // ~1/i^2
        assertEquals(10, Arrays.stream(SyntheticData.zipfSizes(10, 10, 5.0)).sum());
        assertThrows(IllegalArgumentException.class, () -> SyntheticData.zipfSizes(3, 4, 0.0));
    }

    @Test
    void gaussianMixtureIsSeededAndShaped() {
        List<WeightedPoint> a = SyntheticData.gaussianMixture(500, 3, 5, 1.0, 1.0, 100.0, 9);
        List<WeightedPoint> b = SyntheticData.gaussianMixture(500, 3, 5, 1.0, 1.0, 100.0, 9);
        List<WeightedPoint> c = SyntheticData.gaussianMixture(500, 3, 5, 1.0, 1.0, 100.0, 10);
        assertEquals(500, a.size());
        assertEquals(a, b);
        assertNotEquals(a, c);
        assertEquals(3, a.get(0).vector().dim());
    }

    @Test
    void sparseBagOfWordsIsUnitNormSparseAndSeeded() {
        List<WeightedPoint> docs = SyntheticData.sparseBagOfWords(300, 2000, 6, 100, 30, 0.8, 0.0, 3);
        assertEquals(docs, SyntheticData.sparseBagOfWords(300, 2000, 6, 100, 30, 0.8, 0.0, 3));
        long nnz = 0;
        for (WeightedPoint p : docs) {
            assertEquals(2000, p.vector().dim());
            assertEquals(1.0, p.vector().squaredNorm(), 1e-12);
            assertTrue(p.vector().nnz() >= 1 && p.vector().nnz() <= 46, "nnz " + p.vector().nnz());
            nnz += p.vector().nnz();
        }
        assertTrue(nnz / 300.0 > 10 && nnz / 300.0 < 40, "mean nnz " + nnz / 300.0);
    }

    @Test
    void registryBuildsEachSyntheticTypeAndRejectsUnknownParameters() {
        Datasets registry = new Datasets(Path.of("unused"), false);
        Dataset g = registry.load(new DatasetSpec("g", "gaussian-mixture",
                Map.of("n", "1e3", "d", "4", "clusters", "3", "sigma", "1")));
        assertEquals(1000, g.n());
        assertEquals(4, g.dim());
        assertEquals(1000.0, g.weightSum());
        assertTrue(g.benchmark().isEmpty());
        Dataset e = registry.load(new DatasetSpec("e", "esa22", Map.of("k", "3", "alpha", "3")));
        assertEquals(27, e.n());
        assertTrue(e.benchmark().isPresent());
        Dataset s = registry.load(new DatasetSpec("s", "sparse-bow",
                Map.of("n", "50", "d", "500", "topics", "5", "docLength", "10")));
        assertEquals(500, s.dim());
        IllegalArgumentException bad = assertThrows(IllegalArgumentException.class,
                () -> registry.load(new DatasetSpec("x", "esa22", Map.of("k", "3", "alpha", "3", "n", "9"))));
        assertTrue(bad.getMessage().contains("unknown parameter(s) [n]"), bad.getMessage());
        IllegalStateException uncached = assertThrows(IllegalStateException.class,
                () -> registry.load(new DatasetSpec(null, "public", Map.of("name", "covertype"))));
        assertTrue(uncached.getMessage().contains("--allow-download"), uncached.getMessage());
    }
}
