package io.github.c0rwin.coreset.experiments.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.DatasetConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class DenseFileSourceTest {

    @TempDir
    Path dir;

    @Test
    void rowsRoundTripInAnyRangeAndChunking() throws IOException {
        Path f = dir.resolve("p.f64");
        try (DenseFileSource.Writer w = new DenseFileSource.Writer(f, 3, true)) {
            for (int i = 0; i < 10; i++) {
                w.add(new double[] {i, 0.0, -i * 0.5}, 1.0 + i);
            }
        }
        DenseFileSource src = DenseFileSource.open("p", f, Map.of());
        assertEquals(10, src.size());
        assertEquals(3, src.dim());
        try (ChunkReader r = src.open(3, 8)) {
            PointBlock a = r.next(4);
            PointBlock b = r.next(4);
            assertNull(r.next(4));
            assertEquals(4, a.size());
            assertEquals(1, b.size());
            assertEquals(3.0, a.vector(0).get(0));
            assertEquals(0.0, a.vector(0).get(1));
            assertEquals(2, a.nnz(0), "the explicit zero is not stored");
            assertEquals(-3.5, b.vector(0).get(2));
            assertEquals(8.0, b.weight(0));
        }
        assertThrows(IllegalArgumentException.class, () -> src.open(5, 11));
    }

    @Test
    void shardsPartitionAllRows() {
        for (long n : new long[] {1, 7, 1_000_003}) {
            for (int s : new int[] {1, 2, 3, 8}) {
                assertEquals(0, PointSource.shardStart(n, 0, s));
                assertEquals(n, PointSource.shardStart(n, s, s));
                for (int i = 0; i < s; i++) {
                    long size = PointSource.shardStart(n, i + 1, s) - PointSource.shardStart(n, i, s);
                    assertEquals(n / s + (i < n % s ? 1 : 0), size);
                }
            }
        }
    }

    @Test
    void generatedDatasetsAreMaterialisedOnceAndRejectTruncation() throws IOException {
        DatasetConfig spec = new DatasetConfig("gm", "gaussian-mixture",
                Map.of("n", "500", "d", "4", "clusters", "3", "sigma", "1.5", "seed", "9"));
        PointSource a = PointSources.open(spec, dir);
        PointSource b = PointSources.open(spec, dir);
        assertEquals(500, a.size());
        assertEquals(a.provenance().get("sha256"), b.provenance().get("sha256"));
        Path file = Path.of((String) a.provenance().get("file"));
        Files.write(file, java.util.Arrays.copyOf(Files.readAllBytes(file), (int) Files.size(file) - 8));
        assertThrows(IOException.class, () -> DenseFileSource.open("gm", file, Map.of()));
    }
}
