package io.github.c0rwin.coreset.experiments.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.DatasetConfig;
import io.github.c0rwin.coreset.testbed.data.PointFile;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PointSourcesTest {

    @TempDir
    Path dir;

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
    void generatedDataIsWrittenOnceAsPointFileAndReadByShard() throws IOException {
        DatasetConfig spec = new DatasetConfig("gm", "gaussian-mixture",
                Map.of("n", "500", "d", "4", "clusters", "3", "sigma", "1.5", "seed", "9"));
        PointSource a = PointSources.open(spec, dir);
        Path file = Path.of((String) a.provenance().get("file"));
        assertTrue(file.toString().endsWith(".cbin"));
        long written = Files.getLastModifiedTime(file).toMillis();
        PointSource b = PointSources.open(spec, dir);
        assertEquals(written, Files.getLastModifiedTime(file).toMillis(), "materialised once");
        assertEquals(a.provenance().get("checksum"), b.provenance().get("checksum"));
        assertEquals(500, a.size());
        assertEquals(4, a.dim());
        PointBlock all = PointFile.open(file, 1000).next();
        long rows = 0;
        for (int s = 0; s < 3; s++) {
            long from = PointSource.shardStart(500, s, 3);
            try (ChunkReader r = a.open(from, PointSource.shardStart(500, s + 1, 3), 64)) {
                for (PointBlock blk = r.next(); blk != null; blk = r.next()) {
                    assertTrue(blk.size() <= 64);
                    for (int i = 0; i < blk.size(); i++, rows++) {
                        assertEquals(all.vector((int) rows), blk.vector(i));
                    }
                }
                assertNull(r.next());
            }
        }
        assertEquals(500, rows);
    }

    @Test
    void sparseTestbedDataBecomesCsr() throws IOException {
        DatasetConfig spec = new DatasetConfig("bow", "testbed", Map.of("testbedType", "sparse-bow", "n", "300",
                "d", "400", "topics", "4", "docLength", "20"));
        PointSource src = PointSources.open(spec, dir);
        assertTrue(((String) src.provenance().get("file")).endsWith(".csr"));
        assertEquals(400, src.dim());
        assertEquals(300, src.size());
    }

    @Test
    void publicDatasetsAreReadFromTheTestbedCacheAndNeverConvertedHere() {
        DatasetConfig spec = new DatasetConfig("cov", "public",
                Map.of("name", "covertype", "variant", "zscore", "cacheDir", dir.toString()));
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> PointSources.open(spec, dir));
        assertTrue(e.getMessage().contains("datasets convert covertype --variant zscore"), e.getMessage());
    }
}
