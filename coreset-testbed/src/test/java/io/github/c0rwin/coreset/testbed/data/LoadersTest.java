package io.github.c0rwin.coreset.testbed.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.StringReader;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LoadersTest {

    private static Path fixture(String name) throws URISyntaxException {
        return Path.of(LoadersTest.class.getResource("/fixtures/" + name).toURI());
    }

    private static SparseVector v(double... c) {
        return SparseVector.dense(c);
    }

    private static List<SparseVector> vectors(List<WeightedPoint> pts) {
        for (WeightedPoint p : pts) {
            assertEquals(1.0, p.weight());
        }
        return pts.stream().map(WeightedPoint::vector).toList();
    }

    @Test
    void denseCsvWithHeaderDropsNamedColumnsAndSkipsCommentsAndBlankLines() throws Exception {
        List<WeightedPoint> pts = Loaders.load(fixture("dense_header.csv"), Loaders.Format.DENSE,
                new Loaders.Options(Loaders.Header.AUTO, List.of("id", "label"), 0));
        assertEquals(List.of(v(0.5, 1.5), v(-2, 0.3), v(4, 0)), vectors(pts));
    }

    @Test
    void whitespaceFileWithoutHeaderAndItsGzipCopyAgree() throws Exception {
        List<SparseVector> plain = vectors(Loaders.load(fixture("dense_ws.txt"), Loaders.Format.DENSE,
                Loaders.Options.defaults()));
        assertEquals(List.of(v(1, 2), v(3, 4), v(5, 6)), plain);
        assertEquals(plain, vectors(Loaders.load(fixture("dense_ws.txt.gz"), Loaders.Format.DENSE,
                Loaders.Options.defaults())));
        // Dropping by index when there is no header.
        assertEquals(List.of(v(2), v(4), v(6)), vectors(Loaders.load(fixture("dense_ws.txt"), Loaders.Format.DENSE,
                new Loaders.Options(Loaders.Header.ABSENT, List.of("0"), 0))));
    }

    @Test
    void libsvmIsOneBasedWithOptionalLabelAndInferredOrExplicitDim() throws Exception {
        List<SparseVector> inferred = vectors(Loaders.load(fixture("sparse.libsvm"), Loaders.Format.LIBSVM,
                Loaders.Options.defaults()));
        assertEquals(List.of(
                SparseVector.of(4, new int[] {0, 2}, new double[] {0.5, 2}),
                SparseVector.of(4, new int[] {1}, new double[] {1.5}),
                SparseVector.of(4, new int[] {3}, new double[] {1})), inferred);
        List<SparseVector> wide = vectors(Loaders.load(fixture("sparse.libsvm"), Loaders.Format.LIBSVM,
                new Loaders.Options(Loaders.Header.AUTO, List.of(), 10)));
        assertEquals(10, wide.get(0).dim());
        assertEquals(2.0, wide.get(0).get(2));
        IllegalArgumentException tooSmall = assertThrows(IllegalArgumentException.class,
                () -> Loaders.load(fixture("sparse.libsvm"), Loaders.Format.LIBSVM,
                        new Loaders.Options(Loaders.Header.AUTO, List.of(), 3)));
        assertTrue(tooSmall.getMessage().contains("exceeds dim 3"), tooSmall.getMessage());
    }

    @Test
    void legacyWikiFormatIsIdDimThenZeroBasedPairs() throws Exception {
        List<SparseVector> docs = vectors(Loaders.load(fixture("docs.wiki"), Loaders.Format.WIKI,
                Loaders.Options.defaults()));
        assertEquals(List.of(
                SparseVector.of(5, new int[] {0, 4}, new double[] {1.5, 2}),
                SparseVector.of(5, new int[] {2}, new double[] {-1}),
                SparseVector.of(5, new int[0], new double[0])), docs);
    }

    @Test
    void bagOfWordsKeepsEmptyDocumentsAsRowsAndSortsWords() throws Exception {
        List<SparseVector> docs = vectors(Loaders.load(fixture("docword_small.txt"), Loaders.Format.BOW,
                Loaders.Options.defaults()));
        assertEquals(List.of(
                SparseVector.of(5, new int[] {1, 3}, new double[] {3, 1}),
                SparseVector.of(5, new int[0], new double[0]),
                SparseVector.of(5, new int[] {0, 4}, new double[] {1, 2}),
                SparseVector.of(5, new int[] {1, 2}, new double[] {1, 4})), docs);
    }

    @Test
    void zipMembersAreOpenedByNameAndGunzipped(@TempDir Path dir) throws Exception {
        Path zip = dir.resolve("a.zip");
        try (ZipOutputStream z = new ZipOutputStream(Files.newOutputStream(zip))) {
            z.putNextEntry(new ZipEntry("inner/data.csv.gz"));
            ByteArrayOutputStream gz = new ByteArrayOutputStream();
            try (GZIPOutputStream g = new GZIPOutputStream(gz)) {
                g.write("1,2\n3,4\n".getBytes(StandardCharsets.UTF_8));
            }
            z.write(gz.toByteArray());
            z.closeEntry();
        }
        try (BufferedReader r = Loaders.open(zip, "inner/data.csv.gz")) {
            assertEquals(List.of(v(1, 2), v(3, 4)), vectors(Loaders.read(r, Loaders.Format.DENSE,
                    Loaders.Options.defaults())));
        }
        IOException missing = assertThrows(IOException.class, () -> Loaders.open(zip, "nope"));
        assertTrue(missing.getMessage().contains("no zip entry 'nope'"), missing.getMessage());
    }

    private static IllegalArgumentException failure(Loaders.Format format, String text) {
        return assertThrows(IllegalArgumentException.class,
                () -> Loaders.read(new BufferedReader(new StringReader(text)), format, Loaders.Options.defaults()));
    }

    @Test
    void malformedInputFailsWithTheLineNumber() {
        assertTrue(failure(Loaders.Format.DENSE, "1 2\n3\n").getMessage().startsWith("line 2: expected 2 columns"));
        assertTrue(failure(Loaders.Format.DENSE, "1 2\n3 x\n").getMessage().startsWith("line 2: not a number"));
        assertTrue(failure(Loaders.Format.DENSE, "1 NaN\n").getMessage().contains("line 1"));
        assertTrue(failure(Loaders.Format.LIBSVM, "1 0:2\n").getMessage().contains("1-based"));
        assertTrue(failure(Loaders.Format.LIBSVM, "1 2:2 2:3\n").getMessage().contains("duplicate index 1"));
        assertTrue(failure(Loaders.Format.WIKI, "1 4 0 1\n2 5 0 1\n").getMessage().startsWith("line 2: dim 5"));
        assertTrue(failure(Loaders.Format.WIKI, "1 4 4 1\n").getMessage().contains("outside [0, 4)"));
        assertTrue(failure(Loaders.Format.WIKI, "1 4 0\n").getMessage().startsWith("line 1: expected"));
        assertTrue(failure(Loaders.Format.BOW, "2\n3\n2\n1 1 1\n").getMessage().contains("header says 2 entries"));
        assertTrue(failure(Loaders.Format.BOW, "2\n3\n2\n2 1 1\n1 1 1\n").getMessage()
                .startsWith("line 5: document 1 out of order"));
        assertTrue(failure(Loaders.Format.BOW, "1\n3\n2\n1 2 1\n1 2 5\n").getMessage().contains("duplicate word 2"));
        assertTrue(failure(Loaders.Format.BOW, "1\n3\n1\n1 4 1\n").getMessage().contains("word 4"));
    }
}
