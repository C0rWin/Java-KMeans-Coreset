package io.github.c0rwin.coreset.testbed.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.testbed.data.Conversion.Variant;
import io.github.c0rwin.coreset.testbed.data.PointFile.Header;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.StringReader;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.zip.GZIPOutputStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ConversionTest {

    @TempDir
    Path dir;

    private static Path fixture(String name) throws URISyntaxException {
        return Path.of(ConversionTest.class.getResource("/fixtures/" + name).toURI());
    }

    private static List<double[]> denseRows(Path file) throws IOException {
        List<double[]> out = new ArrayList<>();
        PointFile.forEachRow(file, (Loaders.DenseRowSink) row -> out.add(row.clone()));
        return out;
    }

    private static List<SparseVector> vectors(PointSet ps) {
        return ps.asList().stream().map(WeightedPoint::vector).toList();
    }

    @Test
    void denseConversionMatchesTheTextLoaderExactly() throws Exception {
        Loaders.Options drop = new Loaders.Options(Loaders.Header.AUTO, List.of("id", "label"), 0);
        Path target = dir.resolve("h.cbin");
        try (BufferedReader r = Loaders.open(fixture("dense_header.csv"))) {
            Conversion.convertDense(r, drop, 2, target);
        }
        List<WeightedPoint> text = Loaders.load(fixture("dense_header.csv"), Loaders.Format.DENSE, drop);
        assertEquals(text.stream().map(WeightedPoint::vector).toList(), vectors(PointFile.readAll(target)));
    }

    @Test
    void gzippedCsvInsideAZipConvertsBitExactly() throws Exception {
        double[][] rows = PointFileTest.denseRows(500, 6, 7);
        StringBuilder csv = new StringBuilder();
        for (int i = 0; i < rows.length; i++) {
            csv.append("label").append(i % 3); // a symbolic column that is dropped
            for (double v : rows[i]) {
                csv.append(',').append(i % 2 == 0 ? Double.toString(v) : String.format(Locale.ROOT, "%.17e", v));
            }
            csv.append('\n');
        }
        Path zip = dir.resolve("src.zip");
        try (ZipOutputStream z = new ZipOutputStream(Files.newOutputStream(zip))) {
            z.putNextEntry(new ZipEntry("DATA.csv.gz"));
            OutputStream gz = new GZIPOutputStream(z);
            gz.write(csv.toString().getBytes(StandardCharsets.US_ASCII));
            ((GZIPOutputStream) gz).finish();
            z.closeEntry();
        }
        Path target = dir.resolve("z.cbin");
        try (BufferedReader r = Loaders.open(zip, "DATA.csv.gz")) {
            Header h = Conversion.convertDense(r, new Loaders.Options(Loaders.Header.ABSENT, List.of("0"), 0), 6,
                    target);
            assertEquals(500, h.n());
        }
        List<double[]> back = denseRows(target);
        for (int i = 0; i < rows.length; i++) {
            assertArrayEquals(rows[i], back.get(i), "row " + i);
        }
    }

    @Test
    void bagOfWordsConvertsToCsrWithEmptyDocuments() throws Exception {
        Path target = dir.resolve("bow.csr");
        try (BufferedReader r = Loaders.open(fixture("docword_small.txt"))) {
            Header h = Conversion.convertBagOfWords(r, target);
            assertEquals(new Header(PointFile.Layout.CSR, 1, 0, 4, 5, 6, h.checksum()), h);
        }
        assertEquals(Loaders.load(fixture("docword_small.txt"), Loaders.Format.BOW, Loaders.Options.defaults())
                .stream().map(WeightedPoint::vector).toList(), vectors(PointFile.readAll(target)));
    }

    @Test
    void zscoreStandardizesEveryFeatureAndZeroesConstantOnes() throws Exception {
        double[][] rows = {{1, 5, 0}, {2, 5, -3}, {4, 5, 9}, {9, 5, 0.5}};
        Path raw = dir.resolve("r.cbin");
        PointFileTest.writeDense(raw, rows, 3);
        Path z = dir.resolve("r.zscore.cbin");
        Header h = Conversion.derive(raw, z, Variant.ZSCORE);
        assertEquals(PointFile.FLAG_ZSCORE, h.flags());
        List<double[]> out = denseRows(z);
        for (int c = 0; c < 3; c++) {
            double mean = 0.0;
            for (double[] row : rows) {
                mean += row[c];
            }
            mean /= rows.length;
            double ss = 0.0;
            for (double[] row : rows) {
                ss += (row[c] - mean) * (row[c] - mean);
            }
            double std = Math.sqrt(ss / rows.length);
            for (int i = 0; i < rows.length; i++) {
                assertEquals(std > 0 ? (rows[i][c] - mean) / std : 0.0, out.get(i)[c], 0.0, "row " + i + " col " + c);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> Conversion.derive(raw, dir.resolve("t.csr"),
                Variant.TFIDF));
    }

    @Test
    void tfidfDropsWordsInEveryDocumentAndNormalisesRows() throws Exception {
        int[][] idx = {{0, 1, 2}, {0, 2}, {0}, {0, 3}};
        double[][] val = {{1, 2, 1}, {3, 1}, {2}, {1, 5}};
        Path raw = dir.resolve("d.csr");
        try (PointFileWriter w = PointFileWriter.csr(raw, 4, 4, 8, 0)) {
            for (int i = 0; i < 4; i++) {
                w.writeRow(idx[i], val[i], 0, idx[i].length);
            }
            w.finish();
        }
        Path t = dir.resolve("d.tfidf.csr");
        Header h = Conversion.derive(raw, t, Variant.TFIDF);
        assertEquals(PointFile.FLAG_TFIDF, h.flags());
        assertEquals(4, h.nnz()); // word 0 (df = N) is gone
        double a = 2 * Math.log(4.0);
        double b = Math.log(2.0);
        double norm0 = Math.sqrt(a * a + b * b);
        List<SparseVector> expected = List.of(
                SparseVector.of(4, new int[] {1, 2}, new double[] {a / norm0, b / norm0}),
                SparseVector.of(4, new int[] {2}, new double[] {1.0}),
                SparseVector.of(4, new int[0], new double[0]),
                SparseVector.of(4, new int[] {3}, new double[] {1.0}));
        List<SparseVector> actual = vectors(PointFile.readAll(t));
        for (int i = 0; i < 4; i++) {
            assertEquals(expected.get(i).nnz(), actual.get(i).nnz(), "row " + i);
            for (int c = 0; c < 4; c++) {
                assertEquals(expected.get(i).get(c), actual.get(i).get(c), 1e-15, "row " + i + " col " + c);
            }
        }
    }

    private PublicDataset fake(int n) throws IOException {
        Path cache = dir.resolve("cache");
        Path csv = cache.resolve("fake").resolve("data.csv");
        Files.createDirectories(csv.getParent());
        Files.writeString(csv, "a,b,label\n1,2,x\n0,4,y\n5,0,x\n");
        return new PublicDataset("fake", "https://example.invalid/fake/data.csv", PublicDataset.sha256(csv),
                Files.size(csv), "test", "https://example.invalid", Loaders.Format.DENSE, "", Loaders.Header.PRESENT,
                List.of("label"), n, 2, "none", "drops label");
    }

    @Test
    void ensureConvertsOnceFromTheCachedDownloadAndDerivesVariants() throws Exception {
        PublicDataset pd = fake(3);
        Path cache = dir.resolve("cache");
        Path raw = Conversion.ensure(pd, cache, Variant.RAW, false);
        assertEquals(cache.resolve("fake").resolve("fake.cbin"), raw);
        FileTime first = FileTime.fromMillis(1_000_000L);
        Files.setLastModifiedTime(raw, first);
        assertEquals(raw, Conversion.ensure(pd, cache, Variant.RAW, false));
        assertEquals(first, Files.getLastModifiedTime(raw)); // reused, not rewritten
        Path z = Conversion.ensure(pd, cache, Variant.ZSCORE, false);
        assertEquals(cache.resolve("fake").resolve("fake.zscore.cbin"), z);
        assertEquals(List.of(SparseVector.dense(new double[] {1, 2}), SparseVector.dense(new double[] {0, 4}),
                SparseVector.dense(new double[] {5, 0})),
                vectors(PointFile.readAll(raw)));
        assertThrows(IllegalArgumentException.class, () -> Conversion.binaryFile(pd, cache, Variant.TFIDF));

        // Loading through the binary gives the same moments as the text path.
        Dataset viaBinary = Dataset.of("fake", PointFile.readAll(raw), "", "test", "", List.of(), Optional.empty());
        Dataset viaText = Dataset.of("fake", Loaders.load(pd.cachedFile(cache), Loaders.Format.DENSE,
                pd.loaderOptions()), "", "test", "", List.of(), Optional.empty());
        assertEquals(viaText.moments(), viaBinary.moments());
        assertEquals(viaText.nnz(), viaBinary.nnz());
        assertEquals(viaText.weightSum(), viaBinary.weightSum());
    }

    @Test
    void ensureRejectsAShapeMismatchAndMissingDownloads() throws Exception {
        PublicDataset wrong = fake(4);
        Path cache = dir.resolve("cache");
        IllegalStateException shape = assertThrows(IllegalStateException.class,
                () -> Conversion.ensure(wrong, cache, Variant.RAW, false));
        assertTrue(shape.getMessage().contains("3 x 2, descriptor says 4 x 2"), shape.getMessage());
        assertFalse(Files.exists(Conversion.binaryFile(wrong, cache, Variant.RAW)));

        IllegalStateException missing = assertThrows(IllegalStateException.class,
                () -> Conversion.ensure(wrong, dir.resolve("empty"), Variant.ZSCORE, false));
        assertTrue(missing.getMessage().contains("--allow-download"), missing.getMessage());
    }

    @Test
    void variantsParseByName() {
        assertEquals(Variant.TFIDF, Variant.of(" TfIdf "));
        assertEquals("zscore", Variant.ZSCORE.id());
        assertEquals(List.of("raw", "zscore", "tfidf"), Arrays.stream(Variant.values()).map(Variant::id).toList());
        assertThrows(IllegalArgumentException.class, () -> Loaders.read(new BufferedReader(new StringReader("")),
                Loaders.Format.of("bow"), Loaders.Options.defaults()));
    }
}
