package io.github.c0rwin.coreset.testbed.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.testbed.data.PointFile.Header;
import io.github.c0rwin.coreset.testbed.data.PointFile.Layout;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PointFileTest {

    @TempDir
    Path dir;

    /** Rows with about a third zeros and magnitudes from 1e-300 to 1e300. */
    static double[][] denseRows(int n, int d, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] rows = new double[n][d];
        for (double[] row : rows) {
            for (int c = 0; c < d; c++) {
                row[c] = rnd.nextInt(3) == 0 ? 0.0 : rnd.nextGaussian() * Math.pow(10, rnd.nextInt(-300, 301));
            }
        }
        return rows;
    }

    static Header writeDense(Path file, double[][] rows, int d) throws IOException {
        try (PointFileWriter w = PointFileWriter.dense(file, d, 0)) {
            for (double[] row : rows) {
                w.writeRow(row);
            }
            return w.finish();
        }
    }

    private static List<double[]> collectDense(Path file) throws IOException {
        List<double[]> out = new ArrayList<>();
        PointFile.forEachRow(file, (Loaders.DenseRowSink) row -> out.add(row.clone()));
        return out;
    }

    @Test
    void denseRoundTripIsBitExactInEveryChunking() throws IOException {
        double[][] rows = denseRows(103, 7, 1);
        rows[5][2] = Double.MIN_VALUE;
        rows[6][3] = -Double.MAX_VALUE;
        Path file = dir.resolve("a.cbin");
        Header h = writeDense(file, rows, 7);
        long nonZero = Arrays.stream(rows).flatMapToDouble(Arrays::stream).filter(v -> v != 0.0).count();
        assertEquals(new Header(Layout.DENSE, 1, 0, 103, 7, nonZero, h.checksum()), h);
        assertEquals(h.fileBytes(), Files.size(file));
        assertEquals(h, PointFile.verify(file));
        assertEquals(h, PointFile.readHeader(file));

        List<double[]> streamed = collectDense(file);
        for (int i = 0; i < rows.length; i++) {
            assertArrayEquals(rows[i], streamed.get(i), "row " + i);
        }
        for (int chunk : new int[] {1, 10, 103, 1000}) {
            try (PointFile.Reader r = PointFile.open(file, chunk)) {
                int row = 0;
                while (r.hasNext()) {
                    PointBlock b = r.next();
                    assertEquals(Math.min(chunk, 103 - row), b.size());
                    for (int i = 0; i < b.size(); i++, row++) {
                        assertEquals(SparseVector.dense(rows[row]), b.vector(i), "row " + row);
                        assertEquals(1.0, b.weight(i));
                        for (int e = b.rowStart(i); e < b.rowEnd(i); e++) {
                            assertEquals(Double.doubleToRawLongBits(rows[row][b.index(e)]),
                                    Double.doubleToRawLongBits(b.value(e)));
                        }
                    }
                }
                assertEquals(103, row);
                assertEquals(103, r.position());
            }
        }
        PointSet all = PointFile.readAll(file);
        assertEquals(103, all.size());
        assertEquals(nonZero, all.storedEntries());
        for (int i = 0; i < rows.length; i++) {
            assertEquals(SparseVector.dense(rows[i]), all.vector(i));
        }
    }

    @Test
    void sparseRoundTripKeepsEmptyRowsAndAlignsTheValues() throws IOException {
        int[][] idx = {{0, 3, 49}, {}, {7}, {1, 2, 3, 4, 5}, {}, {48, 49}, {10}, {}, {0}};
        long nnz = Arrays.stream(idx).mapToLong(a -> a.length).sum(); // 13: odd, so the index section is padded
        double[][] val = new double[idx.length][];
        SplittableRandom rnd = new SplittableRandom(2);
        for (int i = 0; i < idx.length; i++) {
            val[i] = new double[idx[i].length];
            for (int e = 0; e < val[i].length; e++) {
                val[i][e] = rnd.nextDouble(-1e9, 1e9);
            }
        }
        Path file = dir.resolve("a.csr");
        Header h;
        try (PointFileWriter w = PointFileWriter.csr(file, idx.length, 50, nnz, PointFile.FLAG_TFIDF)) {
            for (int i = 0; i < idx.length; i++) {
                // Offset slices exercise from/to.
                int[] pi = new int[idx[i].length + 2];
                double[] pv = new double[pi.length];
                System.arraycopy(idx[i], 0, pi, 1, idx[i].length);
                System.arraycopy(val[i], 0, pv, 1, val[i].length);
                w.writeRow(pi, pv, 1, 1 + idx[i].length);
            }
            h = w.finish();
        }
        assertEquals(new Header(Layout.CSR, 1, PointFile.FLAG_TFIDF, 9, 50, 13, h.checksum()), h);
        assertEquals(0, h.valuesOffset() % 8);
        assertEquals(h.fileBytes(), Files.size(file));
        PointFile.verify(file);

        for (int chunk : new int[] {1, 4, 100}) {
            try (PointFile.Reader r = PointFile.open(file, chunk)) {
                int row = 0;
                while (r.hasNext()) {
                    PointBlock b = r.next();
                    for (int i = 0; i < b.size(); i++, row++) {
                        assertEquals(SparseVector.of(50, idx[row], val[row]), b.vector(i), "row " + row);
                    }
                }
                assertEquals(9, row);
            }
        }
        List<SparseVector> streamed = new ArrayList<>();
        PointFile.forEachRow(file, (Loaders.SparseRowSink) (i, v, from, to) -> streamed.add(
                SparseVector.of(50, Arrays.copyOfRange(i, from, to), Arrays.copyOfRange(v, from, to))));
        for (int row = 0; row < idx.length; row++) {
            assertEquals(SparseVector.of(50, idx[row], val[row]), streamed.get(row));
        }
        assertThrows(IllegalArgumentException.class,
                () -> PointFile.forEachRow(file, (Loaders.DenseRowSink) row -> { }));
    }

    @Test
    void corruptionTruncationAndForeignFilesAreDetected() throws IOException {
        Path file = dir.resolve("a.cbin");
        writeDense(file, denseRows(20, 3, 3), 3);
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.WRITE)) {
            ch.write(ByteBuffer.wrap(new byte[] {42}), PointFile.HEADER_BYTES + 17);
        }
        PointFile.readHeader(file); // the header alone cannot tell
        IOException crc = assertThrows(IOException.class, () -> PointFile.verify(file));
        assertTrue(crc.getMessage().contains("CRC32C"), crc.getMessage());

        Path cut = dir.resolve("cut.cbin");
        byte[] bytes = Files.readAllBytes(file);
        Files.write(cut, Arrays.copyOf(bytes, bytes.length - 8));
        IOException truncated = assertThrows(IOException.class, () -> PointFile.open(cut, 4));
        assertTrue(truncated.getMessage().contains("header implies"), truncated.getMessage());

        Path foreign = dir.resolve("x.cbin");
        bytes[0] = 'X';
        Files.write(foreign, bytes);
        IOException magic = assertThrows(IOException.class, () -> PointFile.readHeader(foreign));
        assertTrue(magic.getMessage().contains("not a point file"), magic.getMessage());
    }

    @Test
    void writerRejectsInvalidRowsAndLeavesNothingBehind() throws IOException {
        Path file = dir.resolve("b.csr");
        try (PointFileWriter w = PointFileWriter.csr(file, 2, 5, 2, 0)) {
            assertThrows(IllegalArgumentException.class, () -> w.writeRow(new int[] {2, 1}, new double[] {1, 1}, 0, 2));
            assertThrows(IllegalArgumentException.class, () -> w.writeRow(new int[] {5}, new double[] {1}, 0, 1));
            assertThrows(IllegalArgumentException.class, () -> w.writeRow(new int[] {1}, new double[] {0}, 0, 1));
            assertThrows(IllegalStateException.class, () -> w.writeRow(new double[5]));
            w.writeRow(new int[] {1, 2}, new double[] {1, 2}, 0, 2);
            assertThrows(IllegalStateException.class, () -> w.writeRow(new int[] {3}, new double[] {1}, 0, 1));
            assertThrows(IllegalStateException.class, w::finish); // one row short
        }
        assertFalse(Files.exists(file));
        assertFalse(Files.exists(dir.resolve("b.csr.part")));

        Path dense = dir.resolve("b.cbin");
        try (PointFileWriter w = PointFileWriter.dense(dense, 2, 0)) {
            assertThrows(IllegalArgumentException.class, () -> w.writeRow(new double[3]));
            assertThrows(IllegalArgumentException.class, () -> w.writeRow(new double[] {1, Double.NaN}));
        }
        assertFalse(Files.exists(dense));
    }

    @Test
    void emptyFilesRoundTrip() throws IOException {
        Path file = dir.resolve("e.cbin");
        Header h = writeDense(file, new double[0][], 4);
        assertEquals(0, h.n());
        assertEquals(PointFile.HEADER_BYTES, Files.size(file));
        assertEquals(PointSet.EMPTY, PointFile.readAll(file));
        Path csr = dir.resolve("e.csr");
        try (PointFileWriter w = PointFileWriter.csr(csr, 3, 4, 0, 0)) {
            for (int i = 0; i < 3; i++) {
                w.writeRow(new int[0], new double[0], 0, 0);
            }
            w.finish();
        }
        PointSet rows = PointFile.readAll(csr);
        assertEquals(3, rows.size());
        assertEquals(0, rows.storedEntries());
        PointFile.verify(csr);
    }

    @Test
    void rangedReadersReturnExactlyTheirRowsInEveryChunking() throws IOException {
        int n = 37;
        double[][] rows = denseRows(n, 5, 21);
        Path dense = dir.resolve("r.cbin");
        writeDense(dense, rows, 5);
        SplittableRandom rnd = new SplittableRandom(22);
        Path csr = dir.resolve("r.csr");
        int[][] idx = new int[n][];
        long nnz = 0;
        for (int i = 0; i < n; i++) {
            idx[i] = rnd.ints(0, 50).distinct().limit(rnd.nextInt(4)).sorted().toArray(); // some rows empty
            nnz += idx[i].length;
        }
        try (PointFileWriter w = PointFileWriter.csr(csr, n, 50, nnz, 0)) {
            for (int i = 0; i < n; i++) {
                double[] val = new double[idx[i].length];
                Arrays.fill(val, i + 1.0);
                w.writeRow(idx[i], val, 0, val.length);
            }
            w.finish();
        }
        PointSet allDense = PointFile.readAll(dense);
        PointSet allCsr = PointFile.readAll(csr);
        int[][] ranges = {{0, n}, {0, 0}, {n, n}, {5, 6}, {7, 30}, {36, 37}, {12, 37}};
        for (Path file : List.of(dense, csr)) {
            PointSet all = file == dense ? allDense : allCsr;
            for (int[] range : ranges) {
                for (int chunk : new int[] {1, 4, 64}) {
                    List<SparseVector> got = new ArrayList<>();
                    try (PointFile.Reader r = PointFile.open(file, chunk, range[0], range[1])) {
                        assertEquals(range[0], r.position());
                        while (r.hasNext()) {
                            PointBlock b = r.next();
                            assertTrue(b.size() <= chunk);
                            for (int i = 0; i < b.size(); i++) {
                                got.add(b.vector(i));
                            }
                        }
                        assertEquals(range[1], r.position());
                    }
                    assertEquals(range[1] - range[0], got.size(), file + " " + Arrays.toString(range));
                    for (int i = 0; i < got.size(); i++) {
                        assertEquals(all.vector(range[0] + i), got.get(i), file + " row " + (range[0] + i));
                    }
                }
            }
            assertThrows(IllegalArgumentException.class, () -> PointFile.open(file, 4, 3, n + 1));
            assertThrows(IllegalArgumentException.class, () -> PointFile.open(file, 4, 6, 5));
            assertThrows(IllegalArgumentException.class, () -> PointFile.open(file, 4, -1, 5));
        }
    }
}
