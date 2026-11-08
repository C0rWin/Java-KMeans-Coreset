package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.testbed.data.PointFile.Header;
import io.github.c0rwin.coreset.testbed.data.PointFile.Layout;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

/**
 * Writes a {@link PointFile} row by row through fixed buffers. The file is built as {@code <target>.part} and moved
 * into place by {@link #finish()}, so a reader never sees a half-written file; {@link #close()} without
 * {@code finish} deletes it. Dense files learn {@code n} as they go; sparse ones need {@code n} and {@code nnz} up
 * front because the sections are written side by side. Not thread-safe.
 */
public final class PointFileWriter implements AutoCloseable {

    private static final int BUFFER_BYTES = 1 << 20;

    private final Path target;
    private final Path part;
    private final FileChannel ch;
    private final Layout layout;
    private final int d;
    private final int flags;
    private final long expectedRows; // sparse only
    private final long expectedEntries; // sparse only
    private final Section values;
    private final Section indices; // sparse only
    private final Section offsets; // sparse only
    private long rows;
    private long entries;
    private boolean finished;

    private PointFileWriter(Path target, Layout layout, int d, int flags, long n, long nnz) throws IOException {
        if (d < 1) {
            throw new IllegalArgumentException("d must be >= 1: " + d);
        }
        this.target = target;
        this.part = target.resolveSibling(target.getFileName() + ".part");
        this.layout = layout;
        this.d = d;
        this.flags = flags;
        this.expectedRows = n;
        this.expectedEntries = nnz;
        Path parent = target.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        this.ch = FileChannel.open(part, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                StandardOpenOption.READ, StandardOpenOption.WRITE);
        if (layout == Layout.DENSE) {
            values = new Section(PointFile.HEADER_BYTES);
            indices = null;
            offsets = null;
        } else {
            Header shape = new Header(layout, PointFile.VERSION, flags, n, d, nnz, 0L);
            offsets = new Section(PointFile.HEADER_BYTES);
            indices = new Section(shape.indicesOffset());
            values = new Section(shape.valuesOffset());
            offsets.buf.putLong(0L);
        }
    }

    /** A dense file of dimension {@code d}. */
    public static PointFileWriter dense(Path target, int d, int flags) throws IOException {
        return new PointFileWriter(target, Layout.DENSE, d, flags, -1, -1);
    }

    /** A sparse file of exactly {@code n} rows and {@code nnz} entries. */
    public static PointFileWriter csr(Path target, long n, int d, long nnz, int flags) throws IOException {
        if (n < 0 || n > Integer.MAX_VALUE || nnz < 0) {
            throw new IllegalArgumentException("need 0 <= n <= 2^31 - 1 and nnz >= 0: n=" + n + ", nnz=" + nnz);
        }
        return new PointFileWriter(target, Layout.CSR, d, flags, n, nnz);
    }

    public long rows() {
        return rows;
    }

    /** Appends a dense row of exactly {@code d} finite values. */
    public void writeRow(double[] row) throws IOException {
        if (layout != Layout.DENSE) {
            throw new IllegalStateException("sparse file: use writeRow(indices, values, from, to)");
        }
        if (row.length != d) {
            throw new IllegalArgumentException("row " + rows + " has " + row.length + " values, expected " + d);
        }
        if (rows == Integer.MAX_VALUE) {
            throw new IllegalStateException("more than 2^31 - 1 rows");
        }
        for (double v : row) {
            if (!Double.isFinite(v)) {
                throw new IllegalArgumentException("row " + rows + ": value is not finite: " + v);
            }
            entries += v != 0.0 ? 1 : 0;
            values.putDouble(v);
        }
        rows++;
    }

    /** Appends a sparse row from entries {@code [from, to)}: strictly increasing indices, finite non-zero values. */
    public void writeRow(int[] idx, double[] val, int from, int to) throws IOException {
        if (layout != Layout.CSR) {
            throw new IllegalStateException("dense file: use writeRow(double[])");
        }
        if (rows == expectedRows || entries + (to - from) > expectedEntries) {
            throw new IllegalStateException("more than the declared " + expectedRows + " rows or " + expectedEntries
                    + " entries");
        }
        int prev = -1;
        for (int e = from; e < to; e++) {
            int i = idx[e];
            double v = val[e];
            if (i <= prev || i >= d) {
                throw new IllegalArgumentException("row " + rows + ": index " + i + " not increasing or outside [0, "
                        + d + ")");
            }
            if (!Double.isFinite(v) || v == 0.0) {
                throw new IllegalArgumentException("row " + rows + ": value must be finite and non-zero: " + v);
            }
            prev = i;
            indices.putInt(i);
            values.putDouble(v);
        }
        entries += to - from;
        rows++;
        offsets.putLong(entries);
    }

    /** Flushes, writes the header with the payload checksum and moves the file into place. */
    public Header finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("already finished");
        }
        if (layout == Layout.CSR && (rows != expectedRows || entries != expectedEntries)) {
            throw new IllegalStateException("wrote " + rows + " rows and " + entries + " entries, declared "
                    + expectedRows + " and " + expectedEntries);
        }
        values.flush();
        if (layout == Layout.CSR) {
            offsets.flush();
            indices.flush();
        }
        Header shape = new Header(layout, PointFile.VERSION, flags, rows, d, entries, 0L);
        long end = shape.fileBytes();
        if (ch.size() < end) { // trailing zero padding of the index section, or an empty payload
            ch.write(ByteBuffer.allocate(1), end - 1);
        }
        Header h = new Header(layout, PointFile.VERSION, flags, rows, d, entries,
                PointFile.crc32c(ch, PointFile.HEADER_BYTES, end));
        ByteBuffer hb = h.encode();
        while (hb.hasRemaining()) {
            ch.write(hb, hb.position());
        }
        ch.force(false);
        ch.close();
        Files.move(part, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        finished = true;
        return h;
    }

    @Override
    public void close() throws IOException {
        if (!finished) {
            ch.close();
            Files.deleteIfExists(part);
        }
    }

    /** A buffered sequential region of the file. */
    private final class Section {
        final ByteBuffer buf = ByteBuffer.allocate(BUFFER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        long pos;

        Section(long pos) {
            this.pos = pos;
        }

        void putDouble(double v) throws IOException {
            if (buf.remaining() < Double.BYTES) {
                flush();
            }
            buf.putDouble(v);
        }

        void putLong(long v) throws IOException {
            if (buf.remaining() < Long.BYTES) {
                flush();
            }
            buf.putLong(v);
        }

        void putInt(int v) throws IOException {
            if (buf.remaining() < Integer.BYTES) {
                flush();
            }
            buf.putInt(v);
        }

        void flush() throws IOException {
            buf.flip();
            while (buf.hasRemaining()) {
                pos += ch.write(buf, pos);
            }
            buf.clear();
        }
    }
}
