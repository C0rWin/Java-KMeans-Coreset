package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.zip.CRC32C;

/**
 * Compact binary point files, written once by {@link PointFileWriter} and streamed in chunks without per-point
 * objects: {@code .cbin} (dense) and {@code .csr} (sparse, compressed sparse row). Every point has weight 1 and rows
 * keep their source order. All numbers are little-endian; the payload starts 8-byte aligned so it can be mapped.
 *
 * <pre>
 * offset  bytes  field
 *      0      4  magic: "CBIN" (dense) or "CCSR" (sparse)
 *      4      2  version, 1
 *      6      2  flags: {@link #FLAG_ZSCORE}, {@link #FLAG_TFIDF}
 *      8      8  n, rows
 *     16      8  d, dimension
 *     24      8  nnz: non-zero values (dense) or stored entries (sparse)
 *     32      8  CRC32C of the payload
 *     40     24  reserved, zero
 *     64         payload
 *                  dense:  n * d float64, row-major
 *                  sparse: n + 1 int64 row offsets, nnz int32 column indices (strictly increasing per row),
 *                          zero padding to a multiple of 8 bytes, nnz float64 values (finite, non-zero)
 * </pre>
 */
public final class PointFile {

    public static final int HEADER_BYTES = 64;
    public static final int VERSION = 1;
    /** Every feature was standardized to mean 0 and standard deviation 1 (constant features to 0). */
    public static final int FLAG_ZSCORE = 1;
    /** Counts were replaced by L2-normalised tf-idf weights. */
    public static final int FLAG_TFIDF = 2;
    public static final int DEFAULT_CHUNK_ROWS = 1 << 16;
    /** Upper bound on the entries of one chunk (64 MiB of values); a chunk always holds at least one row. */
    static final int MAX_CHUNK_ENTRIES = 1 << 23;

    private PointFile() {
    }

    public enum Layout {
        DENSE("CBIN", ".cbin"), CSR("CCSR", ".csr");

        private final String magic;
        private final String extension;

        Layout(String magic, String extension) {
            this.magic = magic;
            this.extension = extension;
        }

        public String extension() {
            return extension;
        }
    }

    /** The decoded header. */
    public record Header(Layout layout, int version, int flags, long n, int d, long nnz, long checksum) {

        /** Size of the payload implied by the header. */
        public long payloadBytes() {
            return layout == Layout.DENSE ? n * d * Double.BYTES : valuesOffset() - HEADER_BYTES + nnz * Double.BYTES;
        }

        public long fileBytes() {
            return HEADER_BYTES + payloadBytes();
        }

        long indicesOffset() {
            return HEADER_BYTES + (n + 1) * Long.BYTES;
        }

        long valuesOffset() {
            long end = indicesOffset() + nnz * Integer.BYTES;
            return (end + 7) & ~7L;
        }

        ByteBuffer encode() {
            ByteBuffer b = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            b.put(layout.magic.getBytes(StandardCharsets.US_ASCII)).putShort((short) version).putShort((short) flags)
                    .putLong(n).putLong(d).putLong(nnz).putLong(checksum);
            return b.position(0);
        }

        static Header decode(ByteBuffer b, Path file) throws IOException {
            b.order(ByteOrder.LITTLE_ENDIAN);
            byte[] magic = new byte[4];
            b.get(magic);
            String m = new String(magic, StandardCharsets.US_ASCII);
            Layout layout = null;
            for (Layout l : Layout.values()) {
                if (l.magic.equals(m)) {
                    layout = l;
                }
            }
            if (layout == null) {
                throw new IOException(file + ": not a point file (magic '" + m + "')");
            }
            int version = Short.toUnsignedInt(b.getShort());
            if (version != VERSION) {
                throw new IOException(file + ": unsupported version " + version);
            }
            int flags = Short.toUnsignedInt(b.getShort());
            long n = b.getLong();
            long d = b.getLong();
            long nnz = b.getLong();
            long checksum = b.getLong();
            if (n < 0 || n > Integer.MAX_VALUE || d < 0 || d > Integer.MAX_VALUE || nnz < 0
                    || (layout == Layout.DENSE ? nnz > n * d : (n > 0 && d == 0 && nnz > 0))) {
                throw new IOException(file + ": corrupt header (n=" + n + ", d=" + d + ", nnz=" + nnz + ")");
            }
            return new Header(layout, version, flags, n, (int) d, nnz, checksum);
        }
    }

    public static Header readHeader(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            return readHeader(ch, file);
        }
    }

    private static Header readHeader(FileChannel ch, Path file) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(HEADER_BYTES);
        readFully(ch, b, 0, file);
        Header h = Header.decode(b.flip(), file);
        if (ch.size() != h.fileBytes()) {
            throw new IOException(file + ": " + ch.size() + " bytes, header implies " + h.fileBytes());
        }
        return h;
    }

    /** Recomputes the payload checksum. @throws IOException if it differs from the header */
    public static Header verify(Path file) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            Header h = readHeader(ch, file);
            long actual = crc32c(ch, HEADER_BYTES, h.fileBytes());
            if (actual != h.checksum()) {
                throw new IOException(file + ": payload CRC32C " + Long.toHexString(actual) + " does not match header "
                        + Long.toHexString(h.checksum()));
            }
            return h;
        }
    }

    static long crc32c(FileChannel ch, long from, long to) throws IOException {
        CRC32C crc = new CRC32C();
        ByteBuffer b = ByteBuffer.allocateDirect(1 << 20);
        for (long pos = from; pos < to; ) {
            b.clear().limit((int) Math.min(b.capacity(), to - pos));
            int r = ch.read(b, pos);
            if (r < 0) {
                throw new EOFException("payload ends at " + pos + ", expected " + to);
            }
            pos += r;
            crc.update(b.flip());
        }
        return crc.getValue();
    }

    static void readFully(FileChannel ch, ByteBuffer b, long pos, Path file) throws IOException {
        while (b.hasRemaining()) {
            int r = ch.read(b, pos);
            if (r < 0) {
                throw new EOFException(file + ": unexpected end at byte " + pos);
            }
            pos += r;
        }
    }

    /** Opens a chunked reader; {@code chunkRows} caps the rows per block (entries are capped too). */
    public static Reader open(Path file, int chunkRows) throws IOException {
        return new Reader(file, chunkRows);
    }

    /**
     * Opens a chunked reader over rows {@code [from, to)} only. Dense rows are fixed-width and sparse rows are found
     * through the row offsets, so nothing before {@code from} is read: parallel readers each stream their own range.
     */
    public static Reader open(Path file, int chunkRows, long from, long to) throws IOException {
        return new Reader(file, chunkRows).range(from, to);
    }

    /** All rows as one point set whose segments are the chunks; no per-point objects. */
    public static PointSet readAll(Path file) throws IOException {
        try (Reader r = open(file, DEFAULT_CHUNK_ROWS)) {
            List<PointSet> parts = new ArrayList<>();
            while (r.hasNext()) {
                parts.add(PointSet.of(r.next()));
            }
            return PointSet.concat(parts);
        }
    }

    /** Streams the rows of a dense file, zeros included; {@code values} is reused. */
    public static void forEachRow(Path file, Loaders.DenseRowSink sink) throws IOException {
        try (Reader r = open(file, DEFAULT_CHUNK_ROWS)) {
            r.require(Layout.DENSE);
            double[] row = new double[r.header.d()];
            while (r.hasNext()) {
                int rows = r.loadDense();
                for (int i = 0; i < rows; i++) {
                    System.arraycopy(r.dense, i * row.length, row, 0, row.length);
                    sink.row(row);
                }
            }
        }
    }

    /** Streams the rows of a sparse file as slices of reused arrays. */
    public static void forEachRow(Path file, Loaders.SparseRowSink sink) throws IOException {
        try (Reader r = open(file, DEFAULT_CHUNK_ROWS)) {
            r.require(Layout.CSR);
            while (r.hasNext()) {
                int rows = r.loadSparse();
                for (int i = 0; i < rows; i++) {
                    sink.row(r.indices, r.values, r.offsets[i], r.offsets[i + 1]);
                }
            }
        }
    }

    /**
     * Sequential reader that turns each chunk of rows into one {@link PointBlock} (dense rows drop their zeros, as
     * every block does). The I/O buffer is reused; each block owns exactly sized arrays. Not thread-safe.
     */
    public static final class Reader implements AutoCloseable {
        private final Path file;
        private final FileChannel ch;
        private final Header header;
        private final int chunkRows;
        private long row;
        private long end;
        private ByteBuffer io = ByteBuffer.allocate(0);
        // Scratch of the last loaded chunk: dense values, or CSR offsets (relative), indices and values.
        private double[] dense = new double[0];
        private int[] offsets = new int[0];
        private int[] indices = new int[0];
        private double[] values = new double[0];

        private Reader(Path file, int chunkRows) throws IOException {
            if (chunkRows < 1) {
                throw new IllegalArgumentException("chunkRows must be >= 1: " + chunkRows);
            }
            this.file = file;
            this.ch = FileChannel.open(file, StandardOpenOption.READ);
            try {
                this.header = readHeader(ch, file);
            } catch (IOException | RuntimeException e) {
                ch.close();
                throw e;
            }
            int d = Math.max(1, header.d());
            this.chunkRows = header.layout() == Layout.DENSE
                    ? Math.max(1, Math.min(chunkRows, MAX_CHUNK_ENTRIES / d)) : chunkRows;
            this.end = header.n();
        }

        private Reader range(long from, long to) throws IOException {
            if (from < 0 || from > to || to > header.n()) {
                ch.close();
                throw new IllegalArgumentException("rows [" + from + ", " + to + ") outside [0, " + header.n() + "]");
            }
            row = from;
            end = to;
            return this;
        }

        public Header header() {
            return header;
        }

        public boolean hasNext() {
            return row < end;
        }

        /** Index of the next row; rows already returned when reading from row 0. */
        public long position() {
            return row;
        }

        /** The next chunk of rows. @throws NoSuchElementException after the last row */
        public PointBlock next() throws IOException {
            if (!hasNext()) {
                throw new NoSuchElementException();
            }
            return header.layout() == Layout.DENSE ? denseBlock() : sparseBlock();
        }

        private PointBlock denseBlock() throws IOException {
            final int d = header.d();
            final int rows = loadDense();
            int nnz = 0;
            for (int t = 0, len = rows * d; t < len; t++) {
                nnz += dense[t] != 0.0 ? 1 : 0;
            }
            int[] off = new int[rows + 1];
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int i = 0, e = 0, t = 0; i < rows; i++) {
                for (int c = 0; c < d; c++, t++) {
                    double v = dense[t];
                    if (v != 0.0) {
                        idx[e] = c;
                        val[e++] = v;
                    }
                }
                off[i + 1] = e;
            }
            return wrap(off, idx, val, rows);
        }

        private PointBlock sparseBlock() throws IOException {
            final int rows = loadSparse();
            int nnz = offsets[rows];
            return wrap(Arrays.copyOf(offsets, rows + 1), Arrays.copyOf(indices, nnz), Arrays.copyOf(values, nnz),
                    rows);
        }

        private PointBlock wrap(int[] off, int[] idx, double[] val, int rows) {
            double[] w = new double[rows];
            Arrays.fill(w, 1.0);
            return PointBlock.wrap(header.d(), off, idx, val, w);
        }

        /** Reads the next dense chunk into {@link #dense}; returns its rows. */
        private int loadDense() throws IOException {
            final int d = header.d();
            final int rows = (int) Math.min(chunkRows, end - row);
            final int len = rows * d;
            ByteBuffer b = buffer((long) len * Double.BYTES);
            readFully(ch, b, HEADER_BYTES + row * d * Double.BYTES, file);
            if (dense.length < len) {
                dense = new double[len];
            }
            b.flip().asDoubleBuffer().get(dense, 0, len);
            row += rows;
            return rows;
        }

        /** Reads the next sparse chunk into {@link #offsets}, {@link #indices}, {@link #values}; returns its rows. */
        private int loadSparse() throws IOException {
            int rows = (int) Math.min(chunkRows, end - row);
            ByteBuffer b = buffer((long) (rows + 1) * Long.BYTES);
            readFully(ch, b, HEADER_BYTES + row * Long.BYTES, file);
            b.flip();
            long first = b.getLong(0);
            while (rows > 1 && b.getLong(rows * Long.BYTES) - first > MAX_CHUNK_ENTRIES) {
                rows--;
            }
            if (offsets.length < rows + 1) {
                offsets = new int[rows + 1];
            }
            for (int i = 0; i <= rows; i++) {
                long o = b.getLong(i * Long.BYTES) - first;
                if (o < 0 || o > Integer.MAX_VALUE - 8 || (i > 0 && o < offsets[i - 1])
                        || first + o > header.nnz()) {
                    throw new IOException(file + ": corrupt row offset at row " + (row + i));
                }
                offsets[i] = (int) o;
            }
            final int nnz = offsets[rows];
            if (indices.length < nnz) {
                indices = new int[nnz];
                values = new double[nnz];
            }
            b = buffer((long) nnz * Integer.BYTES);
            readFully(ch, b, header.indicesOffset() + first * Integer.BYTES, file);
            b.flip().asIntBuffer().get(indices, 0, nnz);
            b = buffer((long) nnz * Double.BYTES);
            readFully(ch, b, header.valuesOffset() + first * Double.BYTES, file);
            b.flip().asDoubleBuffer().get(values, 0, nnz);
            row += rows;
            return rows;
        }

        private ByteBuffer buffer(long bytes) {
            if (io.capacity() < bytes) {
                io = ByteBuffer.allocate(Math.toIntExact(bytes)).order(ByteOrder.LITTLE_ENDIAN);
            }
            io.clear().limit((int) bytes);
            return io;
        }

        private void require(Layout layout) {
            if (header.layout() != layout) {
                throw new IllegalArgumentException(file + " is " + header.layout() + ", expected " + layout);
            }
        }

        @Override
        public void close() throws IOException {
            ch.close();
        }
    }
}
