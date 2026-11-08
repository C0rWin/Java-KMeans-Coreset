package io.github.c0rwin.coreset.experiments.data;

import io.github.c0rwin.coreset.core.PointBlock;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Dense rows in a flat little-endian file: a 64-byte header ({@code CSXDENSE}, version, flags, n, d) followed by
 * {@code n} rows of {@code d} doubles, plus one weight double per row when the weighted flag is set (otherwise every
 * weight is 1). Fixed-width rows make any row range one positioned read.
 */
public final class DenseFileSource implements PointSource {

    private static final long serialVersionUID = 1L;
    static final long MAGIC = 0x45534E4544585343L; // "CSXDENSE" as a little-endian long
    static final int VERSION = 1;
    static final int HEADER_BYTES = 64;
    private static final int FLAG_WEIGHTED = 1;

    private final String id;
    private final String path;
    private final long n;
    private final int d;
    private final boolean weighted;
    private final LinkedHashMap<String, Object> provenance;

    private DenseFileSource(String id, String path, long n, int d, boolean weighted, Map<String, Object> provenance) {
        this.id = id;
        this.path = path;
        this.n = n;
        this.d = d;
        this.weighted = weighted;
        this.provenance = new LinkedHashMap<>(provenance);
        this.provenance.put("file", path);
        this.provenance.put("format", "dense-f64");
    }

    /** Opens an existing file; its header gives n and d. */
    public static DenseFileSource open(String id, Path file, Map<String, Object> provenance) throws IOException {
        try (FileChannel ch = FileChannel.open(file, StandardOpenOption.READ)) {
            ByteBuffer h = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
            while (h.hasRemaining() && ch.read(h) >= 0) {
                // keep reading the header
            }
            h.flip();
            if (h.remaining() < HEADER_BYTES || h.getLong(0) != MAGIC || h.getInt(8) != VERSION) {
                throw new IOException(file + " is not a dense-f64 points file");
            }
            int flags = h.getInt(12);
            long n = h.getLong(16);
            int d = h.getInt(24);
            boolean weighted = (flags & FLAG_WEIGHTED) != 0;
            long expected = HEADER_BYTES + n * rowBytes(d, weighted);
            if (n < 1 || d < 1 || ch.size() != expected) {
                throw new IOException(file + ": header says " + n + " x " + d + " (" + expected + " bytes), file has "
                        + ch.size() + " bytes");
            }
            return new DenseFileSource(id, file.toAbsolutePath().toString(), n, d, weighted, provenance);
        }
    }

    private static int rowBytes(int d, boolean weighted) {
        return Double.BYTES * (d + (weighted ? 1 : 0));
    }

    @Override
    public String id() {
        return id;
    }

    @Override
    public long size() {
        return n;
    }

    @Override
    public int dim() {
        return d;
    }

    public Path path() {
        return Path.of(path);
    }

    @Override
    public Map<String, Object> provenance() {
        return provenance;
    }

    @Override
    public ChunkReader open(long from, long to) throws IOException {
        if (from < 0 || to > n || from > to) {
            throw new IllegalArgumentException("row range [" + from + ", " + to + ") outside [0, " + n + ")");
        }
        return new Reader(FileChannel.open(Path.of(path), StandardOpenOption.READ), from, to);
    }

    private final class Reader implements ChunkReader {
        private final FileChannel ch;
        private final int rowBytes = rowBytes(d, weighted);
        private long next;
        private final long end;
        private ByteBuffer buf;

        Reader(FileChannel ch, long from, long to) {
            this.ch = ch;
            this.next = from;
            this.end = to;
        }

        @Override
        public PointBlock next(int maxRows) throws IOException {
            int rows = (int) Math.min(maxRows, end - next);
            if (rows <= 0) {
                return null;
            }
            int bytes = Math.multiplyExact(rows, rowBytes);
            if (buf == null || buf.capacity() < bytes) {
                buf = ByteBuffer.allocateDirect(bytes).order(ByteOrder.LITTLE_ENDIAN);
            }
            buf.clear().limit(bytes);
            long pos = HEADER_BYTES + next * rowBytes;
            while (buf.hasRemaining()) {
                int r = ch.read(buf, pos + buf.position());
                if (r < 0) {
                    throw new IOException("unexpected end of " + path + " at row " + next);
                }
            }
            buf.flip();
            int[] offsets = new int[rows + 1];
            int[] indices = new int[rows * d];
            double[] values = new double[rows * d];
            double[] weights = new double[rows];
            int e = 0;
            for (int r = 0, p = 0; r < rows; r++) {
                for (int t = 0; t < d; t++, p += Double.BYTES) {
                    double v = buf.getDouble(p);
                    if (v != 0.0) { // CSR rows hold no explicit zeros, as SparseVector.dense
                        indices[e] = t;
                        values[e++] = v;
                    }
                }
                if (weighted) {
                    weights[r] = buf.getDouble(p);
                    p += Double.BYTES;
                } else {
                    weights[r] = 1.0;
                }
                offsets[r + 1] = e;
            }
            next += rows;
            return PointBlock.wrap(d, offsets, indices, values, weights);
        }

        @Override
        public void close() throws IOException {
            ch.close();
        }
    }

    /** Streams rows into a new file; {@link #close()} writes the header, so a crashed write leaves no valid file. */
    public static final class Writer implements AutoCloseable {
        private final FileChannel ch;
        private final int d;
        private final boolean weighted;
        private final ByteBuffer buf;
        private long rows;

        public Writer(Path file, int d, boolean weighted) throws IOException {
            this.ch = FileChannel.open(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE);
            this.d = d;
            this.weighted = weighted;
            this.buf = ByteBuffer.allocateDirect(Math.max(1 << 20, rowBytes(d, weighted))).order(ByteOrder.LITTLE_ENDIAN);
            ch.position(HEADER_BYTES);
        }

        /** Appends one row; {@code values} is read, not kept. */
        public void add(double[] values, double weight) throws IOException {
            if (values.length != d) {
                throw new IllegalArgumentException("row has " + values.length + " values, expected " + d);
            }
            if (!weighted && weight != 1.0) {
                throw new IllegalArgumentException("unweighted file, row weight " + weight);
            }
            if (buf.remaining() < rowBytes(d, weighted)) {
                drain();
            }
            for (double v : values) {
                buf.putDouble(v);
            }
            if (weighted) {
                buf.putDouble(weight);
            }
            rows++;
        }

        private void drain() throws IOException {
            buf.flip();
            while (buf.hasRemaining()) {
                ch.write(buf);
            }
            buf.clear();
        }

        @Override
        public void close() throws IOException {
            try {
                drain();
                ByteBuffer h = ByteBuffer.allocate(HEADER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
                h.putLong(MAGIC).putInt(VERSION).putInt(weighted ? FLAG_WEIGHTED : 0).putLong(rows).putInt(d);
                h.position(0).limit(HEADER_BYTES);
                long pos = 0;
                while (h.hasRemaining()) {
                    pos += ch.write(h, pos);
                }
                ch.force(true);
            } finally {
                ch.close();
            }
        }
    }
}
