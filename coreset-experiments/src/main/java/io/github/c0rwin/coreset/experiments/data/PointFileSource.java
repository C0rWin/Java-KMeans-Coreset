package io.github.c0rwin.coreset.experiments.data;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.testbed.data.PointFile;
import java.io.IOException;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A testbed {@link PointFile} ({@code .cbin} dense or {@code .csr} sparse) read by row range: every shard reader
 * positions directly at its first row, and rows arrive as CSR {@link PointBlock} chunks.
 */
public final class PointFileSource implements PointSource {

    private static final long serialVersionUID = 1L;

    private final String id;
    private final String path;
    private final long n;
    private final int d;
    private final LinkedHashMap<String, Object> provenance;

    private PointFileSource(String id, String path, PointFile.Header h, Map<String, Object> provenance) {
        this.id = id;
        this.path = path;
        this.n = h.n();
        this.d = h.d();
        this.provenance = new LinkedHashMap<>(provenance);
        this.provenance.put("file", path);
        this.provenance.put("layout", h.layout().toString());
        this.provenance.put("flags", h.flags());
        this.provenance.put("nnz", h.nnz());
        this.provenance.put("checksum", Long.toHexString(h.checksum()));
    }

    /** Opens {@code file} after a header check; {@code provenance} is extended with the header's facts. */
    public static PointFileSource open(String id, Path file, Map<String, Object> provenance) throws IOException {
        PointFile.Header h = PointFile.readHeader(file);
        if (h.n() < 1) {
            throw new IOException(file + " holds no rows");
        }
        return new PointFileSource(id, file.toAbsolutePath().toString(), h, provenance);
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

    @Override
    public Map<String, Object> provenance() {
        return provenance;
    }

    @Override
    public ChunkReader open(long from, long to, int chunkRows) throws IOException {
        PointFile.Reader r = PointFile.open(Path.of(path), chunkRows, from, to);
        return new ChunkReader() {
            @Override
            public PointBlock next() throws IOException {
                return r.hasNext() ? r.next() : null;
            }

            @Override
            public void close() throws IOException {
                r.close();
            }
        };
    }
}
