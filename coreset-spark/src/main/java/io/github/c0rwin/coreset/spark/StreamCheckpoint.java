package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.Coreset;
import java.io.BufferedOutputStream;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.UUID;
import java.util.zip.CRC32;
import java.util.zip.CheckedOutputStream;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

/**
 * Versioned, checksummed files {@code state-<batchId>.bin} holding a streaming tree after a batch, on any Hadoop
 * file system. Each file is written under a temporary name and renamed into place, so a reader sees either the whole
 * state of a batch or none of it (atomic on HDFS and local file systems; object stores without atomic rename need a
 * committing file system).
 */
final class StreamCheckpoint {

    private static final int MAGIC = 0x43535354; // "CSST"
    private static final int VERSION = 1;
    private static final String PREFIX = "state-";
    private static final String SUFFIX = ".bin";
    private static final String TMP_PREFIX = "_tmp-" + PREFIX;
    /** Older states kept besides the newest, for inspection after a failure. */
    static final int RETAIN = 2;

    /** Everything needed to resume: the sink configuration (to reject a changed one) and the tree. */
    record State(long seed, String algorithmId, SortedMap<String, String> params, int treeDepth, int chunkSize,
                 long lastBatchId, long batches, List<Coreset> levels) {

        State {
            params = new TreeMap<>(params);
            levels = List.copyOf(levels);
        }

        boolean sameConfig(long seed, String algorithmId, Map<String, String> params, int treeDepth, int chunkSize) {
            return this.seed == seed && this.algorithmId.equals(algorithmId) && this.params.equals(new TreeMap<>(params))
                    && this.treeDepth == treeDepth && this.chunkSize == chunkSize;
        }
    }

    private final Path dir;
    private final FileSystem fs;

    /** Opens (creating) {@code dir} and deletes temporaries a crashed writer left behind; one writer per directory. */
    StreamCheckpoint(String dir, Configuration conf) throws IOException {
        this.dir = new Path(dir);
        this.fs = this.dir.getFileSystem(conf);
        fs.mkdirs(this.dir);
        for (FileStatus f : fs.listStatus(this.dir, p -> p.getName().startsWith(TMP_PREFIX))) {
            fs.delete(f.getPath(), false);
        }
    }

    /** Newest state, or empty for a fresh directory; a corrupt newest state fails rather than silently rolling back. */
    Optional<State> latest() throws IOException {
        List<FileStatus> files = stateFiles();
        if (files.isEmpty()) {
            return Optional.empty();
        }
        Path newest = files.get(0).getPath();
        byte[] bytes = new byte[Math.toIntExact(files.get(0).getLen())];
        try (FSDataInputStream in = fs.open(newest)) {
            in.readFully(bytes);
        }
        try {
            return Optional.of(decode(bytes));
        } catch (IOException | IllegalArgumentException e) {
            throw new IOException("corrupt coreset checkpoint " + newest + ": " + e.getMessage(), e);
        }
    }

    void write(State state) throws IOException {
        Path tmp = new Path(dir, TMP_PREFIX + state.lastBatchId() + "-" + UUID.randomUUID());
        try (FSDataOutputStream out = fs.create(tmp, false)) {
            encode(state, out);
            out.hsync();
        }
        Path target = new Path(dir, PREFIX + state.lastBatchId() + SUFFIX);
        if (!fs.rename(tmp, target)) {
            fs.delete(tmp, false);
            throw new IOException("could not rename " + tmp + " to " + target);
        }
        List<FileStatus> files = stateFiles();
        for (int i = RETAIN + 1; i < files.size(); i++) {
            fs.delete(files.get(i).getPath(), false);
        }
    }

    /** State files, newest batch first. */
    private List<FileStatus> stateFiles() throws IOException {
        List<FileStatus> files = new ArrayList<>(Arrays.asList(fs.listStatus(dir, p -> batchId(p.getName()) >= 0)));
        files.sort(Comparator.comparingLong((FileStatus f) -> batchId(f.getPath().getName())).reversed());
        return files;
    }

    private static long batchId(String name) {
        if (!name.startsWith(PREFIX) || !name.endsWith(SUFFIX)) {
            return -1;
        }
        try {
            return Long.parseLong(name.substring(PREFIX.length(), name.length() - SUFFIX.length()));
        } catch (NumberFormatException e) {
            return -1;
        }
    }

    /** Writes the state followed by the CRC32 of everything before it; flushes but does not close {@code sink}. */
    static void encode(State s, OutputStream sink) throws IOException {
        BufferedOutputStream buffered = new BufferedOutputStream(sink, 1 << 16);
        CRC32 crc = new CRC32();
        DataOutputStream out = new DataOutputStream(new CheckedOutputStream(buffered, crc));
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        out.writeLong(s.seed());
        out.writeUTF(s.algorithmId());
        out.writeInt(s.params().size());
        for (Map.Entry<String, String> e : s.params().entrySet()) {
            out.writeUTF(e.getKey());
            out.writeUTF(e.getValue());
        }
        out.writeInt(s.treeDepth());
        out.writeInt(s.chunkSize());
        out.writeLong(s.lastBatchId());
        out.writeLong(s.batches());
        out.writeInt(s.levels().size());
        for (Coreset c : s.levels()) {
            byte[] encoded = CoresetCodec.encode(c);
            out.writeInt(encoded.length);
            out.write(encoded);
        }
        out.flush();
        new DataOutputStream(buffered).writeLong(crc.getValue());
        buffered.flush();
    }

    static State decode(byte[] bytes) throws IOException {
        if (bytes.length < 16) {
            throw new IOException("checkpoint too short: " + bytes.length + " bytes");
        }
        int body = bytes.length - 8;
        CRC32 crc = new CRC32();
        crc.update(bytes, 0, body);
        if (ByteBuffer.wrap(bytes, body, 8).getLong() != crc.getValue()) {
            throw new IOException("checksum mismatch");
        }
        DataInputStream in = new DataInputStream(new ByteArrayInputStream(bytes, 0, body));
        if (in.readInt() != MAGIC) {
            throw new IOException("not a coreset checkpoint");
        }
        int version = in.readInt();
        if (version != VERSION) {
            throw new IOException("unsupported checkpoint version " + version);
        }
        long seed = in.readLong();
        String algorithmId = in.readUTF();
        int nParams = in.readInt();
        SortedMap<String, String> params = new TreeMap<>();
        for (int i = 0; i < nParams; i++) {
            params.put(in.readUTF(), in.readUTF());
        }
        int treeDepth = in.readInt();
        int chunkSize = in.readInt();
        long lastBatchId = in.readLong();
        long batches = in.readLong();
        int nLevels = in.readInt();
        if (nLevels != Long.bitCount(batches)) {
            throw new IOException(nLevels + " levels stored for " + batches + " batches");
        }
        List<Coreset> levels = new ArrayList<>(nLevels);
        for (int i = 0; i < nLevels; i++) {
            int len = in.readInt();
            if (len < 0 || len > in.available()) {
                throw new IOException("bad level length " + len);
            }
            levels.add(CoresetCodec.decode(bytes, body - in.available(), len)); // decoded in place
            in.skipNBytes(len);
        }
        if (in.available() != 0) {
            throw new IOException(in.available() + " trailing bytes");
        }
        return new State(seed, algorithmId, params, treeDepth, chunkSize, lastBatchId, batches, levels);
    }
}
