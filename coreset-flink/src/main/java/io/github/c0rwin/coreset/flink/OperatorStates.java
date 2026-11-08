package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/**
 * Checkpointed operator state of this module and its serializers (format version 1). Public only because Flink
 * instantiates serializer snapshots reflectively; not a stable API. Every value is immutable, and every
 * {@code encodedLength} is the exact size its serializer writes, which the state-size metrics report.
 */
public final class OperatorStates {

    private OperatorStates() {
    }

    /**
     * One chunked merge-reduce tree: the current epoch's levels and the partially filled chunk.
     *
     * @param firstChunk index (among all chunks of its stream) of the epoch's first chunk; seeds the merges
     * @param batches    chunks added to the tree this epoch
     * @param levels     occupied tree levels, lowest first
     * @param buffer     points of the chunk being filled
     * @param records    records of the epoch, buffered ones included
     * @param weight     their total weight
     */
    public record Builder(long firstChunk, long batches, List<Coreset> levels, List<WeightedPoint> buffer,
                          long records, double weight) {

        public Builder {
            levels = List.copyOf(levels);
            buffer = List.copyOf(buffer);
            if (batches < 0 || levels.size() != Long.bitCount(batches)) {
                throw new IllegalArgumentException(levels.size() + " levels for " + batches + " batches");
            }
        }

        static final long FIXED_BYTES = 8 + 8 + 8 + 8;

        long encodedLength() {
            return FIXED_BYTES + PointCodec.coresetsLength(levels) + PointCodec.pointsLength(buffer);
        }

        void write(DataOutputView out) throws IOException {
            out.writeLong(firstChunk);
            out.writeLong(batches);
            out.writeLong(records);
            PointCodec.writeF64(weight, out);
            PointCodec.writeCoresets(levels, out);
            PointCodec.writePoints(buffer, out);
        }

        static Builder read(DataInputView in) throws IOException {
            long firstChunk = in.readLong();
            long batches = in.readLong();
            long records = in.readLong();
            double weight = PointCodec.readF64(in);
            List<Coreset> levels = PointCodec.readCoresets(in);
            List<WeightedPoint> buffer = PointCodec.readPoints(in);
            try {
                return new Builder(firstChunk, batches, levels, buffer, records, weight);
            } catch (IllegalArgumentException e) {
                throw new IOException(e.getMessage(), e);
            }
        }

        static void copy(DataInputView in, DataOutputView out) throws IOException {
            out.write(in, (int) FIXED_BYTES);
            PointCodec.copyCoresets(in, out);
            PointCodec.copyPoints(in, out);
        }
    }

    /**
     * State of one subtask of the local stage of {@link FlinkCoresets#stream}.
     *
     * @param fingerprint       {@link CoresetConfig#fingerprint()} of the writer, checked on restore
     * @param records           records summarised by the partials emitted so far
     * @param nextEventBoundary next event-time instant that triggers a partial, {@code Long.MIN_VALUE} if unset
     * @param peakStateBytes    largest encoded size this state reached at a chunk boundary or checkpoint
     */
    public record Local(long fingerprint, int subtask, int parallelism, long partialsEmitted, long records,
                        long nextEventBoundary, long peakStateBytes, Builder builder) {

        public Local {
            Objects.requireNonNull(builder, "builder");
        }

        static final long FIXED_BYTES = 8 + 4 + 4 + 8 + 8 + 8 + 8;
    }

    /** Open window of the local stage of {@link FlinkCoresets#tumblingWindows}. */
    public record OpenWindow(long start, Builder builder) {

        public OpenWindow {
            Objects.requireNonNull(builder, "builder");
        }
    }

    /**
     * State of one subtask of the local stage of {@link FlinkCoresets#tumblingWindows}.
     *
     * @param watermark highest watermark seen; restored so that no emitted window can reopen after a recovery
     */
    public record Window(long fingerprint, int subtask, int parallelism, long watermark, long records,
                         long lateRecords, long partialsEmitted, long peakStateBytes, List<OpenWindow> windows) {

        public Window {
            windows = List.copyOf(windows);
        }

        static final long FIXED_BYTES = 8 + 4 + 4 + 8 + 8 + 8 + 8 + 8 + 4;
    }

    /**
     * Merge-stage tree of one local subtask's partials.
     *
     * @param nextSequence sequence number the next partial of this subtask must carry
     * @param records      records summarised by the subtask's partials so far
     * @param weight       their total weight, summed in sequence order so it does not depend on arrival interleaving
     * @param depths       compressions on the deepest path inside each level, parallel to {@code levels}
     */
    public record SubtaskTree(long nextSequence, long records, double weight, long batches, List<Coreset> levels,
                              List<Integer> depths) {

        public SubtaskTree {
            levels = List.copyOf(levels);
            depths = List.copyOf(depths);
            if (batches < 0 || levels.size() != Long.bitCount(batches) || depths.size() != levels.size()) {
                throw new IllegalArgumentException(
                        levels.size() + " levels and " + depths.size() + " depths for " + batches + " batches");
            }
        }

        static final SubtaskTree EMPTY = new SubtaskTree(0, 0, 0.0, 0, List.of(), List.of());

        static final long FIXED_BYTES = 8 + 8 + 8 + 8;

        long encodedLength() {
            return FIXED_BYTES + PointCodec.coresetsLength(levels) + 4L * depths.size();
        }
    }

    /**
     * State of the merge stage of {@link FlinkCoresets#stream}.
     *
     * @param parallelism   parallelism of the local stage, 0 until the first partial arrived
     * @param sinceSnapshot partials merged since the last count-triggered snapshot
     * @param trees         one tree per local subtask, indexed by subtask
     */
    public record Merge(long fingerprint, int parallelism, long partials, long snapshots, long sinceSnapshot,
                        long peakStateBytes, List<SubtaskTree> trees) {

        public Merge {
            trees = List.copyOf(trees);
        }

        static final long FIXED_BYTES = 8 + 4 + 8 + 8 + 8 + 8 + 4;
    }

    static long encodedLength(Local s) {
        return Local.FIXED_BYTES + s.builder().encodedLength();
    }

    static long encodedLength(Window s) {
        long size = Window.FIXED_BYTES;
        for (OpenWindow w : s.windows()) {
            size += 8 + w.builder().encodedLength();
        }
        return size;
    }

    static long encodedLength(Merge s) {
        long size = Merge.FIXED_BYTES;
        for (SubtaskTree t : s.trees()) {
            size += t.encodedLength();
        }
        return size;
    }

    /** Serializer of {@link Local}. */
    public static final class LocalSerializer extends ImmutableSerializer<Local> {

        private static final long serialVersionUID = 1L;
        public static final LocalSerializer INSTANCE = new LocalSerializer();
        private static final Local EMPTY = new Local(0, 0, 1, 0, 0, Long.MIN_VALUE, 0,
                new Builder(0, 0, List.of(), List.of(), 0, 0.0));

        private LocalSerializer() {
        }

        @Override
        public Local createInstance() {
            return EMPTY;
        }

        @Override
        public void serialize(Local r, DataOutputView out) throws IOException {
            out.writeLong(r.fingerprint());
            out.writeInt(r.subtask());
            out.writeInt(r.parallelism());
            out.writeLong(r.partialsEmitted());
            out.writeLong(r.records());
            out.writeLong(r.nextEventBoundary());
            out.writeLong(r.peakStateBytes());
            r.builder().write(out);
        }

        @Override
        public Local deserialize(DataInputView in) throws IOException {
            long fingerprint = in.readLong();
            int subtask = in.readInt();
            int parallelism = in.readInt();
            long partials = in.readLong();
            long records = in.readLong();
            long boundary = in.readLong();
            long peak = in.readLong();
            return new Local(fingerprint, subtask, parallelism, partials, records, boundary, peak, Builder.read(in));
        }

        @Override
        public void copy(DataInputView in, DataOutputView out) throws IOException {
            out.write(in, (int) Local.FIXED_BYTES);
            Builder.copy(in, out);
        }

        @Override
        public TypeSerializerSnapshot<Local> snapshotConfiguration() {
            return new Snapshot();
        }

        private Object readResolve() {
            return INSTANCE;
        }

        /** Snapshot of {@link LocalSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
        public static final class Snapshot extends FormatSnapshot<Local> {
            public Snapshot() {
                super(() -> INSTANCE);
            }
        }
    }

    /** Serializer of {@link Window}. */
    public static final class WindowSerializer extends ImmutableSerializer<Window> {

        private static final long serialVersionUID = 1L;
        public static final WindowSerializer INSTANCE = new WindowSerializer();
        private static final Window EMPTY = new Window(0, 0, 1, Long.MIN_VALUE, 0, 0, 0, 0, List.of());

        private WindowSerializer() {
        }

        @Override
        public Window createInstance() {
            return EMPTY;
        }

        @Override
        public void serialize(Window r, DataOutputView out) throws IOException {
            out.writeLong(r.fingerprint());
            out.writeInt(r.subtask());
            out.writeInt(r.parallelism());
            out.writeLong(r.watermark());
            out.writeLong(r.records());
            out.writeLong(r.lateRecords());
            out.writeLong(r.partialsEmitted());
            out.writeLong(r.peakStateBytes());
            out.writeInt(r.windows().size());
            for (OpenWindow w : r.windows()) {
                out.writeLong(w.start());
                w.builder().write(out);
            }
        }

        @Override
        public Window deserialize(DataInputView in) throws IOException {
            long fingerprint = in.readLong();
            int subtask = in.readInt();
            int parallelism = in.readInt();
            long watermark = in.readLong();
            long records = in.readLong();
            long late = in.readLong();
            long partials = in.readLong();
            long peak = in.readLong();
            int n = PointCodec.count(in);
            List<OpenWindow> windows = new ArrayList<>(Math.min(n, 1024));
            for (int i = 0; i < n; i++) {
                long start = in.readLong();
                windows.add(new OpenWindow(start, Builder.read(in)));
            }
            return new Window(fingerprint, subtask, parallelism, watermark, records, late, partials, peak, windows);
        }

        @Override
        public void copy(DataInputView in, DataOutputView out) throws IOException {
            out.write(in, (int) Window.FIXED_BYTES - 4);
            int n = in.readInt();
            out.writeInt(n);
            for (int i = 0; i < n; i++) {
                out.writeLong(in.readLong());
                Builder.copy(in, out);
            }
        }

        @Override
        public TypeSerializerSnapshot<Window> snapshotConfiguration() {
            return new Snapshot();
        }

        private Object readResolve() {
            return INSTANCE;
        }

        /** Snapshot of {@link WindowSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
        public static final class Snapshot extends FormatSnapshot<Window> {
            public Snapshot() {
                super(() -> INSTANCE);
            }
        }
    }

    /** Serializer of {@link Merge}. */
    public static final class MergeSerializer extends ImmutableSerializer<Merge> {

        private static final long serialVersionUID = 1L;
        public static final MergeSerializer INSTANCE = new MergeSerializer();
        private static final Merge EMPTY = new Merge(0, 0, 0, 0, 0, 0, List.of());

        private MergeSerializer() {
        }

        @Override
        public Merge createInstance() {
            return EMPTY;
        }

        @Override
        public void serialize(Merge r, DataOutputView out) throws IOException {
            out.writeLong(r.fingerprint());
            out.writeInt(r.parallelism());
            out.writeLong(r.partials());
            out.writeLong(r.snapshots());
            out.writeLong(r.sinceSnapshot());
            out.writeLong(r.peakStateBytes());
            out.writeInt(r.trees().size());
            for (SubtaskTree t : r.trees()) {
                out.writeLong(t.nextSequence());
                out.writeLong(t.records());
                PointCodec.writeF64(t.weight(), out);
                out.writeLong(t.batches());
                PointCodec.writeCoresets(t.levels(), out);
                for (int d : t.depths()) {
                    out.writeInt(d);
                }
            }
        }

        @Override
        public Merge deserialize(DataInputView in) throws IOException {
            long fingerprint = in.readLong();
            int parallelism = in.readInt();
            long partials = in.readLong();
            long snapshots = in.readLong();
            long sinceSnapshot = in.readLong();
            long peak = in.readLong();
            int n = PointCodec.count(in);
            List<SubtaskTree> trees = new ArrayList<>(Math.min(n, 1 << 15));
            for (int i = 0; i < n; i++) {
                long next = in.readLong();
                long records = in.readLong();
                double weight = PointCodec.readF64(in);
                long batches = in.readLong();
                List<Coreset> levels = PointCodec.readCoresets(in);
                List<Integer> depths = new ArrayList<>(levels.size());
                for (int l = 0; l < levels.size(); l++) {
                    depths.add(in.readInt());
                }
                try {
                    trees.add(new SubtaskTree(next, records, weight, batches, levels, depths));
                } catch (IllegalArgumentException e) {
                    throw new IOException(e.getMessage(), e);
                }
            }
            return new Merge(fingerprint, parallelism, partials, snapshots, sinceSnapshot, peak, trees);
        }

        @Override
        public void copy(DataInputView in, DataOutputView out) throws IOException {
            out.write(in, (int) Merge.FIXED_BYTES - 4);
            int n = in.readInt();
            out.writeInt(n);
            for (int i = 0; i < n; i++) {
                out.write(in, (int) SubtaskTree.FIXED_BYTES);
                int levels = in.readInt();
                out.writeInt(levels);
                for (int l = 0; l < levels; l++) {
                    PointCodec.copyCoreset(in, out);
                }
                out.write(in, 4 * levels);
            }
        }

        @Override
        public TypeSerializerSnapshot<Merge> snapshotConfiguration() {
            return new Snapshot();
        }

        private Object readResolve() {
            return INSTANCE;
        }

        /** Snapshot of {@link MergeSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
        public static final class Snapshot extends FormatSnapshot<Merge> {
            public Snapshot() {
                super(() -> INSTANCE);
            }
        }
    }
}
