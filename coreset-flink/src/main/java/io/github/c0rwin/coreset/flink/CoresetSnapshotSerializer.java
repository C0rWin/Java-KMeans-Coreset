package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.io.IOException;
import java.util.List;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/** {@link CoresetSnapshot}: fixed-width header, the coreset, then an optional compressed coreset. */
public final class CoresetSnapshotSerializer extends ImmutableSerializer<CoresetSnapshot> {

    private static final long serialVersionUID = 1L;
    public static final CoresetSnapshotSerializer INSTANCE = new CoresetSnapshotSerializer();
    private static final CoresetSnapshot EMPTY =
            new CoresetSnapshot(Coreset.of(List.of()), null, 0, 0.0, 0, 0, 0, false);

    private CoresetSnapshotSerializer() {
    }

    @Override
    public CoresetSnapshot createInstance() {
        return EMPTY;
    }

    @Override
    public void serialize(CoresetSnapshot r, DataOutputView out) throws IOException {
        out.writeLong(r.records());
        PointCodec.writeF64(r.weight(), out);
        out.writeLong(r.partials());
        out.writeInt(r.levels());
        out.writeInt(r.depth());
        out.writeBoolean(r.last());
        PointCodec.writeCoreset(r.coreset(), out);
        out.writeBoolean(r.compressed() != null);
        if (r.compressed() != null) {
            PointCodec.writeCoreset(r.compressed(), out);
        }
    }

    @Override
    public CoresetSnapshot deserialize(DataInputView in) throws IOException {
        long records = in.readLong();
        double weight = PointCodec.readF64(in);
        long partials = in.readLong();
        int levels = in.readInt();
        int depth = in.readInt();
        boolean last = in.readBoolean();
        Coreset coreset = PointCodec.readCoreset(in);
        Coreset compressed = in.readBoolean() ? PointCodec.readCoreset(in) : null;
        return new CoresetSnapshot(coreset, compressed, records, weight, partials, levels, depth, last);
    }

    @Override
    public void copy(DataInputView in, DataOutputView out) throws IOException {
        out.write(in, 8 + 8 + 8 + 4 + 4 + 1);
        PointCodec.copyCoreset(in, out);
        boolean compressed = in.readBoolean();
        out.writeBoolean(compressed);
        if (compressed) {
            PointCodec.copyCoreset(in, out);
        }
    }

    @Override
    public TypeSerializerSnapshot<CoresetSnapshot> snapshotConfiguration() {
        return new Snapshot();
    }

    private Object readResolve() {
        return INSTANCE;
    }

    /** Snapshot of {@link CoresetSnapshotSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
    public static final class Snapshot extends FormatSnapshot<CoresetSnapshot> {
        public Snapshot() {
            super(() -> INSTANCE);
        }
    }
}
