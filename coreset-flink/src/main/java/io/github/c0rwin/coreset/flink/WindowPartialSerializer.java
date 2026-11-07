package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.io.IOException;
import java.util.List;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/** {@link WindowPartial}: fixed-width header, then the coreset in {@link PointCodec} layout. */
public final class WindowPartialSerializer extends ImmutableSerializer<WindowPartial> {

    private static final long serialVersionUID = 1L;
    public static final WindowPartialSerializer INSTANCE = new WindowPartialSerializer();
    private static final WindowPartial EMPTY = new WindowPartial(0, 0, Coreset.of(List.of()), 0, 0.0, 0);

    private WindowPartialSerializer() {
    }

    @Override
    public WindowPartial createInstance() {
        return EMPTY;
    }

    @Override
    public void serialize(WindowPartial r, DataOutputView out) throws IOException {
        out.writeLong(r.windowStart());
        out.writeInt(r.subtask());
        out.writeLong(r.records());
        PointCodec.writeF64(r.weight(), out);
        out.writeInt(r.depth());
        PointCodec.writeCoreset(r.coreset(), out);
    }

    @Override
    public WindowPartial deserialize(DataInputView in) throws IOException {
        long start = in.readLong();
        int subtask = in.readInt();
        long records = in.readLong();
        double weight = PointCodec.readF64(in);
        int depth = in.readInt();
        return new WindowPartial(start, subtask, PointCodec.readCoreset(in), records, weight, depth);
    }

    @Override
    public void copy(DataInputView in, DataOutputView out) throws IOException {
        out.write(in, 8 + 4 + 8 + 8 + 4);
        PointCodec.copyCoreset(in, out);
    }

    @Override
    public TypeSerializerSnapshot<WindowPartial> snapshotConfiguration() {
        return new Snapshot();
    }

    private Object readResolve() {
        return INSTANCE;
    }

    /** Snapshot of {@link WindowPartialSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
    public static final class Snapshot extends FormatSnapshot<WindowPartial> {
        public Snapshot() {
            super(() -> INSTANCE);
        }
    }
}
