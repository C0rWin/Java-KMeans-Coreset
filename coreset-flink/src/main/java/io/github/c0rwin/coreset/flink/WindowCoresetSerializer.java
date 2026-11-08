package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.io.IOException;
import java.util.List;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/** {@link WindowCoreset}: fixed-width header, then the coreset in {@link PointCodec} layout. */
public final class WindowCoresetSerializer extends ImmutableSerializer<WindowCoreset> {

    private static final long serialVersionUID = 1L;
    public static final WindowCoresetSerializer INSTANCE = new WindowCoresetSerializer();
    private static final WindowCoreset EMPTY = new WindowCoreset(0, 0, Coreset.of(List.of()), 0, 0.0, 0, 0);

    private WindowCoresetSerializer() {
    }

    @Override
    public WindowCoreset createInstance() {
        return EMPTY;
    }

    @Override
    public void serialize(WindowCoreset r, DataOutputView out) throws IOException {
        out.writeLong(r.windowStart());
        out.writeLong(r.windowEnd());
        out.writeLong(r.records());
        PointCodec.writeF64(r.weight(), out);
        out.writeInt(r.partials());
        out.writeInt(r.depth());
        PointCodec.writeCoreset(r.coreset(), out);
    }

    @Override
    public WindowCoreset deserialize(DataInputView in) throws IOException {
        long start = in.readLong();
        long end = in.readLong();
        long records = in.readLong();
        double weight = PointCodec.readF64(in);
        int partials = in.readInt();
        int depth = in.readInt();
        return new WindowCoreset(start, end, PointCodec.readCoreset(in), records, weight, partials, depth);
    }

    @Override
    public void copy(DataInputView in, DataOutputView out) throws IOException {
        out.write(in, 8 + 8 + 8 + 8 + 4 + 4);
        PointCodec.copyCoreset(in, out);
    }

    @Override
    public TypeSerializerSnapshot<WindowCoreset> snapshotConfiguration() {
        return new Snapshot();
    }

    private Object readResolve() {
        return INSTANCE;
    }

    /** Snapshot of {@link WindowCoresetSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
    public static final class Snapshot extends FormatSnapshot<WindowCoreset> {
        public Snapshot() {
            super(() -> INSTANCE);
        }
    }
}
