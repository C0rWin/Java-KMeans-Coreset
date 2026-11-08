package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.io.IOException;
import java.util.List;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/** {@link PartialCoreset}: fixed-width header, then the coreset in {@link PointCodec} layout. */
public final class PartialCoresetSerializer extends ImmutableSerializer<PartialCoreset> {

    private static final long serialVersionUID = 1L;
    public static final PartialCoresetSerializer INSTANCE = new PartialCoresetSerializer();
    private static final PartialCoreset EMPTY = new PartialCoreset(0, 1, 0, Coreset.of(List.of()), 0, 0.0, 0);

    private PartialCoresetSerializer() {
    }

    @Override
    public PartialCoreset createInstance() {
        return EMPTY;
    }

    @Override
    public void serialize(PartialCoreset r, DataOutputView out) throws IOException {
        out.writeInt(r.subtask());
        out.writeInt(r.parallelism());
        out.writeLong(r.sequence());
        out.writeLong(r.records());
        PointCodec.writeF64(r.weight(), out);
        out.writeInt(r.depth());
        PointCodec.writeCoreset(r.coreset(), out);
    }

    @Override
    public PartialCoreset deserialize(DataInputView in) throws IOException {
        int subtask = in.readInt();
        int parallelism = in.readInt();
        long sequence = in.readLong();
        long records = in.readLong();
        double weight = PointCodec.readF64(in);
        int depth = in.readInt();
        return new PartialCoreset(subtask, parallelism, sequence, PointCodec.readCoreset(in), records, weight, depth);
    }

    @Override
    public void copy(DataInputView in, DataOutputView out) throws IOException {
        out.write(in, 4 + 4 + 8 + 8 + 8 + 4);
        PointCodec.copyCoreset(in, out);
    }

    @Override
    public TypeSerializerSnapshot<PartialCoreset> snapshotConfiguration() {
        return new Snapshot();
    }

    private Object readResolve() {
        return INSTANCE;
    }

    /** Snapshot of {@link PartialCoresetSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
    public static final class Snapshot extends FormatSnapshot<PartialCoreset> {
        public Snapshot() {
            super(() -> INSTANCE);
        }
    }
}
