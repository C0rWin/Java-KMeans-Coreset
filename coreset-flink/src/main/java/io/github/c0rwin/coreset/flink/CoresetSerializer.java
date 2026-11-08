package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.io.IOException;
import java.util.List;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/** {@link Coreset} as {@code offset:f64 n:varint [dim:varint (weight:f64 body){n}]}; see {@link PointCodec}. */
public final class CoresetSerializer extends ImmutableSerializer<Coreset> {

    private static final long serialVersionUID = 1L;
    public static final CoresetSerializer INSTANCE = new CoresetSerializer();
    private static final Coreset EMPTY = Coreset.of(List.of());

    private CoresetSerializer() {
    }

    /** Exact number of bytes {@link #serialize} writes for {@code coreset}, from one read of its index arrays. */
    public static long encodedLength(Coreset coreset) {
        return PointCodec.coresetLength(coreset);
    }

    @Override
    public Coreset createInstance() {
        return EMPTY;
    }

    @Override
    public void serialize(Coreset record, DataOutputView target) throws IOException {
        PointCodec.writeCoreset(record, target);
    }

    @Override
    public Coreset deserialize(DataInputView source) throws IOException {
        return PointCodec.readCoreset(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        PointCodec.copyCoreset(source, target);
    }

    @Override
    public TypeSerializerSnapshot<Coreset> snapshotConfiguration() {
        return new Snapshot();
    }

    private Object readResolve() {
        return INSTANCE;
    }

    /** Snapshot of {@link CoresetSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
    public static final class Snapshot extends FormatSnapshot<Coreset> {
        public Snapshot() {
            super(() -> INSTANCE);
        }
    }
}
