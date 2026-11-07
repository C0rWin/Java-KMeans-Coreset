package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.IOException;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/** {@link WeightedPoint} as {@code dim:varint weight:f64 body}; see {@link PointCodec}. */
public final class WeightedPointSerializer extends ImmutableSerializer<WeightedPoint> {

    private static final long serialVersionUID = 1L;
    public static final WeightedPointSerializer INSTANCE = new WeightedPointSerializer();
    private static final WeightedPoint EMPTY = new WeightedPoint(SparseVector.dense(new double[0]), 0.0);

    private WeightedPointSerializer() {
    }

    @Override
    public WeightedPoint createInstance() {
        return EMPTY;
    }

    @Override
    public void serialize(WeightedPoint record, DataOutputView target) throws IOException {
        PointCodec.writePoint(record, target);
    }

    @Override
    public WeightedPoint deserialize(DataInputView source) throws IOException {
        return PointCodec.readPoint(source);
    }

    @Override
    public void copy(DataInputView source, DataOutputView target) throws IOException {
        PointCodec.copyPoint(source, target);
    }

    @Override
    public TypeSerializerSnapshot<WeightedPoint> snapshotConfiguration() {
        return new Snapshot();
    }

    private Object readResolve() {
        return INSTANCE;
    }

    /** Snapshot of {@link WeightedPointSerializer}, format {@value FormatSnapshot#FORMAT_VERSION}. */
    public static final class Snapshot extends FormatSnapshot<WeightedPoint> {
        public Snapshot() {
            super(() -> INSTANCE);
        }
    }
}
