package io.github.c0rwin.coreset.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.IOException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import org.apache.flink.api.common.serialization.SerializerConfigImpl;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.SerializerTestBase;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSchemaCompatibility;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.api.java.typeutils.TypeExtractor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.PipelineOptions;
import org.apache.flink.core.memory.DataInputDeserializer;
import org.apache.flink.core.memory.DataOutputSerializer;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Serializer contracts: Flink's {@link SerializerTestBase} battery (round trips, copies, duplicates, Java
 * serializability, snapshot write/read/restore and compatibility) for every type, plus golden bytes that pin format
 * version 1, exact {@code encodedLength}s, and rejection of corrupt input.
 */
class SerializersTest {

    static final SparseVector SPARSE =
            SparseVector.of(1000, new int[] {3, 130, 999}, new double[] {1.5, -2.25, 1e-300});
    static final SparseVector DENSE = SparseVector.dense(new double[] {0.1, -0.2, 0.0, 4.0});
    static final Coreset SMALL = new Coreset(List.of(new WeightedPoint(SPARSE, 2.0),
            new WeightedPoint(SparseVector.of(1000, new int[0], new double[0]), 0.0)), 3.5);
    static final Coreset DENSE_CORESET = Coreset.of(List.of(new WeightedPoint(DENSE, 1.0),
            new WeightedPoint(SparseVector.dense(new double[] {1, 2, 3, 4}), 0.75)));
    static final Coreset EMPTY = new Coreset(List.of(), 1.25);

    static byte[] bytes(TypeSerializer<Object> s, Object value) throws IOException {
        DataOutputSerializer out = new DataOutputSerializer(64);
        s.serialize(value, out);
        return out.getCopyOfBuffer();
    }

    @SuppressWarnings("unchecked")
    static <T> TypeSerializer<Object> erased(TypeSerializer<T> s) {
        return (TypeSerializer<Object>) (TypeSerializer<?>) s;
    }

    @Nested
    class Points extends SerializerTestBase<WeightedPoint> {
        @Override
        protected TypeSerializer<WeightedPoint> createSerializer() {
            return WeightedPointSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<WeightedPoint> getTypeClass() {
            return WeightedPoint.class;
        }

        @Override
        protected WeightedPoint[] getTestData() {
            return new WeightedPoint[] {new WeightedPoint(SPARSE, 0.5), new WeightedPoint(DENSE, 0.0),
                new WeightedPoint(SparseVector.dense(new double[0]), 7.0),
                new WeightedPoint(SparseVector.of(Integer.MAX_VALUE, new int[] {Integer.MAX_VALUE - 1},
                        new double[] {Double.MIN_VALUE}), Double.MAX_VALUE)};
        }
    }

    @Nested
    class Coresets extends SerializerTestBase<Coreset> {
        @Override
        protected TypeSerializer<Coreset> createSerializer() {
            return CoresetSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<Coreset> getTypeClass() {
            return Coreset.class;
        }

        @Override
        protected Coreset[] getTestData() {
            return new Coreset[] {SMALL, DENSE_CORESET, EMPTY, Coreset.of(List.of())};
        }
    }

    @Nested
    class Partials extends SerializerTestBase<PartialCoreset> {
        @Override
        protected TypeSerializer<PartialCoreset> createSerializer() {
            return PartialCoresetSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<PartialCoreset> getTypeClass() {
            return PartialCoreset.class;
        }

        @Override
        protected PartialCoreset[] getTestData() {
            return new PartialCoreset[] {new PartialCoreset(3, 4, 17, SMALL, 1000, 1234.5, 5),
                new PartialCoreset(0, 1, 0, EMPTY, 0, 0.0, 0)};
        }
    }

    @Nested
    class Snapshots extends SerializerTestBase<CoresetSnapshot> {
        @Override
        protected TypeSerializer<CoresetSnapshot> createSerializer() {
            return CoresetSnapshotSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<CoresetSnapshot> getTypeClass() {
            return CoresetSnapshot.class;
        }

        @Override
        protected CoresetSnapshot[] getTestData() {
            return new CoresetSnapshot[] {new CoresetSnapshot(SMALL, DENSE_CORESET, 10, 11.5, 3, 2, 4, true),
                new CoresetSnapshot(EMPTY, null, 0, 0.0, 0, 0, 0, false)};
        }
    }

    @Nested
    class WindowPartials extends SerializerTestBase<WindowPartial> {
        @Override
        protected TypeSerializer<WindowPartial> createSerializer() {
            return WindowPartialSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<WindowPartial> getTypeClass() {
            return WindowPartial.class;
        }

        @Override
        protected WindowPartial[] getTestData() {
            return new WindowPartial[] {new WindowPartial(-60_000, 2, SMALL, 5, 6.0, 1),
                new WindowPartial(Long.MAX_VALUE - 10, 0, DENSE_CORESET, 2, 1.75, 0)};
        }
    }

    @Nested
    class WindowCoresets extends SerializerTestBase<WindowCoreset> {
        @Override
        protected TypeSerializer<WindowCoreset> createSerializer() {
            return WindowCoresetSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<WindowCoreset> getTypeClass() {
            return WindowCoreset.class;
        }

        @Override
        protected WindowCoreset[] getTestData() {
            return new WindowCoreset[] {new WindowCoreset(0, 1000, SMALL, 40, 41.0, 4, 3),
                new WindowCoreset(1000, 2000, EMPTY, 0, 0.0, 0, 0)};
        }
    }

    static OperatorStates.Builder builder() {
        return new OperatorStates.Builder(16, 5, List.of(SMALL, DENSE_CORESET),
                List.of(new WeightedPoint(DENSE, 1.0), new WeightedPoint(SPARSE, 3.0)), 77, 78.5);
    }

    @Nested
    class LocalStates extends SerializerTestBase<OperatorStates.Local> {
        @Override
        protected TypeSerializer<OperatorStates.Local> createSerializer() {
            return OperatorStates.LocalSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<OperatorStates.Local> getTypeClass() {
            return OperatorStates.Local.class;
        }

        @Override
        protected OperatorStates.Local[] getTestData() {
            return new OperatorStates.Local[] {
                new OperatorStates.Local(42L, 1, 4, 9, 1000, 3000, 12345, builder()),
                new OperatorStates.Local(-1L, 0, 1, 0, 0, Long.MIN_VALUE, 0,
                        new OperatorStates.Builder(0, 0, List.of(), List.of(), 0, 0.0))};
        }
    }

    @Nested
    class WindowStates extends SerializerTestBase<OperatorStates.Window> {
        @Override
        protected TypeSerializer<OperatorStates.Window> createSerializer() {
            return OperatorStates.WindowSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<OperatorStates.Window> getTypeClass() {
            return OperatorStates.Window.class;
        }

        @Override
        protected OperatorStates.Window[] getTestData() {
            return new OperatorStates.Window[] {
                new OperatorStates.Window(7L, 2, 4, 1999, 500, 3, 9, 4096, List.of(
                        new OperatorStates.OpenWindow(0, builder()),
                        new OperatorStates.OpenWindow(1000, new OperatorStates.Builder(0, 1, List.of(EMPTY),
                                List.of(), 1, 1.0)))),
                new OperatorStates.Window(7L, 0, 1, Long.MIN_VALUE, 0, 0, 0, 0, List.of())};
        }
    }

    @Nested
    class MergeStates extends SerializerTestBase<OperatorStates.Merge> {
        @Override
        protected TypeSerializer<OperatorStates.Merge> createSerializer() {
            return OperatorStates.MergeSerializer.INSTANCE;
        }

        @Override
        protected int getLength() {
            return -1;
        }

        @Override
        protected Class<OperatorStates.Merge> getTypeClass() {
            return OperatorStates.Merge.class;
        }

        @Override
        protected OperatorStates.Merge[] getTestData() {
            return new OperatorStates.Merge[] {
                new OperatorStates.Merge(5L, 2, 7, 3, 1, 8192, List.of(
                        new OperatorStates.SubtaskTree(4, 100, 101.5, 3, List.of(SMALL, DENSE_CORESET),
                                List.of(2, 5)),
                        OperatorStates.SubtaskTree.EMPTY)),
                new OperatorStates.Merge(5L, 0, 0, 0, 0, 0, List.of())};
        }
    }

    /**
     * Pins format version 1 (a change needs a new snapshot version and a migration path). The coreset bytes are
     * exactly what coreset-spark's CoresetCodec writes for the same coreset.
     */
    @Test
    void goldenBytesOfVersionOne() throws IOException {
        assertEquals("0000000000000000" + "01" + "04" + "000000000000f03f"
                        + "03" + "00" + "01" + "02"
                        + "9a9999999999b93f" + "9a9999999999c9bf" + "0000000000001040",
                HexFormat.of().formatHex(bytes(erased(CoresetSerializer.INSTANCE),
                        Coreset.of(List.of(new WeightedPoint(DENSE, 1.0))))));
        assertEquals("e807" + "0000000000000040" + "03" + "03" + "7f" + "e506"
                        + "000000000000f83f" + "00000000000002c0" + "59f3f8c21f6ea501",
                HexFormat.of().formatHex(bytes(erased(WeightedPointSerializer.INSTANCE),
                        new WeightedPoint(SPARSE, 2.0))));
    }

    @Test
    void encodedLengthsAreExact() throws IOException {
        for (Coreset c : List.of(SMALL, DENSE_CORESET, EMPTY)) {
            assertEquals(bytes(erased(CoresetSerializer.INSTANCE), c).length, CoresetSerializer.encodedLength(c));
        }
        for (OperatorStates.Local s : new LocalStates().getTestData()) {
            assertEquals(bytes(erased(OperatorStates.LocalSerializer.INSTANCE), s).length,
                    OperatorStates.encodedLength(s));
        }
        for (OperatorStates.Window s : new WindowStates().getTestData()) {
            assertEquals(bytes(erased(OperatorStates.WindowSerializer.INSTANCE), s).length,
                    OperatorStates.encodedLength(s));
        }
        for (OperatorStates.Merge s : new MergeStates().getTestData()) {
            assertEquals(bytes(erased(OperatorStates.MergeSerializer.INSTANCE), s).length,
                    OperatorStates.encodedLength(s));
        }
    }

    @Test
    void snapshotsRoundTripAndStayCompatible() throws IOException {
        for (TypeSerializer<?> s : List.of(WeightedPointSerializer.INSTANCE, CoresetSerializer.INSTANCE,
                PartialCoresetSerializer.INSTANCE, CoresetSnapshotSerializer.INSTANCE, WindowPartialSerializer.INSTANCE,
                WindowCoresetSerializer.INSTANCE, OperatorStates.LocalSerializer.INSTANCE,
                OperatorStates.WindowSerializer.INSTANCE, OperatorStates.MergeSerializer.INSTANCE)) {
            assertSnapshotCompatible(s);
        }
    }

    @SuppressWarnings("unchecked")
    private static <T> void assertSnapshotCompatible(TypeSerializer<T> serializer) throws IOException {
        DataOutputSerializer out = new DataOutputSerializer(64);
        TypeSerializerSnapshot.writeVersionedSnapshot(out, serializer.snapshotConfiguration());
        TypeSerializerSnapshot<T> restored = TypeSerializerSnapshot.readVersionedSnapshot(
                new DataInputDeserializer(out.getCopyOfBuffer()), SerializersTest.class.getClassLoader());
        assertEquals(1, ((FormatSnapshot<T>) restored).readVersion());
        assertEquals(serializer.snapshotConfiguration(), restored);
        assertSame(serializer, restored.restoreSerializer());
        TypeSerializerSchemaCompatibility<T> compat =
                serializer.snapshotConfiguration().resolveSchemaCompatibility(restored);
        assertTrue(compat.isCompatibleAsIs(), serializer + ": " + compat);
    }

    @Test
    @SuppressWarnings({"unchecked", "rawtypes"})
    void snapshotsRejectFutureFormatsAndForeignSerializers() {
        TypeSerializerSnapshot<Coreset> snapshot = CoresetSerializer.INSTANCE.snapshotConfiguration();
        IOException e = assertThrows(IOException.class, () -> snapshot.readSnapshot(FormatSnapshot.FORMAT_VERSION + 1,
                new DataInputDeserializer(new byte[0]), SerializersTest.class.getClassLoader()));
        assertTrue(e.getMessage().contains("format version 2"), e.getMessage());
        TypeSerializerSnapshot foreign = WeightedPointSerializer.INSTANCE.snapshotConfiguration();
        assertTrue(snapshot.resolveSchemaCompatibility(foreign).isIncompatible());
    }

    @Test
    void corruptInputFailsWithIOException() throws IOException {
        byte[] good = bytes(erased(CoresetSerializer.INSTANCE), SMALL);
        byte[] truncated = Arrays.copyOf(good, good.length - 3);
        assertThrows(IOException.class,
                () -> CoresetSerializer.INSTANCE.deserialize(new DataInputDeserializer(truncated)));
        // Layout: dim(1) weight(8) nnz(1) delta(1) delta(1); a zero second delta repeats an index.
        byte[] unordered = bytes(erased(WeightedPointSerializer.INSTANCE),
                new WeightedPoint(SparseVector.of(10, new int[] {1, 2}, new double[] {1, 1}), 1.0));
        unordered[1 + 8 + 1 + 1] = 0; // second delta 0 -> repeated index
        assertThrows(IOException.class,
                () -> WeightedPointSerializer.INSTANCE.deserialize(new DataInputDeserializer(unordered)));
        byte[] negativeWeight = bytes(erased(WeightedPointSerializer.INSTANCE), new WeightedPoint(DENSE, 1.0));
        negativeWeight[8] = (byte) 0xBF; // sign byte of the little-endian weight: 1.0 -> -1.0
        assertThrows(IOException.class,
                () -> WeightedPointSerializer.INSTANCE.deserialize(new DataInputDeserializer(negativeWeight)));
    }

    @Test
    void typeInfoIsKryoFreeAndFactoriesResolveCoreTypes() {
        assertTrue(CoresetTypes.POINT.createSerializer(new SerializerConfigImpl()) instanceof WeightedPointSerializer);
        assertEquals(CoresetTypes.POINT, new WeightedPointTypeInfoFactory().createTypeInfo(WeightedPoint.class,
                Map.of()));
        Configuration conf = new Configuration();
        conf.set(PipelineOptions.SERIALIZATION_CONFIG, CoresetTypes.serializationConfig());
        SerializerConfigImpl config = new SerializerConfigImpl();
        config.configure(conf, SerializersTest.class.getClassLoader());
        assertEquals(Map.of(WeightedPoint.class, WeightedPointTypeInfoFactory.class,
                Coreset.class, CoresetTypeInfoFactory.class), config.getRegisteredTypeInfoFactories());
        TypeInformation<Coreset> extracted = TypeExtractor.createTypeInfo(Coreset.class);
        assertEquals(CoresetTypes.CORESET, extracted);
        assertEquals(CoresetTypes.POINT, TypeExtractor.createTypeInfo(WeightedPoint.class));
    }
}
