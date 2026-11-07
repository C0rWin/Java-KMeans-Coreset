package io.github.c0rwin.coreset.flink;

import java.io.IOException;
import java.util.function.Supplier;
import org.apache.flink.api.common.typeutils.TypeSerializer;
import org.apache.flink.api.common.typeutils.TypeSerializerSchemaCompatibility;
import org.apache.flink.api.common.typeutils.TypeSerializerSnapshot;
import org.apache.flink.core.memory.DataInputView;
import org.apache.flink.core.memory.DataOutputView;

/**
 * Snapshot of a stateless serializer whose snapshot version is the version of its binary format. Format
 * {@value #FORMAT_VERSION} is the only one so far; a state written by a newer format fails to restore with a clear
 * error instead of being misread. A future format change bumps the version and maps older versions to a migration.
 */
public abstract class FormatSnapshot<T> implements TypeSerializerSnapshot<T> {

    /** Binary format written by the serializers of this module. */
    public static final int FORMAT_VERSION = 1;

    private final Supplier<? extends TypeSerializer<T>> serializer;
    private int readVersion = FORMAT_VERSION;

    protected FormatSnapshot(Supplier<? extends TypeSerializer<T>> serializer) {
        this.serializer = serializer;
    }

    @Override
    public int getCurrentVersion() {
        return FORMAT_VERSION;
    }

    /** Format version this snapshot was read with ({@link #FORMAT_VERSION} for a freshly taken one). */
    public int readVersion() {
        return readVersion;
    }

    @Override
    public void writeSnapshot(DataOutputView out) {
        // The format version, written by Flink in front of the snapshot, is all the configuration there is.
    }

    @Override
    public void readSnapshot(int version, DataInputView in, ClassLoader classLoader) throws IOException {
        if (version < 1 || version > FORMAT_VERSION) {
            throw new IOException(getClass().getName() + ": state was written with format version " + version
                    + ", this build reads up to " + FORMAT_VERSION);
        }
        readVersion = version;
    }

    @Override
    public TypeSerializer<T> restoreSerializer() {
        return serializer.get();
    }

    @Override
    public TypeSerializerSchemaCompatibility<T> resolveSchemaCompatibility(TypeSerializerSnapshot<T> oldSnapshot) {
        if (oldSnapshot.getClass() != getClass()) {
            return TypeSerializerSchemaCompatibility.incompatible();
        }
        return ((FormatSnapshot<T>) oldSnapshot).readVersion == FORMAT_VERSION
                ? TypeSerializerSchemaCompatibility.compatibleAsIs()
                : TypeSerializerSchemaCompatibility.incompatible();
    }

    @Override
    public boolean equals(Object obj) {
        return obj != null && obj.getClass() == getClass()
                && ((FormatSnapshot<?>) obj).readVersion == readVersion;
    }

    @Override
    public int hashCode() {
        return getClass().hashCode() * 31 + readVersion;
    }

    @Override
    public String toString() {
        return getClass().getName() + "{format " + readVersion + "}";
    }
}
