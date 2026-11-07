package io.github.c0rwin.coreset.flink;

import java.util.Objects;
import org.apache.flink.api.common.serialization.SerializerConfig;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.api.common.typeutils.TypeSerializer;

/**
 * Type information backed by one of this module's serializers, so Flink never falls back to Kryo for these types.
 * Instances are the constants of {@link CoresetTypes}.
 */
public final class CoresetTypeInfo<T> extends TypeInformation<T> {

    private static final long serialVersionUID = 1L;

    private final Class<T> type;
    private final TypeSerializer<T> serializer;

    CoresetTypeInfo(Class<T> type, TypeSerializer<T> serializer) {
        this.type = Objects.requireNonNull(type, "type");
        this.serializer = Objects.requireNonNull(serializer, "serializer");
    }

    @Override
    public boolean isBasicType() {
        return false;
    }

    @Override
    public boolean isTupleType() {
        return false;
    }

    @Override
    public int getArity() {
        return 1;
    }

    @Override
    public int getTotalFields() {
        return 1;
    }

    @Override
    public Class<T> getTypeClass() {
        return type;
    }

    @Override
    public boolean isKeyType() {
        return false;
    }

    @Override
    public TypeSerializer<T> createSerializer(SerializerConfig config) {
        return serializer;
    }

    @Override
    public String toString() {
        return "CoresetTypeInfo<" + type.getSimpleName() + ">";
    }

    @Override
    public boolean equals(Object obj) {
        return obj instanceof CoresetTypeInfo<?> other && other.canEqual(this) && type == other.type
                && serializer.equals(other.serializer);
    }

    @Override
    public int hashCode() {
        return type.hashCode();
    }

    @Override
    public boolean canEqual(Object obj) {
        return obj instanceof CoresetTypeInfo;
    }
}
