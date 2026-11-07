package io.github.c0rwin.coreset.spark;

import com.esotericsoftware.kryo.Kryo;
import com.esotericsoftware.kryo.Serializer;
import com.esotericsoftware.kryo.io.Input;
import com.esotericsoftware.kryo.io.Output;
import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.function.Function;
import org.apache.spark.serializer.KryoRegistrator;

/**
 * Registers {@link CoresetCodec}-based Kryo serializers for the core value types. Kryo's default field serializer
 * cannot rebuild records or enforce {@link SparseVector}'s invariants, so set
 * {@code spark.kryo.registrator=io.github.c0rwin.coreset.spark.CoresetKryoRegistrator} when using
 * {@code KryoSerializer} with RDDs of these types.
 */
public final class CoresetKryoRegistrator implements KryoRegistrator {

    @Override
    public void registerClasses(Kryo kryo) {
        kryo.register(SparseVector.class,
                new CodecSerializer<SparseVector>(CoresetCodec::encode, CoresetCodec::decodeVector));
        kryo.register(WeightedPoint.class,
                new CodecSerializer<WeightedPoint>(CoresetCodec::encode, CoresetCodec::decodePoint));
        kryo.register(Coreset.class, new CodecSerializer<Coreset>(CoresetCodec::encode, CoresetCodec::decode));
    }

    private static final class CodecSerializer<T> extends Serializer<T> {
        private final Function<T, byte[]> encode;
        private final Function<byte[], T> decode;

        CodecSerializer(Function<T, byte[]> encode, Function<byte[], T> decode) {
            super(false, true); // values are immutable: no null handling, copies may share
            this.encode = encode;
            this.decode = decode;
        }

        @Override
        public void write(Kryo kryo, Output output, T value) {
            byte[] bytes = encode.apply(value);
            output.writeInt(bytes.length, true);
            output.writeBytes(bytes);
        }

        @Override
        public T read(Kryo kryo, Input input, Class<T> type) {
            return decode.apply(input.readBytes(input.readInt(true)));
        }

        @Override
        public T copy(Kryo kryo, T original) {
            return original;
        }
    }
}
