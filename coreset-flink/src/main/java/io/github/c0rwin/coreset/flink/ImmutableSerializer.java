package io.github.c0rwin.coreset.flink;

import java.io.IOException;
import org.apache.flink.api.common.typeutils.base.TypeSerializerSingleton;
import org.apache.flink.core.memory.DataInputView;

/**
 * Base of this module's serializers. Every value they handle is immutable, so copies are the value itself: chained
 * operators and the heap state backend's synchronous snapshot phase pass references instead of re-serializing.
 */
public abstract class ImmutableSerializer<T> extends TypeSerializerSingleton<T> {

    private static final long serialVersionUID = 1L;

    @Override
    public final boolean isImmutableType() {
        return true;
    }

    @Override
    public final T copy(T from) {
        return from;
    }

    @Override
    public final T copy(T from, T reuse) {
        return from;
    }

    @Override
    public final int getLength() {
        return -1;
    }

    @Override
    public final T deserialize(T reuse, DataInputView source) throws IOException {
        return deserialize(source);
    }
}
