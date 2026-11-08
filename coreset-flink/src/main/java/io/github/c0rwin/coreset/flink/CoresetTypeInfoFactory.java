package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.lang.reflect.Type;
import java.util.Map;
import org.apache.flink.api.common.typeinfo.TypeInfoFactory;
import org.apache.flink.api.common.typeinfo.TypeInformation;

/** Resolves {@link Coreset} to {@link CoresetTypes#CORESET}; see {@link CoresetTypes#serializationConfig()}. */
public final class CoresetTypeInfoFactory extends TypeInfoFactory<Coreset> {

    @Override
    public TypeInformation<Coreset> createTypeInfo(Type t, Map<String, TypeInformation<?>> genericParameters) {
        return CoresetTypes.CORESET;
    }
}
