package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.WeightedPoint;
import java.lang.reflect.Type;
import java.util.Map;
import org.apache.flink.api.common.typeinfo.TypeInfoFactory;
import org.apache.flink.api.common.typeinfo.TypeInformation;

/** Resolves {@link WeightedPoint} to {@link CoresetTypes#POINT}; see {@link CoresetTypes#serializationConfig()}. */
public final class WeightedPointTypeInfoFactory extends TypeInfoFactory<WeightedPoint> {

    @Override
    public TypeInformation<WeightedPoint> createTypeInfo(Type t, Map<String, TypeInformation<?>> genericParameters) {
        return CoresetTypes.POINT;
    }
}
