package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.List;
import org.apache.flink.api.common.typeinfo.TypeInformation;
import org.apache.flink.configuration.PipelineOptions;

/**
 * Kryo-free type information for the values this module reads and writes. Declare the input stream with
 * {@code .returns(CoresetTypes.POINT)} (or pass it to {@code fromSource}); for user types holding points or coresets,
 * register the factories with {@link #serializationConfig()}.
 */
public final class CoresetTypes {

    public static final TypeInformation<WeightedPoint> POINT =
            new CoresetTypeInfo<>(WeightedPoint.class, WeightedPointSerializer.INSTANCE);
    public static final TypeInformation<Coreset> CORESET =
            new CoresetTypeInfo<>(Coreset.class, CoresetSerializer.INSTANCE);
    public static final TypeInformation<PartialCoreset> PARTIAL =
            new CoresetTypeInfo<>(PartialCoreset.class, PartialCoresetSerializer.INSTANCE);
    public static final TypeInformation<CoresetSnapshot> SNAPSHOT =
            new CoresetTypeInfo<>(CoresetSnapshot.class, CoresetSnapshotSerializer.INSTANCE);
    public static final TypeInformation<WindowPartial> WINDOW_PARTIAL =
            new CoresetTypeInfo<>(WindowPartial.class, WindowPartialSerializer.INSTANCE);
    public static final TypeInformation<WindowCoreset> WINDOW_CORESET =
            new CoresetTypeInfo<>(WindowCoreset.class, WindowCoresetSerializer.INSTANCE);

    private CoresetTypes() {
    }

    /**
     * Value for {@link PipelineOptions#SERIALIZATION_CONFIG} ({@code pipeline.serialization-config}) that makes type
     * extraction use {@link #POINT} and {@link #CORESET} wherever a {@link WeightedPoint} or {@link Coreset} appears,
     * e.g. as a field of a user POJO. Flink registers these factories JVM-wide on first use.
     */
    public static List<String> serializationConfig() {
        return List.of(
                WeightedPoint.class.getName() + ": {type: typeinfo, class: "
                        + WeightedPointTypeInfoFactory.class.getName() + "}",
                Coreset.class.getName() + ": {type: typeinfo, class: " + CoresetTypeInfoFactory.class.getName() + "}");
    }
}
