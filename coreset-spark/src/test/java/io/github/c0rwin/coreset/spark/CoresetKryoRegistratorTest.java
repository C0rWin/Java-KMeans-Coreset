package io.github.c0rwin.coreset.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.nio.ByteBuffer;
import org.apache.spark.SparkConf;
import org.apache.spark.serializer.KryoSerializer;
import org.apache.spark.serializer.SerializerInstance;
import org.junit.jupiter.api.Test;
import scala.reflect.ClassTag;
import scala.reflect.ClassTag$;

class CoresetKryoRegistratorTest {

    private static <T> T roundTrip(SerializerInstance kryo, T value, Class<T> type) {
        ClassTag<T> tag = ClassTag$.MODULE$.apply(type);
        ByteBuffer bytes = kryo.serialize(value, tag);
        return kryo.deserialize(bytes, tag);
    }

    @Test
    void kryoUsesTheCodecForCoreValueTypes() {
        SparkConf conf = new SparkConf(false)
                .set("spark.kryo.registrator", CoresetKryoRegistrator.class.getName())
                .set("spark.kryo.registrationRequired", "true");
        SerializerInstance kryo = new KryoSerializer(conf).newInstance();
        Coreset c = CoresetCodecTest.sparseCoreset(200, 10_000, 30, 3);

        assertEquals(c, roundTrip(kryo, c, Coreset.class));
        WeightedPoint p = c.points().get(7);
        assertEquals(p, roundTrip(kryo, p, WeightedPoint.class));
        assertEquals(p.vector(), roundTrip(kryo, p.vector(), io.github.c0rwin.coreset.core.SparseVector.class));

        int kryoBytes = kryo.serialize(c, ClassTag$.MODULE$.apply(Coreset.class)).remaining();
        int codecBytes = CoresetCodec.encode(c).length;
        System.out.printf("kryo coreset: %d B, codec %d B%n", kryoBytes, codecBytes);
        assertTrue(kryoBytes <= codecBytes + 8, "class id and length prefix only");
    }
}
