package io.github.c0rwin.coreset.examples.lightweight;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.spark.BuildResult;
import io.github.c0rwin.coreset.spark.SparkCoresets;
import java.util.List;
import java.util.Map;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Spark local mode: executors resolve the registered id themselves, or run the code-supplied factory. */
class SparkPluginTest {

    private static SparkSession spark;
    private static List<WeightedPoint> points;
    private static JavaRDD<WeightedPoint> rdd;

    @BeforeAll
    static void start() {
        spark = SparkSession.builder()
                .master("local[4]")
                .appName("coreset-examples")
                .config("spark.ui.enabled", "false")
                .config("spark.driver.host", "127.0.0.1")
                .config("spark.driver.bindAddress", "127.0.0.1")
                .getOrCreate();
        points = Blobs.points(40_000, 9);
        rdd = JavaSparkContext.fromSparkContext(spark.sparkContext()).parallelize(points, 8).cache();
    }

    @AfterAll
    static void stop() {
        spark.stop();
    }

    @Test
    void registeredIdAndFactoryBuildTheSameValidCoreset() {
        BuildResult byId = SparkCoresets.build(rdd, "lightweight", Map.of("m", "500"), 23L, 2);
        BuildResult byFactory = SparkCoresets.build(rdd, new LightweightCoresetFactory(500), 23L, 2, 1_000);
        assertEquals(byId.coreset(), byFactory.coreset());

        Coreset full = Coreset.of(points);
        Coreset c = byId.coreset();
        double distortion = Blobs.distortion(full, c);
        double weightErr = Math.abs(c.totalWeight() - full.totalWeight()) / full.totalWeight();
        System.out.printf("lightweight on spark: %d partitions, depth %d, %d points, distortion %.4f, weight rel err"
                + " %.2e%n", byId.metrics().partitions(), byId.metrics().compressionDepth(), c.size(), distortion,
                weightErr);
        assertTrue(c.size() <= 500);
        assertTrue(distortion < LightweightCoresetTest.MAX_STREAM_DISTORTION, "distortion " + distortion);
        assertTrue(weightErr < LightweightCoresetTest.MAX_WEIGHT_REL_ERR, "weight rel err " + weightErr);
    }
}
