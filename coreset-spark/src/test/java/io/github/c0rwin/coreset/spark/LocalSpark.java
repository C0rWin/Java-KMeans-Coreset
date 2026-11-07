package io.github.c0rwin.coreset.spark;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

/** One local[4] session for all test classes (one SparkContext per JVM); up to 3 attempts per task. */
final class LocalSpark {

    private static SparkSession session;

    private LocalSpark() {
    }

    static synchronized SparkSession session() {
        if (session == null) {
            session = SparkSession.builder()
                    .master("local[4,3]")
                    .appName("coreset-spark-tests")
                    .config("spark.ui.enabled", "false")
                    .config("spark.driver.host", "127.0.0.1")
                    .config("spark.driver.bindAddress", "127.0.0.1")
                    .config("spark.sql.shuffle.partitions", "4")
                    .config("spark.sql.streaming.forceDeleteTempCheckpointLocation", "true")
                    .getOrCreate();
        }
        return session;
    }

    static JavaSparkContext jsc() {
        return JavaSparkContext.fromSparkContext(session().sparkContext());
    }
}
