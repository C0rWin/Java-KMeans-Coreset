package io.github.c0rwin.coreset.spark;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.core.validate.IntegrityCheck;
import java.io.IOException;
import java.io.Serializable;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.stream.Stream;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.execution.streaming.MemoryStream;
import org.apache.spark.sql.streaming.StreamingQuery;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import scala.jdk.javaapi.CollectionConverters;

class CoresetStreamSinkTest {

    private static final int K = 4;
    private static final int BATCHES = 6;
    private static final int BATCH_POINTS = 3_000;
    private static final long SEED = 21;
    private static final Map<String, String> PARAMS = Map.of("k", "4", "m", "150");
    private static final PointColumns COLUMNS = PointColumns.vector("features").withWeight("weight");
    /** Max empirical distortion of the streamed sensitivity(k=4, m=150) snapshot over 6 batches (measured: 1.14). */
    private static final double MAX_DISTORTION = 1.20;

    private static SparkSession spark;
    private static List<List<PointRow>> batches;

    /** Bean for the memory source: features array<double>, weight double. */
    public static final class PointRow implements Serializable {
        private double[] features;
        private double weight;

        public PointRow() {
        }

        PointRow(WeightedPoint p) {
            features = p.vector().toDense();
            weight = p.weight();
        }

        public double[] getFeatures() {
            return features;
        }

        public void setFeatures(double[] features) {
            this.features = features;
        }

        public double getWeight() {
            return weight;
        }

        public void setWeight(double weight) {
            this.weight = weight;
        }
    }

    @BeforeAll
    static void setUp() {
        spark = LocalSpark.session();
        double[][] centers = Mixture.centers(K, 5, 10.0, SEED);
        batches = new ArrayList<>();
        for (int b = 0; b < BATCHES; b++) {
            List<PointRow> rows = new ArrayList<>();
            for (WeightedPoint p : Mixture.points(centers, BATCH_POINTS, SEED * 100 + b, true)) {
                rows.add(new PointRow(p));
            }
            batches.add(rows);
        }
    }

    private static MemoryStream<PointRow> source() {
        return MemoryStream.apply(4, Encoders.bean(PointRow.class), spark.sqlContext());
    }

    private static StreamingQuery start(MemoryStream<PointRow> source, CoresetStreamSink sink, Path queryCheckpoint)
            throws Exception {
        return source.toDF().writeStream()
                .option("checkpointLocation", queryCheckpoint.toString())
                .foreachBatch(sink)
                .start();
    }

    private static void feed(MemoryStream<PointRow> source, StreamingQuery query, int from, int to) {
        for (int b = from; b < to; b++) {
            source.addData(CollectionConverters.asScala(batches.get(b)));
            query.processAllAvailable();
        }
    }

    private static CoresetStreamSink open(Path dir) throws IOException {
        return CoresetStreamSink.open(spark, dir.toString(), COLUMNS, "sensitivity", PARAMS, SEED, 2);
    }

    private static Coreset fullData() {
        List<WeightedPoint> all = new ArrayList<>();
        for (List<PointRow> rows : batches) {
            for (PointRow r : rows) {
                all.add(new WeightedPoint(io.github.c0rwin.coreset.core.SparseVector.dense(r.features), r.weight));
            }
        }
        return Coreset.of(all);
    }

    private static long stateFiles(Path dir) throws IOException {
        try (Stream<Path> files = Files.list(dir)) {
            return files.filter(p -> p.getFileName().toString().matches("state-\\d+\\.bin")).count();
        }
    }

    @Test
    void restartedQueryResumesExactlyAndReplaysAreIgnored(@TempDir Path tmp) throws Exception {
        // Reference: one uninterrupted query.
        CoresetStreamSink reference = open(tmp.resolve("ref-sink"));
        assertTrue(reference.snapshot().isEmpty());
        assertNull(reference.lastBuildMetrics());
        MemoryStream<PointRow> refSource = source();
        StreamingQuery refQuery = start(refSource, reference, tmp.resolve("ref-query"));
        feed(refSource, refQuery, 0, BATCHES);
        refQuery.stop();
        assertEquals(BATCHES - 1, reference.lastBatchId());
        assertEquals(BATCHES, reference.batchesAdded());

        // Same stream, stopped after 3 batches; the restarted query gets a fresh sink opened from the checkpoint.
        Path sinkDir = tmp.resolve("sink");
        Path queryDir = tmp.resolve("query");
        CoresetStreamSink first = open(sinkDir);
        MemoryStream<PointRow> src = source();
        StreamingQuery q1 = start(src, first, queryDir);
        feed(src, q1, 0, 3);
        q1.stop();
        Coreset beforeRestart = first.snapshot();

        CoresetStreamSink resumed = open(sinkDir);
        assertEquals(2, resumed.lastBatchId());
        assertEquals(beforeRestart, resumed.snapshot());
        StreamingQuery q2 = start(src, resumed, queryDir);
        feed(src, q2, 3, BATCHES);
        q2.stop();

        assertEquals(BATCHES - 1, resumed.lastBatchId());
        assertEquals(reference.snapshot(), resumed.snapshot());
        assertEquals(reference.compressedSnapshot(), resumed.compressedSnapshot());
        assertEquals(0, resumed.skippedBatches());
        assertEquals(1 + StreamCheckpoint.RETAIN, stateFiles(sinkDir));

        // A batch replayed after its state was checkpointed (crash before Spark's commit) is skipped.
        Dataset<Row> replay = spark.createDataset(batches.get(4), Encoders.bean(PointRow.class)).toDF();
        CoresetStreamSink afterCrash = open(sinkDir);
        assertFalse(afterCrash.apply(replay, 4));
        assertFalse(afterCrash.apply(replay, BATCHES - 1));
        assertEquals(2, afterCrash.skippedBatches());
        assertEquals(reference.snapshot(), afterCrash.snapshot());

        // The streamed coreset is a valid summary of everything seen.
        Coreset full = fullData();
        Coreset summary = resumed.snapshot();
        Distortion.Report report = Distortion.evaluate(full, summary,
                CandidateSolutions.standard(full, summary, K, 10, 2, new SplittableRandom(5)));
        double weightErr = IntegrityCheck.check(full, summary).weightRelErr();
        System.out.printf("streamed coreset: %d points from %d batches, distortion max %.4f mean %.4f,"
                        + " weight rel err %.2e, last batch metrics %s%n", summary.size(), BATCHES, report.max(),
                report.mean(), weightErr, reference.lastBuildMetrics());
        assertTrue(report.max() < MAX_DISTORTION, "distortion " + report.max());
        assertTrue(weightErr < 0.05, "weight rel err " + weightErr);
    }

    @Test
    void checkpointRejectsChangedConfigurationAndCorruption(@TempDir Path tmp) throws Exception {
        Path dir = tmp.resolve("sink");
        CoresetStreamSink sink = open(dir);
        Dataset<Row> batch = spark.createDataset(batches.get(0), Encoders.bean(PointRow.class)).toDF();
        assertTrue(sink.apply(batch, 0));
        assertTrue(sink.apply(batch, 7)); // batchIds need not be contiguous
        assertEquals(7, sink.lastBatchId());

        assertThrows(IllegalStateException.class,
                () -> CoresetStreamSink.open(spark, dir.toString(), COLUMNS, "sensitivity", PARAMS, SEED + 1, 2));
        assertThrows(IllegalStateException.class, () -> CoresetStreamSink.open(spark, dir.toString(), COLUMNS,
                "sensitivity", Map.of("k", "4", "m", "151"), SEED, 2));

        byte[] state = Files.readAllBytes(dir.resolve("state-7.bin"));
        StreamCheckpoint.State decoded = StreamCheckpoint.decode(state);
        assertEquals(7, decoded.lastBatchId());
        assertEquals(List.of(sink.snapshot()), decoded.levels()); // two batches merged into level 1
        state[state.length / 2] ^= 0x10;
        assertThrows(IOException.class, () -> StreamCheckpoint.decode(state));
        assertThrows(IOException.class, () -> StreamCheckpoint.decode(new byte[] {1, 2, 3}));
    }
}
