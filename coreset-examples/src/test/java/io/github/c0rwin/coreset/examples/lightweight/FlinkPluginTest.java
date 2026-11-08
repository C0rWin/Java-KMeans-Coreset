package io.github.c0rwin.coreset.examples.lightweight;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.flink.CoresetConfig;
import io.github.c0rwin.coreset.flink.CoresetSnapshot;
import io.github.c0rwin.coreset.flink.CoresetTypes;
import io.github.c0rwin.coreset.flink.FlinkCoresets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.apache.flink.api.common.functions.Partitioner;
import org.apache.flink.api.java.functions.KeySelector;
import org.apache.flink.configuration.CheckpointingOptions;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.PipelineOptions;
import org.apache.flink.runtime.testutils.MiniClusterResourceConfiguration;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.test.junit5.MiniClusterExtension;
import org.apache.flink.util.CloseableIterator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.RegisterExtension;

/** Flink MiniCluster: task managers resolve the registered id themselves, or run the code-supplied factory. */
class FlinkPluginTest {

    private static final int PARALLELISM = 4;
    private static final List<WeightedPoint> POINTS = Blobs.points(32_000, 10);

    @RegisterExtension
    static final MiniClusterExtension CLUSTER = new MiniClusterExtension(
            new MiniClusterResourceConfiguration.Builder()
                    .setNumberTaskManagers(1)
                    .setNumberSlotsPerTaskManager(PARALLELISM)
                    .build());

    private static final class HashKey implements KeySelector<WeightedPoint, Integer> {
        @Override
        public Integer getKey(WeightedPoint p) {
            return p.vector().hashCode();
        }
    }

    private static final class ByHash implements Partitioner<Integer> {
        @Override
        public int partition(Integer key, int partitions) {
            return Math.floorMod(key, partitions);
        }
    }

    private static CoresetSnapshot last(CoresetConfig config) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        Configuration conf = new Configuration();
        conf.set(PipelineOptions.GENERIC_TYPES, false); // every type has a serializer of its own
        conf.set(CheckpointingOptions.CHECKPOINTING_INTERVAL, Duration.ofMillis(50));
        env.configure(conf);
        // One source subtask and a content-based partitioner (rebalance starts at a random channel): every local
        // subtask sees the same points in the same order on every run.
        DataStream<WeightedPoint> points = env.fromData(POINTS, CoresetTypes.POINT)
                .partitionCustom(new ByHash(), new HashKey())
                .map(p -> p).returns(CoresetTypes.POINT).setParallelism(PARALLELISM);
        CoresetSnapshot last = null;
        try (CloseableIterator<CoresetSnapshot> out = FlinkCoresets.stream(points, config).executeAndCollect()) {
            while (out.hasNext()) {
                CoresetSnapshot s = out.next();
                if (s.last()) {
                    last = s;
                }
            }
        }
        assertNotNull(last, "no final snapshot");
        return last;
    }

    @Test
    void registeredIdAndFactoryStreamTheSameValidCoreset() throws Exception {
        CoresetSnapshot byId = last(CoresetConfig.builder("lightweight", Map.of("m", "500"), 29L)
                .compressSnapshots(true).build());
        CoresetSnapshot byFactory = last(CoresetConfig.builder(new LightweightCoresetFactory(500), 29L)
                .chunkSize(1_000).compressSnapshots(true).build());
        assertEquals(byId.coreset(), byFactory.coreset());
        assertEquals(byId.compressed(), byFactory.compressed());

        Coreset full = Coreset.of(POINTS);
        Coreset c = byId.compressed();
        double distortion = Blobs.distortion(full, c);
        double weightErr = Math.abs(c.totalWeight() - full.totalWeight()) / full.totalWeight();
        System.out.printf("lightweight on flink: %d records, union %d points, compressed %d points, distortion %.4f,"
                + " weight rel err %.2e%n", byId.records(), byId.coreset().size(), c.size(), distortion, weightErr);
        assertEquals(POINTS.size(), byId.records());
        assertTrue(c.size() <= 500);
        assertTrue(distortion < LightweightCoresetTest.MAX_STREAM_DISTORTION, "distortion " + distortion);
        assertTrue(weightErr < LightweightCoresetTest.MAX_WEIGHT_REL_ERR, "weight rel err " + weightErr);
    }
}
