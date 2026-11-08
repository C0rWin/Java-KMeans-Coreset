package io.github.c0rwin.coreset.node;

import static io.github.c0rwin.coreset.node.IngestResult.Status.ACCEPTED;
import static io.github.c0rwin.coreset.node.IngestResult.Status.BUSY;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.node.testing.TestAlgorithms;
import io.github.c0rwin.coreset.node.testing.TestData;
import io.github.c0rwin.coreset.node.testing.TestResolver;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** Generators are a function of (seed, partition, producer, seq): independent across partitions, stable on retry. */
class PartitionSeedsTest {

    private static final Duration WAIT = Duration.ofSeconds(30);
    private static final int DIM = 8;
    private static final CoresetAlgorithm SAMPLE = TestAlgorithms.sample(8);

    private CoresetNode node;

    @AfterEach
    void close() {
        if (node != null) {
            node.close();
        }
    }

    private Coreset ingestAndSnapshot(String session, List<List<WeightedPoint>> batches) throws Exception {
        for (int seq = 0; seq < batches.size(); seq++) {
            assertEquals(ACCEPTED, TestData.offer(node, session, "p", seq, batches.get(seq), WAIT).status());
        }
        assertTrue(node.awaitApplied(session, WAIT));
        return node.snapshot(session).coreset();
    }

    private static List<List<WeightedPoint>> batches(int count) {
        SplittableRandom rng = new SplittableRandom(31);
        List<List<WeightedPoint>> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(TestData.batch(rng, 64, DIM));
        }
        return out;
    }

    @Test
    void equalSeqsInDistinctPartitionsUseIndependentStreams() throws Exception {
        node = new CoresetNode(new TestResolver(Map.of("sample8", SAMPLE)), MetricsSink.NOOP, Duration.ofMillis(200));
        SessionSpec spec = new SessionSpec("sample8", Map.of(), 5L, 1, 4);
        List<List<WeightedPoint>> input = batches(1);
        node.createSession("a", spec, 1);
        node.createSession("b", spec, 2);
        node.createSession("c", spec, 1);
        node.createSession("d", spec);
        Coreset a = ingestAndSnapshot("a", input);
        assertNotEquals(a, ingestAndSnapshot("b", input));
        assertEquals(a, ingestAndSnapshot("c", input)); // same (seed, partition, producer, seq): same sample
        assertNotEquals(a, ingestAndSnapshot("d", input));
    }

    @Test
    void aBusyRetryReproducesTheSameSnapshotWithinAPartition() throws Exception {
        TestAlgorithms.Gated gated = new TestAlgorithms.Gated(SAMPLE);
        node = new CoresetNode(new TestResolver(Map.of("sample8", SAMPLE, "gated8", gated)), MetricsSink.NOOP,
                Duration.ofMillis(200));
        List<List<WeightedPoint>> input = batches(4);
        node.createSession("reference", new SessionSpec("sample8", Map.of(), 5L, 1, 2), 3);
        Coreset reference = ingestAndSnapshot("reference", input);

        node.createSession("retried", new SessionSpec("gated8", Map.of(), 5L, 1, 2), 3);
        assertEquals(ACCEPTED, node.ingest("retried", "p", 0, input.get(0)).status());
        gated.awaitEntered(); // the worker holds seq 0; seqs 1 and 2 fill the ring, seq 3 bounces
        assertEquals(ACCEPTED, node.ingest("retried", "p", 1, input.get(1)).status());
        assertEquals(ACCEPTED, node.ingest("retried", "p", 2, input.get(2)).status());
        assertEquals(BUSY, node.ingest("retried", "p", 3, input.get(3)).status());
        gated.release();
        assertTimeoutPreemptively(WAIT, () -> {
            while (node.ingest("retried", "p", 3, input.get(3)).status() == BUSY) {
                Thread.onSpinWait();
            }
        });
        assertTrue(node.awaitApplied("retried", WAIT));
        assertEquals(reference, node.snapshot("retried").coreset());
    }
}
