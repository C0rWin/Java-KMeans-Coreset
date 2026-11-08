package io.github.c0rwin.coreset.node;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.node.testing.TestProvider;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** Message shapes built from a session whose summary is its raw input (identity algorithm). */
class NodeMessageTest {

    private static final int N = 400;
    private CoresetNode node;

    @AfterEach
    void close() {
        if (node != null) {
            node.close();
        }
    }

    private CoresetNode nodeWith(MessageSpec message, long partition) throws InterruptedException {
        node = new CoresetNode(TestProvider.registry(), MetricsSink.NOOP, Duration.ofMillis(200));
        node.createSession("s", new SessionSpec("identity", Map.of(), 11L, 1, 16, TreeMode.SINGLE_WRITER, message),
                partition);
        SplittableRandom rnd = new SplittableRandom(3);
        for (int b = 0; b < 4; b++) {
            List<WeightedPoint> batch = new ArrayList<>();
            for (int i = 0; i < N / 4; i++) {
                int cluster = rnd.nextInt(4);
                batch.add(new WeightedPoint(SparseVector.dense(new double[] {cluster * 20 + rnd.nextGaussian(),
                        (cluster % 2) * 20 + rnd.nextGaussian()}), 1.0 + rnd.nextInt(3)));
            }
            assertEquals(IngestResult.Status.ACCEPTED, node.ingest("s", b, batch).status());
        }
        assertTrue(node.awaitApplied("s", Duration.ofSeconds(30)));
        return node;
    }

    @Test
    void perDrawSendsOneRecordPerDrawWithTagsOfDistinctDraws() throws Exception {
        NodeMessage m = nodeWith(MessageSpec.perDraw(256, 4, 8), 0).message("s", 0);
        double total = node.snapshot("s").coreset().totalWeight();
        assertEquals(Packetization.PER_DRAW, m.packetization());
        assertEquals(256, m.draws());
        assertEquals(256, m.recordCount());
        assertEquals(32, m.packets());
        assertEquals(total, m.nodeWeight(), 1e-9);
        assertEquals(N, m.nodePoints());
        assertTrue(m.complete());
        // equal tags are equal points with equal per-draw weights w / (m q)
        PointBlock r = m.records();
        int[] tags = m.tags();
        for (int i = 0; i < r.size(); i++) {
            for (int j = i + 1; j < r.size(); j++) {
                if (tags[i] == tags[j]) {
                    assertEquals(r.vector(i), r.vector(j));
                    assertEquals(r.weight(i), r.weight(j), 0.0);
                }
            }
        }
        // unbiased total weight: E sum_draws w / (m q) = W; one realization lies close for this tame input
        assertEquals(total, m.recordWeight(), 0.25 * total);
    }

    @Test
    void mergedCarriesDrawCountsOfTheSameSample() throws Exception {
        NodeMessage merged = nodeWith(MessageSpec.merged(256, 4, 8), 0).message("s", 0);
        assertEquals(256, Arrays.stream(merged.tags()).sum());
        assertTrue(merged.recordCount() < 256);
    }

    @Test
    void streamsShareTheDistributionButDrawIndependently() throws Exception {
        nodeWith(MessageSpec.perDraw(128, 4, 1), 0);
        NodeMessage a = node.message("s", 0);
        NodeMessage again = node.message("s", 0);
        NodeMessage b = node.message("s", 1);
        assertEquals(a.records(), again.records());
        assertArrayEquals(a.tags(), again.tags());
        assertNotEquals(a.records(), b.records());
        // the weight w / (m q) of a point is a function of q only, so it agrees across streams
        for (int i = 0; i < a.recordCount(); i++) {
            for (int j = 0; j < b.recordCount(); j++) {
                if (a.records().vector(i).equals(b.records().vector(j))) {
                    assertEquals(a.records().weight(i), b.records().weight(j), 1e-9);
                }
            }
        }
    }

    @Test
    void cfTriplesCarryPerCellOffsetsThatSumToTheGlobalOffset() throws Exception {
        NodeMessage cf = nodeWith(MessageSpec.cfTriples(4, 1, 10), 0).message("s", 0);
        node.close();
        NodeMessage global = nodeWith(MessageSpec.globalOffset(4, 1, 10), 0).message("s", 0);
        assertEquals(4, cf.recordCount());
        assertEquals(0.0, cf.offset());
        assertEquals(Arrays.stream(cf.cellOffsets()).sum(), global.offset(), 1e-9 * global.offset());
        assertEquals(0, global.cellOffsets().length);
        assertEquals(cf.recordWeight(), node.snapshot("s").coreset().totalWeight(), 1e-9);
    }

    @Test
    void summaryIsTheSnapshotAsIs() throws Exception {
        NodeMessage m = nodeWith(MessageSpec.SUMMARY, 0).message("s", 0);
        assertEquals(node.snapshot("s").coreset().pointSet().compact(), m.records());
        assertEquals(N, m.draws());
    }

    @Test
    void frozenMessageIsServedForItsStreamAndReplicasAreKeptPerOwner() throws Exception {
        nodeWith(MessageSpec.perDraw(64, 4, 4), 0);
        NodeMessage frozen = node.freeze("s", 0);
        assertSame(frozen, node.message("s", 0));
        assertNotEquals(frozen.records(), node.message("s", 5).records());
        assertNull(node.replica("s", 3));
        node.storeReplica("s", 3, frozen);
        assertSame(frozen, node.replica("s", 3));
    }

    @Test
    void replicasAreBounded() throws Exception {
        nodeWith(MessageSpec.perDraw(64, 4, 4), 0);
        NodeMessage m = node.message("s", 0);
        for (int owner = 0; owner < Session.MAX_REPLICAS; owner++) {
            node.storeReplica("s", owner, m);
        }
        node.storeReplica("s", 0, m); // replacing an owner's copy is fine
        assertThrows(NodeLimitException.class, () -> node.storeReplica("s", Session.MAX_REPLICAS, m));
    }

    @Test
    void specsValidateTheirShape() {
        assertThrows(IllegalArgumentException.class, () -> MessageSpec.perDraw(0, 4, 1));
        assertThrows(IllegalArgumentException.class, () -> MessageSpec.perDraw(8, 0, 1));
        assertThrows(IllegalArgumentException.class, () -> MessageSpec.cfTriples(4, 0, 1));
        assertEquals(Packetization.MEAN_CF_TRIPLES, Packetization.fromId("mean-cf"));
        assertThrows(IllegalArgumentException.class, () -> Packetization.fromId("nope"));
    }
}
