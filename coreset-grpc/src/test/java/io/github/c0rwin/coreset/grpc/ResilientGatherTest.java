package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.grpc.proto.ReplicateResponse;
import io.github.c0rwin.coreset.grpc.resilience.Census;
import io.github.c0rwin.coreset.grpc.resilience.ChannelModel;
import io.github.c0rwin.coreset.grpc.resilience.Decoder;
import io.github.c0rwin.coreset.grpc.resilience.GroupLayout;
import io.github.c0rwin.coreset.grpc.resilience.LossyTransport;
import io.github.c0rwin.coreset.grpc.resilience.ResilienceOptions;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot.GroupOutcome;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot.SourceOutcome;
import io.github.c0rwin.coreset.grpc.resilience.SurvivorReweighting;
import io.github.c0rwin.coreset.node.MessageSpec;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.grpc.StatusRuntimeException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class ResilientGatherTest {

    private static final Duration WAIT = Duration.ofSeconds(30);
    private static final Map<String, CoresetAlgorithm> SENS =
            Map.of("sens", new SensitivityCoreset(3, 60, Seeding.kMeansPlusPlus(0)));
    private static final List<SparseVector> Q = List.of(SparseVector.dense(new double[] {1, 1}),
            SparseVector.dense(new double[] {14, 2}), SparseVector.dense(new double[] {-5, 13}));

    private static CoordinatorOptions options(int inFlight) {
        return new CoordinatorOptions(inFlight, WAIT, Duration.ofMinutes(5), Duration.ofMillis(1),
                Duration.ofMillis(20), 5, CoordinatorOptions.DEFAULT_MAX_MESSAGE_BYTES, WAIT, WAIT, 1 << 24, 2);
    }

    private static SessionSpec spec(String algorithm, MessageSpec message) {
        return new SessionSpec(algorithm, Map.of(), 3L, 1, 64, TreeMode.SINGLE_WRITER, message);
    }

    private static List<WeightedPoint> mixture(int n, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        double[][] centers = {{0, 0}, {12, 3}, {-6, 15}, {25, -10}};
        List<WeightedPoint> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int c = rnd.nextInt(10) < 6 ? 0 : 1 + rnd.nextInt(3);
            out.add(new WeightedPoint(SparseVector.dense(new double[] {centers[c][0] + rnd.nextGaussian() * (1 + c),
                    centers[c][1] + rnd.nextGaussian()}), 1.0));
        }
        return out;
    }

    private static void ingest(CoresetCoordinator c, String session, GroupLayout layout, List<WeightedPoint> points,
                               int batch) throws Exception {
        try (IngestStream s = c.openIngest(session, "p", layout)) {
            for (int from = 0; from < points.size(); from += batch) {
                s.submit(points.subList(from, Math.min(points.size(), from + batch)));
            }
            s.flush(WAIT);
        }
    }

    private static double total(Census c) {
        double w = 0;
        for (double x : c.nodeWeight()) {
            w += x;
        }
        return w;
    }

    @Test
    void crashedNodeIsSurvivedAndReweightedByRealizedWeight() throws Exception {
        try (ResilienceCluster cluster = new ResilienceCluster(4, Map.of());
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(), options(8))) {
            GroupLayout layout = GroupLayout.independent(4);
            c.createSession("s", spec("identity", MessageSpec.SUMMARY));
            ingest(c, "s", layout, mixture(4000, 1), 100);
            Census census = c.census("s", layout, WAIT);
            double w = total(census);
            assertEquals(4000, w, 1e-9);
            cluster.kill(2);
            assertThrows(StatusRuntimeException.class, () -> c.gather("s", WAIT)); // the baseline fails outright
            ResilienceOptions base = ResilienceOptions.baseline(4).withCallDeadline(Duration.ofSeconds(5));

            ResilientSnapshot none = c.gatherResilient("s", base);
            assertEquals(1, none.lostGroups());
            assertEquals(1.0, none.survivorScale());
            assertEquals(w - census.nodeWeight()[2], none.coreset().totalWeight(), 1e-9);
            GroupOutcome lost = none.groups().get(2);
            assertFalse(lost.delivered());
            assertTrue(lost.sources().get(0).error().contains("UNAVAILABLE"), lost.sources().get(0).error());

            ResilientSnapshot byWeight = c.gatherResilient("s",
                    base.withReweighting(SurvivorReweighting.REALIZED_WEIGHT, census));
            assertEquals(w / (w - census.nodeWeight()[2]), byWeight.survivorScale(), 1e-12);
            assertEquals(w, byWeight.coreset().totalWeight(), 1e-9);

            ResilientSnapshot byNodes = c.gatherResilient("s",
                    base.withReweighting(SurvivorReweighting.NODE_FRACTION, null));
            assertEquals(4.0 / 3.0, byNodes.survivorScale(), 1e-12);
        }
    }

    /** prop:ht-unbiased (b): on equal-size random partitions, (N / M) times the survivors' cost is unbiased. */
    @Test
    void realizedWeightReweightingIsUnbiasedOverRandomPartitions() throws Exception {
        List<WeightedPoint> data = mixture(1200, 2);
        double truth = Cost.cost(data, Q);
        int runs = 120;
        double sum = 0.0;
        double sumSq = 0.0;
        double sumNone = 0.0;
        try (ResilienceCluster cluster = new ResilienceCluster(4, Map.of());
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(), options(16))) {
            GroupLayout layout = GroupLayout.independent(4);
            ResilienceOptions base = ResilienceOptions.baseline(4).withCallDeadline(Duration.ofSeconds(5));
            for (int run = 0; run < runs; run++) {
                String session = "r" + run;
                List<WeightedPoint> shuffled = new ArrayList<>(data);
                Collections.shuffle(shuffled, new Random(run));
                c.createSession(session, spec("identity", MessageSpec.SUMMARY));
                ingest(c, session, layout, shuffled, 10);
                Census census = c.census(session, layout, WAIT);
                cluster.nodes.get(1).closeSession(session); // node 1's share is lost after the census
                ResilientSnapshot s = c.gatherResilient(session,
                        base.withReweighting(SurvivorReweighting.REALIZED_WEIGHT, census));
                assertEquals(1, s.lostGroups());
                double cost = Cost.cost(s.coreset(), Q);
                sum += cost;
                sumSq += cost * cost;
                sumNone += cost / s.survivorScale();
                c.closeSession(session);
            }
        }
        double mean = sum / runs;
        double se = Math.sqrt(Math.max(0, sumSq / runs - mean * mean) / runs);
        assertEquals(truth, mean, 4 * se);
        assertEquals(0.75, sumNone / runs / truth, 0.02, "without reweighting only the survivors are summarized");
    }

    @Test
    void replicaHoldersServeTheIdenticalCoresetAfterTheOwnerCrashes() throws Exception {
        try (ResilienceCluster cluster = new ResilienceCluster(3, SENS);
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(), options(8))) {
            GroupLayout layout = GroupLayout.independent(3);
            c.createSession("s", spec("sens", MessageSpec.perDraw(128, 3, 8)));
            ingest(c, "s", layout, mixture(3000, 3), 250);
            List<ReplicateResponse> replies = c.replicate("s", 1, cluster.names, layout, WAIT);
            for (ReplicateResponse r : replies) {
                assertEquals(1, r.getStored(), r.toString());
                assertEquals(128, r.getRecords());
            }
            ResilienceOptions opts = ResilienceOptions.baseline(3).withDecoder(Decoder.REALIZED_COUNT)
                    .withReplication(1).withCallDeadline(Duration.ofSeconds(5));
            ResilientSnapshot before = c.gatherResilient("s", opts);
            cluster.kill(1);
            ResilientSnapshot after = c.gatherResilient("s", opts);
            assertEquals(0, after.lostGroups());
            assertEquals(before.coreset(), after.coreset());
            SourceOutcome failover = after.groups().get(1).sources().get(0);
            assertEquals(2, failover.servedBy());
            assertTrue(failover.viaReplica());
            assertTrue(failover.error().contains("UNAVAILABLE"), failover.error());
            // without replicas the same crash loses the node
            assertEquals(1, c.gatherResilient("s", opts.withReplication(0)).lostGroups());
        }
    }

    @Test
    void ratelessReplicasPoolTheirIndependentDraws() throws Exception {
        try (ResilienceCluster cluster = new ResilienceCluster(4, SENS);
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(), options(1))) {
            GroupLayout layout = GroupLayout.rateless(4, 2);
            c.createSession("s", spec("sens", MessageSpec.perDraw(64, 3, 1)), layout);
            ingest(c, "s", layout, mixture(2000, 4), 200);
            c.census("s", layout, WAIT); // waits until every accepted batch is applied
            // members of a group hold the same summary but sample it with their own stream
            assertEquals(cluster.nodes.get(0).snapshot("s").coreset(), cluster.nodes.get(1).snapshot("s").coreset());
            assertFalse(cluster.nodes.get(0).message("s", 0).records()
                    .equals(cluster.nodes.get(1).message("s", 1).records()));
            ResilienceOptions opts = ResilienceOptions.baseline(4).withLayout(layout).withDecoder(Decoder.REALIZED_COUNT)
                    .withCallDeadline(Duration.ofSeconds(5));
            ResilientSnapshot both = c.gatherResilient("s", opts);
            for (GroupOutcome g : both.groups()) {
                assertEquals(128, g.recordsReceived());
                assertEquals(0.5, g.decoderScale(), 0.0, "w / (M q) with M = 2 m1");
            }
            cluster.kill(1);
            ResilientSnapshot one = c.gatherResilient("s", opts);
            assertEquals(0, one.lostGroups());
            assertEquals(64, one.groups().get(0).recordsReceived());
            assertEquals(1.0, one.groups().get(0).decoderScale(), 0.0);
        }
    }

    @Test
    void fractionalRepetitionRecoversTheSameCoresetFromAnyMember() throws Exception {
        try (ResilienceCluster cluster = new ResilienceCluster(4, SENS);
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(), options(1))) {
            GroupLayout layout = GroupLayout.fractionalRepetition(4, 1);
            c.createSession("s", spec("sens", MessageSpec.perDraw(96, 3, 4)), layout);
            ingest(c, "s", layout, mixture(2000, 5), 200);
            c.census("s", layout, WAIT);
            ResilienceOptions opts = ResilienceOptions.baseline(4).withLayout(layout).withDecoder(Decoder.REALIZED_COUNT)
                    .withAwaitApplied(Duration.ZERO).withCallDeadline(Duration.ofSeconds(2));
            Coreset all = c.gatherResilient("s", opts).coreset();
            cluster.delay(0, 5_000); // node 0 straggles past the deadline: node 1 alone covers group 0
            ResilientSnapshot withoutZero = c.gatherResilient("s", opts);
            cluster.delay(0, 0);
            cluster.delay(1, 5_000);
            ResilientSnapshot withoutOne = c.gatherResilient("s", opts);
            cluster.delay(1, 0);
            assertEquals(all, withoutZero.coreset());
            assertEquals(all, withoutOne.coreset());
            assertEquals(0, withoutOne.lostGroups());
        }
    }

    @Test
    void fractionalRepetitionFillsLostPacketsFromTheOtherMember() throws Exception {
        LossyTransport lossy = new LossyTransport(new ChannelModel.Bernoulli(0.5), 31);
        try (ResilienceCluster cluster = new ResilienceCluster(4, SENS);
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(lossy), options(1))) {
            GroupLayout layout = GroupLayout.fractionalRepetition(4, 1);
            c.createSession("s", spec("sens", MessageSpec.perDraw(400, 3, 1)), layout);
            ingest(c, "s", layout, mixture(2000, 6), 200);
            ResilientSnapshot s = c.gatherResilient("s", ResilienceOptions.baseline(4).withLayout(layout)
                    .withDecoder(Decoder.REALIZED_COUNT));
            for (GroupOutcome g : s.groups()) {
                double share = (double) g.recordsReceived() / g.recordsSent();
                assertEquals(0.75, share, 0.07, "a packet is lost only if both copies are: 1 - 0.5^2");
            }
        }
    }

    @Test
    void theGatherProceedsOnceEveryGroupHasAQuorum() throws Exception {
        try (ResilienceCluster cluster = new ResilienceCluster(4, SENS);
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(), options(1))) {
            GroupLayout layout = GroupLayout.fractionalRepetition(4, 1);
            c.createSession("s", spec("sens", MessageSpec.perDraw(64, 3, 4)), layout);
            ingest(c, "s", layout, mixture(1000, 7), 100);
            cluster.delay(3, 4_000);
            long start = System.nanoTime();
            ResilientSnapshot s = c.gatherResilient("s", ResilienceOptions.baseline(4).withLayout(layout)
                    .withDecoder(Decoder.REALIZED_COUNT).withCallDeadline(Duration.ofSeconds(20)).withMinResponders(1));
            long ms = (System.nanoTime() - start) / 1_000_000;
            assertTrue(ms < 3_000, "waited " + ms + " ms for a straggler");
            assertEquals(0, s.lostGroups());
            SourceOutcome straggler = s.groups().get(1).sources().get(1);
            assertEquals(3, straggler.node());
            assertFalse(straggler.completed());
        }
    }

    /** End to end with 30% packet loss: realized-count decoding stays a small-distortion coreset of all the data. */
    @Test
    void perDrawMessagesUnderPacketLossStayWithinTheDistortionBound() throws Exception {
        List<WeightedPoint> data = mixture(8000, 8);
        Coreset full = Coreset.of(data);
        LossyTransport lossy = new LossyTransport(new ChannelModel.Bernoulli(0.3), 99);
        try (ResilienceCluster cluster = new ResilienceCluster(4, Map.of("sens",
                new SensitivityCoreset(4, 200, Seeding.kMeansPlusPlus(0))));
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(lossy), options(8))) {
            GroupLayout layout = GroupLayout.independent(4);
            c.createSession("s", spec("sens", MessageSpec.perDraw(400, 4, 4)));
            ingest(c, "s", layout, data, 500);
            ResilienceOptions opts = ResilienceOptions.baseline(4).withDecoder(Decoder.REALIZED_COUNT);
            ResilientSnapshot s = c.gatherResilient("s", opts);
            assertEquals(0.3, lossy.observedLossRate(), 0.05);
            double distortion = Distortion.evaluate(full, s.coreset(),
                    CandidateSolutions.kMeansPlusPlusOnFull(full, 4, 10, new SplittableRandom(1))).max() - 1.0;
            assertTrue(distortion < 0.25, "distortion " + distortion);
            assertEquals(1.0, s.coreset().totalWeight() / data.size(), 0.1);
            ResilientSnapshot asSent = c.gatherResilient("s", opts.withDecoder(Decoder.AS_SENT).withStreamOffset(4));
            assertEquals(0.7, asSent.coreset().totalWeight() / data.size(), 0.07, "as-sent loses the erased mass");
        }
    }

    @Test
    void theSummaryBaselineThroughStreamsMatchesTheClassicGather() throws Exception {
        try (ResilienceCluster cluster = new ResilienceCluster(3, SENS);
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(), options(8))) {
            c.createSession("s", spec("sens", MessageSpec.SUMMARY));
            ingest(c, "s", GroupLayout.independent(3), mixture(3000, 9), 300);
            Coreset classic = c.gather("s", WAIT).coreset();
            Coreset streamed = c.gatherResilient("s", ResilienceOptions.baseline(3)).coreset();
            assertEquals(classic.totalWeight(), streamed.totalWeight(), 1e-9 * classic.totalWeight());
            assertEquals(Cost.cost(classic, Q), Cost.cost(streamed, Q), 1e-9 * Cost.cost(classic, Q));
            assertEquals(classic.size(), streamed.size());
        }
    }
}
