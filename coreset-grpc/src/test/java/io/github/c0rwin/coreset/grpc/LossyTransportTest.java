package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.resilience.ChannelModel;
import io.github.c0rwin.coreset.grpc.resilience.Decoder;
import io.github.c0rwin.coreset.grpc.resilience.LossyTransport;
import io.github.c0rwin.coreset.grpc.resilience.ResilienceOptions;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot;
import io.github.c0rwin.coreset.node.MessageSpec;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** The lossy interceptor's drop statistics and its independence from the sender. */
class LossyTransportTest {

    private static final Duration WAIT = Duration.ofSeconds(30);
    private static final int DRAWS = 4000;

    private static long[] run(ChannelModel model, long seed, int gathers) throws Exception {
        LossyTransport lossy = new LossyTransport(model, seed);
        long[] received = new long[gathers];
        try (ResilienceCluster cluster = new ResilienceCluster(2, Map.of());
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(lossy), CoordinatorOptions.defaults())) {
            c.createSession("s", new SessionSpec("identity", Map.of(), 1L, 1, 16, TreeMode.SINGLE_WRITER,
                    MessageSpec.perDraw(DRAWS, 2, 1)));
            SplittableRandom rnd = new SplittableRandom(3);
            List<WeightedPoint> points = new ArrayList<>();
            for (int i = 0; i < 1000; i++) {
                points.add(new WeightedPoint(SparseVector.dense(new double[] {rnd.nextGaussian(), rnd.nextGaussian()}),
                        1.0));
            }
            try (IngestStream s = c.openIngest("s", "p")) {
                s.submit(points.subList(0, 500));
                s.submit(points.subList(500, 1000));
                s.flush(WAIT);
            }
            ResilienceOptions opts = ResilienceOptions.baseline(2).withDecoder(Decoder.REALIZED_COUNT);
            for (int g = 0; g < gathers; g++) {
                ResilientSnapshot snap = c.gatherResilient("s", opts.withStreamOffset(g * 2L));
                assertEquals(0, snap.lostGroups(), "headers are never dropped");
                assertEquals(2L * DRAWS, snap.packetsSent());
                received[g] = snap.packetsReceived();
            }
            long total = lossy.delivered() + lossy.dropped();
            assertEquals((long) gathers * 2 * DRAWS, total);
            double rho = model.lossRate();
            double observed = lossy.observedLossRate();
            if (model instanceof ChannelModel.Bernoulli) {
                assertEquals(rho, observed, 4 * Math.sqrt(rho * (1 - rho) / total), "binomial band");
            } else {
                assertEquals(rho, observed, 0.05);
            }
        }
        return received;
    }

    @Test
    void bernoulliDropsItsRateOfPacketsAndNoHeaders() throws Exception {
        run(new ChannelModel.Bernoulli(0.3), 5, 5);
    }

    @Test
    void gilbertDropsItsStationaryRate() throws Exception {
        run(ChannelModel.GilbertElliott.gilbert(0.3, 6), 5, 5);
    }

    /** Losses depend on the transport seed and the stream key only, not on what the sender sampled. */
    @Test
    void lossesAreReproducibleFromTheTransportSeed() throws Exception {
        ChannelModel m = new ChannelModel.Bernoulli(0.3);
        long[] a = run(m, 77, 3);
        long[] b = run(m, 77, 3);
        long[] c = run(m, 78, 3);
        assertEquals(java.util.Arrays.toString(a), java.util.Arrays.toString(b));
        assertNotEquals(java.util.Arrays.toString(a), java.util.Arrays.toString(c));
    }

    @Test
    void otherMethodsPassUntouched() throws Exception {
        LossyTransport lossy = new LossyTransport(new ChannelModel.Bernoulli(0.9), 1);
        try (ResilienceCluster cluster = new ResilienceCluster(1, Map.of());
             CoresetCoordinator c = new CoresetCoordinator(cluster.channels(lossy), CoordinatorOptions.defaults())) {
            c.createSession("s", new SessionSpec("identity", Map.of(), 1L, 1, 16));
            try (IngestStream s = c.openIngest("s", "p")) {
                s.submit(List.of(new WeightedPoint(SparseVector.dense(new double[] {1, 2}), 1.0)));
                s.flush(WAIT);
            }
            assertEquals(1, c.gather("s", WAIT).coreset().size());
            assertEquals(0, lossy.dropped() + lossy.delivered());
        }
    }
}
