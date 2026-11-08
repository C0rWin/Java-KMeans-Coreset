package io.github.c0rwin.coreset.grpc.resilience;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.node.MessageSpec;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/**
 * prop:cf-triples on four far-apart blobs that the cell partition recovers exactly: whatever cells are lost, per-cell
 * triples decode to the exact mean coreset of the received blobs (cost of every single center equal to that of the
 * received points), while a global offset keeps the offsets of the lost cells.
 */
class CfTriplesTest {

    private static final double[][] BLOBS = {{0, 0}, {100, 0}, {0, 100}, {100, 100}};
    private static final List<SparseVector> CENTERS = List.of(SparseVector.dense(new double[] {3, -7}),
            SparseVector.dense(new double[] {50, 50}), SparseVector.dense(new double[] {120, 90}));

    private static List<WeightedPoint> blobs(SplittableRandom rnd) {
        List<WeightedPoint> out = new ArrayList<>();
        for (double[] b : BLOBS) {
            for (int i = 0; i < 150; i++) {
                out.add(new WeightedPoint(SparseVector.dense(new double[] {b[0] + rnd.nextGaussian() * 2,
                        b[1] + rnd.nextGaussian() * 3}), 0.5 + rnd.nextDouble()));
            }
        }
        return out;
    }

    private static int blobOf(SparseVector v) {
        int best = 0;
        for (int i = 1; i < BLOBS.length; i++) {
            if (v.squaredDistance(SparseVector.dense(BLOBS[i])) < v.squaredDistance(SparseVector.dense(BLOBS[best]))) {
                best = i;
            }
        }
        return best;
    }

    @Test
    void triplesDecodeToTheExactMeanCoresetOfTheReceivedCells() throws Exception {
        SplittableRandom rnd = new SplittableRandom(5);
        List<WeightedPoint> data = blobs(rnd);
        try (TestMessages cf = new TestMessages(data, MessageSpec.cfTriples(4, 1, 20));
             TestMessages global = new TestMessages(data, MessageSpec.globalOffset(4, 1, 20))) {
            assertEquals(4, cf.message(0).recordCount());
            for (int trial = 0; trial < 20; trial++) {
                ChannelModel channel = new ChannelModel.Bernoulli(0.5);
                long seed = 100 + trial;
                ReceivedMessage a = cf.received(0, channel.open(seed));
                ReceivedMessage b = global.received(0, channel.open(seed)); // same cells lost in both
                boolean[] received = new boolean[BLOBS.length];
                for (Packet p : a.packets()) {
                    PointBlock block = io.github.c0rwin.coreset.grpc.ProtoCodec.blockFromProto(p.getRecordsList(),
                            TestMessages.MAX_DIM);
                    received[blobOf(block.vector(0))] = true;
                }
                List<WeightedPoint> receivedPoints = new ArrayList<>();
                double lostOffsets = 0.0;
                for (WeightedPoint p : data) {
                    int blob = blobOf(p.vector());
                    if (received[blob]) {
                        receivedPoints.add(p);
                    }
                }
                Coreset decodedCf = MessageDecoder.decode(List.of(a), Decoder.AS_SENT, 0, 1.0, TestMessages.MAX_DIM)
                        .coreset();
                Coreset decodedGlobal = MessageDecoder.decode(List.of(b), Decoder.AS_SENT, 0, 1.0,
                        TestMessages.MAX_DIM).coreset();
                for (SparseVector x : CENTERS) {
                    List<SparseVector> q = List.of(x);
                    double exact = receivedPoints.isEmpty() ? 0.0 : Cost.cost(receivedPoints, q);
                    assertEquals(exact, Cost.cost(decodedCf, q), 1e-9 * Math.max(1.0, exact),
                            "CF triples: cost of the received data, trial " + trial);
                }
                // the global offset still counts every lost cell's spread
                double cfOffset = decodedCf.offset();
                lostOffsets = decodedGlobal.offset() - cfOffset;
                int lost = 0;
                for (boolean r : received) {
                    lost += r ? 0 : 1;
                }
                if (lost > 0) {
                    assertTrue(lostOffsets > 0.0, "trial " + trial);
                } else {
                    assertEquals(0.0, lostOffsets, 1e-9 * decodedGlobal.offset());
                }
            }
        }
    }

    /** With the realized-mass option the received cells stand for the whole message (an HT-style estimate). */
    @Test
    void realizedWeightRescalesCellsAndTheirOffsetsTogether() throws Exception {
        List<WeightedPoint> data = blobs(new SplittableRandom(6));
        try (TestMessages cf = new TestMessages(data, MessageSpec.cfTriples(4, 1, 20))) {
            ReceivedMessage all = cf.received(0, () -> true);
            ReceivedMessage three = new ReceivedMessage(0, all.header(), all.packets().subList(0, 3));
            Coreset asSent = MessageDecoder.decode(List.of(three), Decoder.AS_SENT, 0, 1.0, TestMessages.MAX_DIM)
                    .coreset();
            MessageDecoder.Decoded d = MessageDecoder.decode(List.of(three), Decoder.REALIZED_WEIGHT, 0, 1.0,
                    TestMessages.MAX_DIM);
            double scale = all.header().getRecordWeight() / three.weightReceived();
            assertEquals(scale, d.decoderScale(), 1e-12);
            assertEquals(asSent.offset() * scale, d.coreset().offset(), 1e-9 * d.coreset().offset());
            assertEquals(all.header().getRecordWeight(), d.coreset().totalWeight(), 1e-9 * d.coreset().totalWeight());
        }
    }
}
