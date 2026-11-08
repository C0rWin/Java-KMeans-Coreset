package io.github.c0rwin.coreset.grpc.resilience;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.node.MessageSpec;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * thm:erasure-invariance and its foils: per-draw messages decoded by realized count stay unbiased under any
 * content-oblivious channel, as-sent weights lose the erased mass, and merged records leave an m-independent floor.
 */
class ErasureInvarianceTest {

    private static final int RUNS = 600;
    private static final List<SparseVector> Q = List.of(SparseVector.dense(new double[] {1, 1}),
            SparseVector.dense(new double[] {14, 2}), SparseVector.dense(new double[] {-5, 13}));

    /** Mean and standard error of the decoded cost of Q over {@link #RUNS} streams. */
    private static double[] decodedCost(TestMessages messages, ChannelModel channel, Decoder decoder, long seed) {
        double sum = 0.0;
        double sumSq = 0.0;
        for (int run = 0; run < RUNS; run++) {
            ReceivedMessage r = messages.received(run, channel.open(seed + run));
            Coreset c = MessageDecoder.decode(List.of(r), decoder, channel.lossRate(), 1.0, TestMessages.MAX_DIM)
                    .coreset();
            double cost = Cost.cost(c, Q);
            sum += cost;
            sumSq += cost * cost;
        }
        double mean = sum / RUNS;
        double var = Math.max(0.0, sumSq / RUNS - mean * mean);
        return new double[] {mean, Math.sqrt(var / RUNS)};
    }

    @Test
    void realizedCountDecoderIsUnbiasedUnderMemorylessLoss() throws Exception {
        List<WeightedPoint> data = TestMessages.mixture(2000, 1);
        double truth = Cost.cost(data, Q);
        try (TestMessages m = new TestMessages(data, MessageSpec.perDraw(200, 3, 1))) {
            double[] realized = decodedCost(m, new ChannelModel.Bernoulli(0.3), Decoder.REALIZED_COUNT, 11);
            assertEquals(truth, realized[0], 4 * realized[1], "realized count must be unbiased");
            double[] asSent = decodedCost(m, new ChannelModel.Bernoulli(0.3), Decoder.AS_SENT, 11);
            // as-sent keeps only the delivered mass: biased low by the loss rate
            assertEquals(0.7, asSent[0] / truth, 0.03);
        }
    }

    @Test
    void realizedCountDecoderIsUnbiasedUnderBurstsAndMultiRecordPackets() throws Exception {
        List<WeightedPoint> data = TestMessages.mixture(2000, 2);
        double truth = Cost.cost(data, Q);
        try (TestMessages m = new TestMessages(data, MessageSpec.perDraw(200, 3, 4))) {
            ChannelModel bursty = ChannelModel.GilbertElliott.gilbert(0.4, 5.0);
            double[] realized = decodedCost(m, bursty, Decoder.REALIZED_COUNT, 23);
            assertEquals(truth, realized[0], 4 * realized[1], "bursts change only the law of M'");
        }
    }

    /** prop:merged-floor-lb on P = {0, 1}: a merged record of 1 is erased with probability rho, for every m. */
    @Test
    void mergedRecordsHaveAnErrorFloorThatPerDrawRecordsDoNot() throws Exception {
        List<WeightedPoint> two = List.of(new WeightedPoint(SparseVector.dense(new double[] {0}), 1.0),
                new WeightedPoint(SparseVector.dense(new double[] {1}), 1.0));
        List<SparseVector> origin = List.of(SparseVector.dense(new double[] {0}));
        ChannelModel channel = new ChannelModel.Bernoulli(0.3);
        int mergedFails = 0;
        int perDrawFails = 0;
        try (TestMessages merged = new TestMessages(two, MessageSpec.merged(50, 1, 1));
             TestMessages perDraw = new TestMessages(two, MessageSpec.perDraw(50, 1, 1))) {
            for (int run = 0; run < RUNS; run++) {
                Coreset a = MessageDecoder.decode(List.of(merged.received(run, channel.open(run))),
                        Decoder.HORVITZ_THOMPSON, 0.3, 1.0, TestMessages.MAX_DIM).coreset();
                Coreset b = MessageDecoder.decode(List.of(perDraw.received(run, channel.open(run))),
                        Decoder.REALIZED_COUNT, 0.3, 1.0, TestMessages.MAX_DIM).coreset();
                mergedFails += Cost.cost(a, origin) == 0.0 ? 1 : 0; // the point 1 was lost: cost(P, {0}) = 1
                perDrawFails += Cost.cost(b, origin) == 0.0 ? 1 : 0;
            }
        }
        double rate = (double) mergedFails / RUNS;
        assertEquals(0.3, rate, 4 * Math.sqrt(0.3 * 0.7 / RUNS), "floor rho (1 - 2^-m)");
        assertEquals(0, perDrawFails, "((1 + rho) / 2)^m = 0.65^50 ~ 4e-10");
    }

    @Test
    void perDrawRecordsOfOneDrawAreMergedBackByTag() throws Exception {
        try (TestMessages m = new TestMessages(TestMessages.mixture(300, 3), MessageSpec.perDraw(400, 3, 8))) {
            ReceivedMessage all = m.received(0, () -> true);
            MessageDecoder.Decoded d = MessageDecoder.decode(List.of(all), Decoder.REALIZED_COUNT, 0, 1.0,
                    TestMessages.MAX_DIM);
            assertEquals(400, d.recordsReceived());
            assertEquals(1.0, d.decoderScale());
            assertTrue(d.coreset().size() < 400, "duplicate draws collapse to one point each");
            assertEquals(m.message(0).recordWeight(), d.coreset().totalWeight(), 1e-9 * d.coreset().totalWeight());
        }
    }
}
