package io.github.c0rwin.coreset.grpc.resilience;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ChannelModelTest {

    private static final int PACKETS = 200_000;

    @Test
    void bernoulliLosesItsRate() {
        ErasureChannel c = new ChannelModel.Bernoulli(0.3).open(1);
        int lost = 0;
        for (int i = 0; i < PACKETS; i++) {
            lost += c.deliver() ? 0 : 1;
        }
        assertEquals(0.3, (double) lost / PACKETS, 4 * Math.sqrt(0.21 / PACKETS));
    }

    @Test
    void gilbertHasItsStationaryRateAndMeanBurstLength() {
        ChannelModel.GilbertElliott g = ChannelModel.GilbertElliott.gilbert(0.3, 8.0);
        assertEquals(0.3, g.lossRate(), 1e-12);
        assertEquals(1.0 - g.goodToBad() - g.badToGood(), g.lambda(), 1e-12);
        ErasureChannel c = g.open(2);
        int lost = 0;
        int bursts = 0;
        boolean prevLost = false;
        for (int i = 0; i < PACKETS; i++) {
            boolean l = !c.deliver();
            lost += l ? 1 : 0;
            bursts += l && !prevLost ? 1 : 0;
            prevLost = l;
        }
        assertEquals(0.3, (double) lost / PACKETS, 0.02); // correlated: a looser band than Bernoulli
        assertEquals(8.0, (double) lost / bursts, 0.5);
    }

    @Test
    void equalSeedsGiveEqualIndicatorsAndDistinctSeedsDiffer() {
        ChannelModel m = new ChannelModel.Bernoulli(0.5);
        ErasureChannel a = m.open(9);
        ErasureChannel b = m.open(9);
        ErasureChannel c = m.open(10);
        StringBuilder sa = new StringBuilder();
        StringBuilder sb = new StringBuilder();
        StringBuilder sc = new StringBuilder();
        for (int i = 0; i < 64; i++) {
            sa.append(a.deliver() ? '1' : '0');
            sb.append(b.deliver() ? '1' : '0');
            sc.append(c.deliver() ? '1' : '0');
        }
        assertEquals(sa.toString(), sb.toString());
        assertNotEquals(sa.toString(), sc.toString());
    }

    @Test
    void parsesTheCommandLineForms() {
        assertInstanceOf(ChannelModel.Perfect.class, ChannelModel.parse("none"));
        assertEquals(new ChannelModel.Bernoulli(0.3), ChannelModel.parse("bernoulli:0.3"));
        assertEquals(ChannelModel.GilbertElliott.gilbert(0.2, 4), ChannelModel.parse("gilbert:0.2,4"));
        assertEquals(new ChannelModel.GilbertElliott(0.1, 0.5, 0.0, 0.8), ChannelModel.parse("ge:0.1,0.5,0,0.8"));
        assertThrows(IllegalArgumentException.class, () -> ChannelModel.parse("bernoulli:1.0"));
        assertThrows(IllegalArgumentException.class, () -> ChannelModel.parse("ge:0.1"));
        assertTrue(ChannelModel.parse("bernoulli:0.3").toString().startsWith("bernoulli"));
    }

    @Test
    void lossBoundsFollowCorLossEps() {
        // (b): mu = 700, sqrt(2 * 700 * ln 20) = 64.7 -> m_eta = ceil(635.3) = 636
        assertEquals(636, LossBounds.memorylessDraws(1000, 0.3, 0.05, 1));
        // (c): B = 4, mu_B = 175, deviation sqrt(2 * 175 * ln 20) = 32.4 -> 4 * ceil(142.6) = 572
        assertEquals(572, LossBounds.memorylessDraws(1000, 0.3, 0.05, 4));
        // (d) with lambda = 0 is Chebyshev-Cantelli on Bernoulli: V = 210, sqrt(210 * 19) = 63.2 -> 637
        assertEquals(637, LossBounds.gilbertDraws(1000, 0.3, 0.0, 0.05));
        assertTrue(LossBounds.gilbertDraws(1000, 0.3, 0.8, 0.05) < 637);
        assertEquals(Math.sqrt(1000.0 / 636), LossBounds.inflation(1000, 636), 1e-12);
        assertEquals(0, LossBounds.memorylessDraws(4, 0.9, 0.01, 1));
    }
}
