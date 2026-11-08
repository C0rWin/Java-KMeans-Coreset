package io.github.c0rwin.coreset.experiments.lossy;

import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.LossyConfig;
import java.util.SplittableRandom;

/** Decides per message whether the channel loses it. Implementations are seeded and thread-safe. */
public interface ChannelModel {

    boolean drop();

    /** Builds the configured model; {@code none} never drops. */
    static ChannelModel of(LossyConfig c) {
        return switch (c.model()) {
            case "none" -> () -> false;
            case "bernoulli" -> new Bernoulli(require(c.p(), "p"), c.seed());
            case "gilbert-elliott" -> new GilbertElliott(require(c.pGoodToBad(), "pGoodToBad"),
                    require(c.pBadToGood(), "pBadToGood"), c.lossGood() != null ? c.lossGood() : 0.0,
                    c.lossBad() != null ? c.lossBad() : 1.0, c.seed());
            default -> throw new IllegalArgumentException("unknown channel model " + c.model()
                    + ", expected none, bernoulli or gilbert-elliott");
        };
    }

    private static double require(Double v, String name) {
        if (v == null || !(v >= 0.0 && v <= 1.0)) {
            throw new IllegalArgumentException("channel model needs " + name + " in [0, 1], got " + v);
        }
        return v;
    }

    /** Independent losses with probability {@code p}. */
    final class Bernoulli implements ChannelModel {
        private final double p;
        private final SplittableRandom rnd;

        public Bernoulli(double p, long seed) {
            this.p = p;
            this.rnd = new SplittableRandom(seed);
        }

        @Override
        public synchronized boolean drop() {
            return rnd.nextDouble() < p;
        }
    }

    /**
     * Two-state Markov channel (Gilbert-Elliott): bursts of loss while in the bad state. Per message the state moves
     * good to bad with {@code pGoodToBad} and back with {@code pBadToGood}, then the message is lost with the state's
     * loss rate. Mean burst length is {@code 1 / pBadToGood}.
     */
    final class GilbertElliott implements ChannelModel {
        private final double pGoodToBad;
        private final double pBadToGood;
        private final double lossGood;
        private final double lossBad;
        private final SplittableRandom rnd;
        private boolean bad;

        public GilbertElliott(double pGoodToBad, double pBadToGood, double lossGood, double lossBad, long seed) {
            this.pGoodToBad = pGoodToBad;
            this.pBadToGood = pBadToGood;
            this.lossGood = lossGood;
            this.lossBad = lossBad;
            this.rnd = new SplittableRandom(seed);
        }

        @Override
        public synchronized boolean drop() {
            bad = bad ? rnd.nextDouble() >= pBadToGood : rnd.nextDouble() < pGoodToBad;
            return rnd.nextDouble() < (bad ? lossBad : lossGood);
        }
    }
}
