package io.github.c0rwin.coreset.grpc.resilience;

import java.util.Locale;
import java.util.SplittableRandom;

/**
 * A content-oblivious packet erasure channel (def:oblivious): delivery indicators depend only on the channel's own
 * seed, never on payloads or on the sender's randomness.
 */
public sealed interface ChannelModel
        permits ChannelModel.Perfect, ChannelModel.Bernoulli, ChannelModel.GilbertElliott {

    /** Indicators of one stream; equal seeds give equal sequences. */
    ErasureChannel open(long seed);

    /** Stationary loss rate rho = 1 - E[delivered] / N. */
    double lossRate();

    /** Delivers everything. */
    record Perfect() implements ChannelModel {
        @Override
        public ErasureChannel open(long seed) {
            return () -> true;
        }

        @Override
        public double lossRate() {
            return 0.0;
        }

        @Override
        public String toString() {
            return "none";
        }
    }

    /** Memoryless channel: each packet is lost independently with probability {@code loss}. */
    record Bernoulli(double loss) implements ChannelModel {
        public Bernoulli {
            if (!(loss >= 0.0 && loss < 1.0)) {
                throw new IllegalArgumentException("loss must be in [0, 1): " + loss);
            }
        }

        @Override
        public ErasureChannel open(long seed) {
            SplittableRandom rng = new SplittableRandom(seed);
            return () -> rng.nextDouble() >= loss;
        }

        @Override
        public double lossRate() {
            return loss;
        }

        @Override
        public String toString() {
            return "bernoulli:" + loss;
        }
    }

    /**
     * Two-state Markov channel started in its stationary law: in state Good a packet is lost with {@code lossGood},
     * in Bad with {@code lossBad}; after each packet Good turns Bad with {@code goodToBad} and Bad turns Good with
     * {@code badToGood}. With {@code lossGood = 0, lossBad = 1} (Gilbert) the indicators are the chain itself, whose
     * second eigenvalue {@link #lambda()} enters cor:loss-eps (d).
     */
    record GilbertElliott(double goodToBad, double badToGood, double lossGood, double lossBad)
            implements ChannelModel {
        public GilbertElliott {
            if (!(goodToBad > 0.0 && goodToBad <= 1.0 && badToGood > 0.0 && badToGood <= 1.0)) {
                throw new IllegalArgumentException("transition probabilities must be in (0, 1]");
            }
            if (!(lossGood >= 0.0 && lossGood <= 1.0 && lossBad >= 0.0 && lossBad <= 1.0)) {
                throw new IllegalArgumentException("loss probabilities must be in [0, 1]");
            }
        }

        /** Gilbert channel with mean loss {@code rho} and mean burst length {@code burst} packets. */
        public static GilbertElliott gilbert(double rho, double burst) {
            if (!(rho > 0.0 && rho < 1.0) || !(burst >= 1.0)) {
                throw new IllegalArgumentException("need 0 < rho < 1 and burst >= 1");
            }
            double badToGood = 1.0 / burst;
            return new GilbertElliott(rho * badToGood / (1.0 - rho), badToGood, 0.0, 1.0);
        }

        /** Stationary probability of Bad. */
        public double badShare() {
            return goodToBad / (goodToBad + badToGood);
        }

        /** Second eigenvalue {@code 1 - goodToBad - badToGood} of the chain. */
        public double lambda() {
            return 1.0 - goodToBad - badToGood;
        }

        @Override
        public double lossRate() {
            double bad = badShare();
            return bad * lossBad + (1.0 - bad) * lossGood;
        }

        @Override
        public ErasureChannel open(long seed) {
            SplittableRandom rng = new SplittableRandom(seed);
            boolean[] bad = {rng.nextDouble() < badShare()};
            return () -> {
                boolean delivered = rng.nextDouble() >= (bad[0] ? lossBad : lossGood);
                bad[0] = bad[0] ? rng.nextDouble() >= badToGood : rng.nextDouble() < goodToBad;
                return delivered;
            };
        }

        @Override
        public String toString() {
            return "ge:" + goodToBad + "," + badToGood + "," + lossGood + "," + lossBad;
        }
    }

    /**
     * Parses {@code none}, {@code bernoulli:RHO}, {@code gilbert:RHO,BURST} or
     * {@code ge:GOOD_TO_BAD,BAD_TO_GOOD,LOSS_GOOD,LOSS_BAD}.
     */
    static ChannelModel parse(String spec) {
        String s = spec.trim().toLowerCase(Locale.ROOT);
        int colon = s.indexOf(':');
        String kind = colon < 0 ? s : s.substring(0, colon);
        double[] a = colon < 0 ? new double[0] : numbers(s.substring(colon + 1));
        switch (kind) {
            case "none":
                if (a.length == 0) {
                    return new Perfect();
                }
                break;
            case "bernoulli":
                if (a.length == 1) {
                    return new Bernoulli(a[0]);
                }
                break;
            case "gilbert":
                if (a.length == 2) {
                    return GilbertElliott.gilbert(a[0], a[1]);
                }
                break;
            case "ge":
                if (a.length == 4) {
                    return new GilbertElliott(a[0], a[1], a[2], a[3]);
                }
                break;
            default:
                break;
        }
        throw new IllegalArgumentException("bad channel '" + spec + "': expected none, bernoulli:RHO, gilbert:RHO,BURST"
                + " or ge:GOOD_TO_BAD,BAD_TO_GOOD,LOSS_GOOD,LOSS_BAD");
    }

    private static double[] numbers(String csv) {
        String[] parts = csv.split(",");
        double[] out = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            try {
                out[i] = Double.parseDouble(parts[i].trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("not a number: '" + parts[i] + "'", e);
            }
        }
        return out;
    }
}
