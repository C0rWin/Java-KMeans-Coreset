package io.github.c0rwin.coreset.planner;

import java.util.Objects;

/**
 * The network and the cluster the plan runs on.
 *
 * @param rho stationary packet erasure rate of every link, in [0, 1)
 * @param meanBurst mean length of an erasure burst in packets (1 = memoryless); with {@code rho} it fixes a Gilbert
 *     channel, see {@link #gilbertElliott()}
 * @param nodes number of worker nodes holding data
 * @param failPre probability that a node crashes before its local summary is computed or replicated (F_pre)
 * @param failPost probability that a node crashes after computing and replicating, before upload (F_post)
 * @param partition how points are assigned to nodes
 * @param stragglers number of nodes that answer late
 * @param stragglerDelaySeconds extra delay of a straggler
 * @param rttSeconds round-trip time of every link
 * @param bandwidthBytesPerSecond capacity of every link and of the coordinator's ingress
 * @param maxMessageBytes largest datagram (or gRPC message) payload, headers included
 * @param dataPreReplicated input already lives on r+1 nodes, so data-level redundancy costs no placement traffic
 */
public record EnvironmentProfile(double rho, double meanBurst, int nodes, double failPre, double failPost,
                                 Partition partition, int stragglers, double stragglerDelaySeconds,
                                 double rttSeconds, double bandwidthBytesPerSecond, int maxMessageBytes,
                                 boolean dataPreReplicated) {

    public enum Partition { RANDOM, ADVERSARIAL }

    /**
     * Two-state chain with loss probability 1 in the bad state and 0 in the good one. {@code pGoodToBad} and
     * {@code pBadToGood} are transition probabilities per packet; {@code lambda = 1 - pGB - pBG} is the second
     * eigenvalue used by cor:loss-eps(d).
     */
    public record GilbertElliott(double pGoodToBad, double pBadToGood) {
        public double lossRate() {
            double s = pGoodToBad + pBadToGood;
            return s == 0.0 ? 0.0 : pGoodToBad / s;
        }

        public double lambda() {
            return 1.0 - pGoodToBad - pBadToGood;
        }
    }

    public EnvironmentProfile {
        Objects.requireNonNull(partition, "partition");
        if (!(rho >= 0.0 && rho < 1.0) || !(meanBurst >= 1.0) || nodes < 1 || !(failPre >= 0.0 && failPre < 1.0)
                || !(failPost >= 0.0 && failPost < 1.0) || stragglers < 0 || stragglers >= nodes
                || !(stragglerDelaySeconds >= 0.0) || !(rttSeconds >= 0.0) || !(bandwidthBytesPerSecond > 0.0)
                || maxMessageBytes < 64) {
            throw new IllegalArgumentException("invalid environment: " + this);
        }
        if (rho > 0.0 && meanBurst > 1.0 && rho / (1.0 - rho) / meanBurst >= 1.0) {
            throw new IllegalArgumentException("rho=" + rho + " with mean burst " + meanBurst
                    + " needs a good-to-bad probability >= 1");
        }
    }

    /** A loss-free, failure-free LAN-like default: 16 nodes, 0.5 ms RTT, 1 Gbit/s, 1400-byte datagrams. */
    public static EnvironmentProfile lan(int nodes) {
        return new EnvironmentProfile(0.0, 1.0, nodes, 0.0, 0.0, Partition.RANDOM, 0, 0.0, 5e-4, 125e6, 1400, false);
    }

    public EnvironmentProfile withRho(double newRho) {
        return new EnvironmentProfile(newRho, meanBurst, nodes, failPre, failPost, partition, stragglers,
                stragglerDelaySeconds, rttSeconds, bandwidthBytesPerSecond, maxMessageBytes, dataPreReplicated);
    }

    /** Memoryless when {@code meanBurst == 1}: then pBG = 1 - rho and pGB = rho, so lambda = 0. */
    public GilbertElliott gilbertElliott() {
        double pBG = meanBurst <= 1.0 ? 1.0 - rho : 1.0 / meanBurst;
        double pGB = rho == 0.0 ? 0.0 : (meanBurst <= 1.0 ? rho : rho * pBG / (1.0 - rho));
        return new GilbertElliott(pGB, pBG);
    }

    /** Probability that a node is unavailable at upload time: crashed in either phase. */
    public double failTotal() {
        return 1.0 - (1.0 - failPre) * (1.0 - failPost);
    }
}
