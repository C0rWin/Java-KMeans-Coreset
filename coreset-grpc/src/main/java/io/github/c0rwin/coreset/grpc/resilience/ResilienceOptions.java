package io.github.c0rwin.coreset.grpc.resilience;

import java.time.Duration;
import java.util.Objects;

/**
 * Knobs of a resilient gather ({@code CoresetCoordinator.gatherResilient}).
 *
 * @param layout        node groups and how their messages combine
 * @param decoder       per-message reweighting of the delivered records
 * @param knownLossRate loss probability assumed by {@link Decoder#HORVITZ_THOMPSON}
 * @param reweighting   scaling of the delivered groups when whole groups are lost
 * @param census        totals for {@link SurvivorReweighting#REALIZED_WEIGHT}, or null
 * @param replication   r: node i's message may also be fetched from its replica holders i+1 .. i+r (mod nodes)
 * @param awaitApplied  how long a node may wait for its accepted batches before building its message
 * @param callDeadline  deadline of one fetch; a node that misses it (and whose holders miss it) is a straggler
 * @param minResponders stop waiting once every group has this many completed messages (0: wait for all, or the
 *                      deadline); unfinished fetches are cancelled, as when the coordinator proceeds with the first
 *                      responders
 * @param streamOffset  added to every member's sampling stream: an independent sample of the same summaries (a
 *                      frozen message is served only for its own stream)
 */
public record ResilienceOptions(GroupLayout layout, Decoder decoder, double knownLossRate,
                                SurvivorReweighting reweighting, Census census, int replication,
                                Duration awaitApplied, Duration callDeadline, int minResponders, long streamOffset) {

    public ResilienceOptions {
        Objects.requireNonNull(layout, "layout");
        Objects.requireNonNull(decoder, "decoder");
        Objects.requireNonNull(reweighting, "reweighting");
        Objects.requireNonNull(awaitApplied, "awaitApplied");
        Objects.requireNonNull(callDeadline, "callDeadline");
        if (!(knownLossRate >= 0.0 && knownLossRate < 1.0)) {
            throw new IllegalArgumentException("knownLossRate must be in [0, 1): " + knownLossRate);
        }
        if (replication < 0 || replication >= layout.nodes()) {
            throw new IllegalArgumentException("replication must be in [0, nodes): " + replication);
        }
        if (minResponders < 0 || awaitApplied.isNegative() || callDeadline.isNegative() || callDeadline.isZero()) {
            throw new IllegalArgumentException("minResponders and awaitApplied >= 0, callDeadline > 0");
        }
        if (reweighting == SurvivorReweighting.REALIZED_WEIGHT && census == null) {
            throw new IllegalArgumentException("weight reweighting needs a census taken before failures");
        }
        if (streamOffset < 0) {
            throw new IllegalArgumentException("streamOffset must be >= 0: " + streamOffset);
        }
    }

    /**
     * The non-resilient baseline over {@code nodes} independent nodes: weights as sent, no survivor scaling, no
     * replicas; each node first waits up to 10 s for its accepted batches, then each fetch has 30 s.
     */
    public static ResilienceOptions baseline(int nodes) {
        return new ResilienceOptions(GroupLayout.independent(nodes), Decoder.AS_SENT, 0.0, SurvivorReweighting.NONE,
                null, 0, Duration.ofSeconds(10), Duration.ofSeconds(30), 0, 0);
    }

    public ResilienceOptions withLayout(GroupLayout v) {
        return new ResilienceOptions(v, decoder, knownLossRate, reweighting, census, replication, awaitApplied,
                callDeadline, minResponders, streamOffset);
    }

    public ResilienceOptions withDecoder(Decoder v) {
        return new ResilienceOptions(layout, v, knownLossRate, reweighting, census, replication, awaitApplied,
                callDeadline, minResponders, streamOffset);
    }

    public ResilienceOptions withKnownLossRate(double v) {
        return new ResilienceOptions(layout, decoder, v, reweighting, census, replication, awaitApplied,
                callDeadline, minResponders, streamOffset);
    }

    public ResilienceOptions withReweighting(SurvivorReweighting v, Census c) {
        return new ResilienceOptions(layout, decoder, knownLossRate, v, c, replication, awaitApplied,
                callDeadline, minResponders, streamOffset);
    }

    public ResilienceOptions withReplication(int v) {
        return new ResilienceOptions(layout, decoder, knownLossRate, reweighting, census, v, awaitApplied,
                callDeadline, minResponders, streamOffset);
    }

    public ResilienceOptions withAwaitApplied(Duration v) {
        return new ResilienceOptions(layout, decoder, knownLossRate, reweighting, census, replication, v,
                callDeadline, minResponders, streamOffset);
    }

    public ResilienceOptions withCallDeadline(Duration v) {
        return new ResilienceOptions(layout, decoder, knownLossRate, reweighting, census, replication, awaitApplied,
                v, minResponders, streamOffset);
    }

    public ResilienceOptions withMinResponders(int v) {
        return new ResilienceOptions(layout, decoder, knownLossRate, reweighting, census, replication, awaitApplied,
                callDeadline, v, streamOffset);
    }

    public ResilienceOptions withStreamOffset(long v) {
        return new ResilienceOptions(layout, decoder, knownLossRate, reweighting, census, replication, awaitApplied,
                callDeadline, minResponders, v);
    }

    /** Sampling stream requested from {@code node}. */
    public long streamOf(int node) {
        return layout.streamOf(node) + streamOffset;
    }
}
