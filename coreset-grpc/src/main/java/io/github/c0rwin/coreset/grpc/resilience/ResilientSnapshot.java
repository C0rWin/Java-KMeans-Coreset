package io.github.c0rwin.coreset.grpc.resilience;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.List;
import java.util.Objects;

/**
 * Result of a resilient gather: the decoded union and what happened to every group's messages.
 *
 * @param coreset       union of the decoded groups, survivor scaling applied
 * @param survivorScale factor applied for lost groups (1 when none was lost or no reweighting was asked for)
 * @param groups        per group, in group order
 */
public record ResilientSnapshot(Coreset coreset, double survivorScale, List<GroupOutcome> groups) {

    public ResilientSnapshot {
        Objects.requireNonNull(coreset, "coreset");
        groups = List.copyOf(groups);
    }

    /**
     * @param group           group index
     * @param delivered       some member's header arrived (a delivered message may still have lost every packet)
     * @param nodeWeight      input weight from the delivered header, NaN if none
     * @param decoderScale    factor the decoder put on the received weights
     * @param recordsSent     records of the decoded message(s)
     * @param recordsReceived records delivered and decoded
     * @param sources         one per member
     */
    public record GroupOutcome(int group, boolean delivered, double nodeWeight, double decoderScale, long recordsSent,
                               long recordsReceived, List<SourceOutcome> sources) {
        public GroupOutcome {
            sources = List.copyOf(sources);
        }
    }

    /**
     * @param node            member whose message was fetched
     * @param servedBy        node that served it: the member, a replica holder, or -1 if none answered
     * @param completed       the stream ended normally (not a straggler, crash or cancelled)
     * @param error           last failure on the way, empty if none
     * @param packetsSent     packets in the message header, 0 without a header
     * @param packetsReceived packets delivered
     */
    public record SourceOutcome(int node, int servedBy, boolean completed, String error, long packetsSent,
                                long packetsReceived) {
        public boolean viaReplica() {
            return servedBy >= 0 && servedBy != node;
        }
    }

    public int lostGroups() {
        int lost = 0;
        for (GroupOutcome g : groups) {
            lost += g.delivered() ? 0 : 1;
        }
        return lost;
    }

    public long packetsSent() {
        long n = 0;
        for (GroupOutcome g : groups) {
            for (SourceOutcome s : g.sources()) {
                n += s.packetsSent();
            }
        }
        return n;
    }

    public long packetsReceived() {
        long n = 0;
        for (GroupOutcome g : groups) {
            for (SourceOutcome s : g.sources()) {
                n += s.packetsReceived();
            }
        }
        return n;
    }
}
