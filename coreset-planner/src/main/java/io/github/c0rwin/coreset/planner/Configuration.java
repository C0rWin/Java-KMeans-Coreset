package io.github.c0rwin.coreset.planner;

import java.util.Objects;

/**
 * One point of the planner's search space. {@code m} is the total number of draws (or mean-coreset cells) the
 * leaves emit before oversampling; the per-node share is {@code ceil(m / units)}.
 *
 * @param family coreset construction
 * @param packetization how draws become records
 * @param splitBudget per-record count budget for {@link Packetization#SPLIT} (ignored otherwise)
 * @param transport reliable stream or datagrams, with or without FEC and oversampling
 * @param fecData data packets per FEC block ({@link Transport#DATAGRAM_FEC} only)
 * @param fecParity parity packets per FEC block ({@link Transport#DATAGRAM_FEC} only)
 * @param redundancy replication scheme
 * @param redundancyLevel r for post-compute replication and fractional repetition, mu for rateless replicas
 * @param ratelessSpare spare members per rateless group (group size mu + spare)
 * @param arity tree arity; 0 is the flat topology (all nodes send straight to the coordinator)
 * @param checkpoint internal tree nodes copy their summary to one peer before forwarding
 * @param stragglers how the coordinator treats stragglers when there is no redundancy to absorb them
 * @param m total target draws or cells
 */
public record Configuration(Family family, Packetization packetization, int splitBudget, Transport transport,
                            int fecData, int fecParity, Redundancy redundancy, int redundancyLevel, int ratelessSpare,
                            int arity, boolean checkpoint, StragglerPolicy stragglers, int m) {

    public enum Family { UNIFORM, SENSITIVITY, LIGHTWEIGHT, MEAN;
        public boolean sampling() {
            return this != MEAN;
        }
    }

    /** PER_DRAW, MERGED and SPLIT apply to sampling families; CELLS and CELLS_CF to the mean coreset. */
    public enum Packetization { PER_DRAW, MERGED, SPLIT, CELLS, CELLS_CF }

    /**
     * RELIABLE: TCP/gRPC stream. DATAGRAM: one shot, no recovery. DATAGRAM_OVERSAMPLE: one shot with enough extra
     * draws for the channel quantile. DATAGRAM_FEC: systematic block code (data, parity) plus oversampling of the
     * residual. DATAGRAM_RAPTORQ: systematic fountain code (RFC 6330) with repair symbols sized for the channel
     * quantile. EARLY_CLOSE: per-draw streaming until the receiver acknowledges the target count (LTP-style early
     * close; prop:a-stream).
     */
    public enum Transport { RELIABLE, DATAGRAM, DATAGRAM_OVERSAMPLE, DATAGRAM_FEC, DATAGRAM_RAPTORQ, EARLY_CLOSE;
        public boolean datagram() {
            return this != RELIABLE;
        }
    }

    public enum Redundancy { NONE, POST_REPLICATION, FRACTIONAL_REPETITION, RATELESS }

    public enum StragglerPolicy { WAIT, DROP }

    public Configuration {
        Objects.requireNonNull(family, "family");
        Objects.requireNonNull(packetization, "packetization");
        Objects.requireNonNull(transport, "transport");
        Objects.requireNonNull(redundancy, "redundancy");
        Objects.requireNonNull(stragglers, "stragglers");
        if (family.sampling() == (packetization == Packetization.CELLS || packetization == Packetization.CELLS_CF)) {
            throw new IllegalArgumentException(packetization + " does not apply to " + family);
        }
        if (m < 1 || arity < 0 || arity == 1) {
            throw new IllegalArgumentException("need m >= 1 and arity 0 or >= 2: m=" + m + " arity=" + arity);
        }
        if (transport == Transport.DATAGRAM_FEC && (fecData < 1 || fecParity < 1)) {
            throw new IllegalArgumentException("FEC needs data and parity packets >= 1");
        }
        if (redundancy != Redundancy.NONE && redundancyLevel < 1) {
            throw new IllegalArgumentException(redundancy + " needs level >= 1");
        }
        if (packetization == Packetization.SPLIT && splitBudget < 2) {
            throw new IllegalArgumentException("split budget must be >= 2 (1 is per-draw)");
        }
    }

    public Configuration withM(int newM) {
        return new Configuration(family, packetization, splitBudget, transport, fecData, fecParity, redundancy,
                redundancyLevel, ratelessSpare, arity, checkpoint, stragglers, newM);
    }

    public boolean oversamples() {
        return transport == Transport.DATAGRAM_OVERSAMPLE || transport == Transport.DATAGRAM_FEC;
    }

    /** Short human-readable form, e.g. {@code sensitivity/per-draw/datagram+fec(16,4)/post-r1/flat m=4000}. */
    public String describe() {
        StringBuilder b = new StringBuilder();
        b.append(family.name().toLowerCase()).append('/').append(packetization.name().toLowerCase().replace('_', '-'));
        if (packetization == Packetization.SPLIT) {
            b.append("(beta=").append(splitBudget).append(')');
        }
        b.append('/').append(switch (transport) {
            case RELIABLE -> "reliable";
            case DATAGRAM -> "datagram";
            case DATAGRAM_OVERSAMPLE -> "datagram+oversample";
            case DATAGRAM_FEC -> "datagram+fec(" + fecData + "," + fecParity + ")+oversample";
            case DATAGRAM_RAPTORQ -> "datagram+raptorq";
            case EARLY_CLOSE -> "datagram+early-close";
        });
        b.append('/').append(switch (redundancy) {
            case NONE -> "no-redundancy";
            case POST_REPLICATION -> "post-r" + redundancyLevel;
            case FRACTIONAL_REPETITION -> "fr-r" + redundancyLevel;
            case RATELESS -> "rateless-mu" + redundancyLevel + "+" + ratelessSpare;
        });
        b.append('/').append(arity == 0 ? "flat" : "tree-a" + arity + (checkpoint ? "+ckpt" : ""));
        b.append("/stragglers-").append(stragglers.name().toLowerCase());
        b.append(" m=").append(m);
        return b.toString();
    }
}
