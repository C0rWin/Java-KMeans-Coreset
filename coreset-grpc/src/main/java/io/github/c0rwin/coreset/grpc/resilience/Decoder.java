package io.github.c0rwin.coreset.grpc.resilience;

import java.util.Locale;

/** How received records of a message are reweighted to stand for the whole message. */
public enum Decoder {

    /** Weights as sent: the non-resilient baseline, biased low under loss. */
    AS_SENT("as-sent"),

    /** Horvitz-Thompson with a known loss probability rho: every weight times 1 / (1 - rho) (thm:poisson-erasure). */
    HORVITZ_THOMPSON("ht"),

    /**
     * Realized count: weights times (draws sent) / (draws received), counted from the header and the delivered
     * records (thm:erasure-invariance); pooled over the replicas of a rateless group (thm:rateless).
     */
    REALIZED_COUNT("realized"),

    /** Realized mass: weights (and per-cell offsets) times (record weight sent) / (record weight received). */
    REALIZED_WEIGHT("realized-weight");

    private final String id;

    Decoder(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    public static Decoder fromId(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        for (Decoder d : values()) {
            if (d.id.equals(s)) {
                return d;
            }
        }
        throw new IllegalArgumentException("unknown decoder: " + id + " (expected as-sent, ht, realized or "
                + "realized-weight)");
    }
}
