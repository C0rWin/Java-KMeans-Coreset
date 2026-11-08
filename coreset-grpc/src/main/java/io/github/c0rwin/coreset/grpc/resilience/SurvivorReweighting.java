package io.github.c0rwin.coreset.grpc.resilience;

import java.util.Locale;

/** How the coordinator scales the surviving groups' messages when whole groups are lost (prop:ht-unbiased). */
public enum SurvivorReweighting {

    /** No scaling: the output summarizes the survivors only (composability alone). */
    NONE("none"),

    /** Node-realized estimator: times groups / delivered groups. */
    NODE_FRACTION("nodes"),

    /**
     * Realized surviving weight: times (total weight, from a {@link Census} taken before failures) / (weight in the
     * delivered headers).
     */
    REALIZED_WEIGHT("weight");

    private final String id;

    SurvivorReweighting(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    public static SurvivorReweighting fromId(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        for (SurvivorReweighting r : values()) {
            if (r.id.equals(s)) {
                return r;
            }
        }
        throw new IllegalArgumentException("unknown reweighting: " + id + " (expected none, nodes or weight)");
    }
}
