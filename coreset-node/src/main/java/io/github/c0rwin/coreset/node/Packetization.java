package io.github.c0rwin.coreset.node;

import java.util.Locale;

/**
 * How a node turns its summary into the records of a snapshot message ({@link NodeMessage}). Only the per-draw and
 * per-cell forms keep their guarantees when packets are lost; the others are the baselines they are measured against.
 */
public enum Packetization {

    /** The summary's points as they are, in tree order: the non-resilient baseline (default). */
    SUMMARY("summary"),

    /** m sensitivity draws of the summary, one record per distinct point carrying its draw count (tag). */
    MERGED("merged"),

    /**
     * The same m draws, one record per draw in a seeded random order (thm:erasure-invariance): any delivered subset is
     * again an i.i.d. sample, and the realized-count decoder keeps it unbiased.
     */
    PER_DRAW("per-draw"),

    /** Mean coreset of the summary over m cells, with the summed offset in the header (prop:cf-triples (c)). */
    MEAN_GLOBAL_OFFSET("mean-global"),

    /** Mean coreset over m cells with one (mean, weight, offset) triple per record (prop:cf-triples (b)). */
    MEAN_CF_TRIPLES("mean-cf");

    private final String id;

    Packetization(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    /** True when records are sensitivity draws (MERGED, PER_DRAW). */
    public boolean sampled() {
        return this == MERGED || this == PER_DRAW;
    }

    /** True when records are mean-coreset cells (MEAN_GLOBAL_OFFSET, MEAN_CF_TRIPLES). */
    public boolean cells() {
        return this == MEAN_GLOBAL_OFFSET || this == MEAN_CF_TRIPLES;
    }

    /** Parses {@link #id()} (case-insensitive). */
    public static Packetization fromId(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        for (Packetization p : values()) {
            if (p.id.equals(s)) {
                return p;
            }
        }
        throw new IllegalArgumentException("unknown packetization: " + id
                + " (expected summary, merged, per-draw, mean-global or mean-cf)");
    }
}
