package io.github.c0rwin.coreset.node;

import java.util.Locale;

/** How a session's workers organise merge-and-reduce; see {@link CoresetNode} for the tradeoffs. */
public enum TreeMode {

    /** Each worker owns a private {@code MergeReduceTree}; snapshots are always consistent. */
    SINGLE_WRITER("single-writer"),

    /** All workers share one lock-free tree of CAS slots; snapshots are validated optimistically. */
    CAS("cas");

    private final String id;

    TreeMode(String id) {
        this.id = id;
    }

    public String id() {
        return id;
    }

    /** Parses {@link #id()} (case-insensitive). */
    public static TreeMode fromId(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        for (TreeMode m : values()) {
            if (m.id.equals(s)) {
                return m;
            }
        }
        throw new IllegalArgumentException("unknown tree mode: " + id + " (expected single-writer or cas)");
    }
}
