package io.github.c0rwin.coreset.grpc.resilience;

import java.util.Arrays;
import java.util.Locale;
import java.util.Objects;

/**
 * How nodes are grouped around the same data. Every member of a group receives every batch routed to the group and is
 * created with the group's partition, so members hold the same summary when their batches arrive in the same order
 * (one submitting thread, no BUSY retries reordering them).
 *
 * @param mode   decoding of a group's messages
 * @param groups node indices per group; every node in exactly one group
 */
public record GroupLayout(Mode mode, int[][] groups) {

    public enum Mode {
        /** One node per group: each node holds its own share (the default). */
        INDEPENDENT("independent"),
        /**
         * Fractional repetition (thm:fr-optimal): members share the sampling stream, so their messages are
         * identical and any one suffices; the decoder takes each packet from whichever member delivered it.
         */
        FRACTIONAL_REPETITION("fr"),
        /**
         * Rateless replicas (thm:rateless): members share q but sample with their own stream, and the decoder pools
         * every delivered draw of the group.
         */
        RATELESS("rateless");

        private final String id;

        Mode(String id) {
            this.id = id;
        }

        public String id() {
            return id;
        }

        public static Mode fromId(String id) {
            String s = id.toLowerCase(Locale.ROOT);
            for (Mode m : values()) {
                if (m.id.equals(s)) {
                    return m;
                }
            }
            throw new IllegalArgumentException("unknown layout: " + id + " (expected independent, fr or rateless)");
        }
    }

    public GroupLayout {
        Objects.requireNonNull(mode, "mode");
        groups = groups.clone();
        int nodes = 0;
        for (int g = 0; g < groups.length; g++) {
            groups[g] = groups[g].clone();
            if (groups[g].length == 0 || mode == Mode.INDEPENDENT && groups[g].length != 1) {
                throw new IllegalArgumentException("group " + g + " has " + groups[g].length + " members");
            }
            nodes += groups[g].length;
        }
        boolean[] seen = new boolean[nodes];
        for (int[] members : groups) {
            for (int node : members) {
                if (node < 0 || node >= nodes || seen[node]) {
                    throw new IllegalArgumentException("nodes must be 0.." + (nodes - 1) + ", each in one group");
                }
                seen[node] = true;
            }
        }
    }

    public static GroupLayout independent(int nodes) {
        return contiguous(Mode.INDEPENDENT, nodes, nodes);
    }

    /**
     * {@code floor(nodes / (stragglers + 1))} groups of contiguous nodes with sizes between {@code stragglers + 1} and
     * {@code 2 * stragglers + 1}, so any {@code stragglers} slow nodes leave every group a member.
     */
    public static GroupLayout fractionalRepetition(int nodes, int stragglers) {
        if (stragglers < 0 || stragglers >= nodes) {
            throw new IllegalArgumentException("need 0 <= stragglers < nodes");
        }
        return contiguous(Mode.FRACTIONAL_REPETITION, nodes, nodes / (stragglers + 1));
    }

    /** {@code floor(nodes / groupSize)} groups of contiguous nodes, the remainder spread one per group. */
    public static GroupLayout rateless(int nodes, int groupSize) {
        if (groupSize < 1 || groupSize > nodes) {
            throw new IllegalArgumentException("need 1 <= groupSize <= nodes");
        }
        return contiguous(Mode.RATELESS, nodes, nodes / groupSize);
    }

    private static GroupLayout contiguous(Mode mode, int nodes, int count) {
        if (nodes < 1) {
            throw new IllegalArgumentException("nodes must be >= 1: " + nodes);
        }
        int[][] groups = new int[count][];
        for (int g = 0, next = 0; g < count; g++) {
            int size = nodes / count + (g < nodes % count ? 1 : 0);
            groups[g] = new int[size];
            for (int i = 0; i < size; i++) {
                groups[g][i] = next++;
            }
        }
        return new GroupLayout(mode, groups);
    }

    public int nodes() {
        int n = 0;
        for (int[] g : groups) {
            n += g.length;
        }
        return n;
    }

    public int groupCount() {
        return groups.length;
    }

    /** Members of group {@code g} (a copy). */
    public int[] members(int g) {
        return groups[g].clone();
    }

    public int groupOf(int node) {
        for (int g = 0; g < groups.length; g++) {
            for (int m : groups[g]) {
                if (m == node) {
                    return g;
                }
            }
        }
        throw new IllegalArgumentException("node " + node + " not in the layout");
    }

    /** Session partition of {@code node}: its group, so members derive the same summary randomness. */
    public long partitionOf(int node) {
        return groupOf(node);
    }

    /** Sampling stream of {@code node}'s message: its rank in the group for rateless groups, else 0. */
    public long streamOf(int node) {
        if (mode != Mode.RATELESS) {
            return 0;
        }
        int[] members = groups[groupOf(node)];
        for (int i = 0; i < members.length; i++) {
            if (members[i] == node) {
                return i;
            }
        }
        throw new AssertionError();
    }

    @Override
    public int[][] groups() {
        int[][] out = new int[groups.length][];
        for (int g = 0; g < groups.length; g++) {
            out[g] = groups[g].clone();
        }
        return out;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof GroupLayout l && mode == l.mode && Arrays.deepEquals(groups, l.groups);
    }

    @Override
    public int hashCode() {
        return 31 * mode.hashCode() + Arrays.deepHashCode(groups);
    }

    @Override
    public String toString() {
        return mode.id() + Arrays.deepToString(groups);
    }
}
