package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import java.util.List;

/**
 * Union of per-node session snapshots.
 *
 * @param coreset union of every node's coreset (offsets added)
 * @param nodes   per-node snapshots in node order, with their completeness counters
 */
public record GatheredSnapshot(Coreset coreset, List<NodeSnapshot> nodes) {

    public GatheredSnapshot {
        nodes = List.copyOf(nodes);
    }

    /** True when every node reflects every batch it accepted. */
    public boolean complete() {
        return nodes.stream().allMatch(NodeSnapshot::complete);
    }

    public long acceptedBatches() {
        return nodes.stream().mapToLong(NodeSnapshot::acceptedBatches).sum();
    }

    public long appliedBatches() {
        return nodes.stream().mapToLong(NodeSnapshot::appliedBatches).sum();
    }

    public double pendingWeight() {
        return nodes.stream().mapToDouble(NodeSnapshot::pendingWeight).sum();
    }
}
