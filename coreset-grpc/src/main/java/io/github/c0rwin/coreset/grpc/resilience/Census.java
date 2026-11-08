package io.github.c0rwin.coreset.grpc.resilience;

/**
 * Per-node input weight and points from message headers, taken before failures so that survivor reweighting knows
 * the total (prop:ht-unbiased: "available if counts are reported before the failure epoch").
 *
 * @param nodeWeight weight per node, NaN where the node did not answer
 * @param nodePoints points per node, -1 where the node did not answer
 */
public record Census(double[] nodeWeight, long[] nodePoints) {

    public Census {
        nodeWeight = nodeWeight.clone();
        nodePoints = nodePoints.clone();
        if (nodeWeight.length != nodePoints.length) {
            throw new IllegalArgumentException("one weight and one point count per node");
        }
    }

    @Override
    public double[] nodeWeight() {
        return nodeWeight.clone();
    }

    @Override
    public long[] nodePoints() {
        return nodePoints.clone();
    }

    /** Weight of group {@code g}: that of its first member that answered (members hold the same data), else NaN. */
    public double groupWeight(GroupLayout layout, int g) {
        for (int node : layout.members(g)) {
            if (node < nodeWeight.length && !Double.isNaN(nodeWeight[node])) {
                return nodeWeight[node];
            }
        }
        return Double.NaN;
    }

    /** Sum of the group weights, NaN if some group has none. */
    public double totalWeight(GroupLayout layout) {
        double total = 0.0;
        for (int g = 0; g < layout.groupCount(); g++) {
            total += groupWeight(layout, g);
        }
        return total;
    }
}
