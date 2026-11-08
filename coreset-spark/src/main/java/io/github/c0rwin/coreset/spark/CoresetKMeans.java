package io.github.c0rwin.coreset.spark;

import io.github.c0rwin.coreset.core.SparseVector;
import java.util.List;

/**
 * k-means solved on a distributed coreset and evaluated on the full data.
 *
 * @param centers       best centers over the restarts, by coreset cost
 * @param coresetCost   their cost on the coreset (offset included)
 * @param full          their cost on the full data, from one distributed pass
 * @param bestRestart   index of the winning restart
 * @param restarts      k-means++ plus weighted Lloyd restarts run on the coreset
 * @param build         the coreset and its build metrics
 * @param solveNanos    driver time of the restarts
 * @param evaluateNanos time of the full-data pass
 */
public record CoresetKMeans(List<SparseVector> centers, double coresetCost, FullDataCost full, int bestRestart,
                            int restarts, BuildResult build, long solveNanos, long evaluateNanos) {

    public CoresetKMeans {
        centers = List.copyOf(centers);
    }

    /**
     * {@code max(r, 1/r)} for {@code r = coresetCost / full.cost()}: the coreset's distortion at the solution actually
     * deployed, one query of the supremum and so a lower bound. Optimising on the coreset tends to push r below 1.
     */
    public double deployedDistortion() {
        double r = full.cost() > 0.0 ? coresetCost / full.cost() : (coresetCost == 0.0 ? 1.0 : Double.POSITIVE_INFINITY);
        return r >= 1.0 ? r : 1.0 / r;
    }
}
