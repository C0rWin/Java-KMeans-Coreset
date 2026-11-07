package io.github.c0rwin.coreset.core.algo;

import io.github.c0rwin.coreset.core.SparseVector;
import java.util.List;

/**
 * A center set together with each input point's nearest center and squared distance to it, as produced by a
 * {@link Seeding}. Points are in input order; zero-weight points are assigned too.
 */
public final class Clustering {

    private final List<SparseVector> centers;
    final int[] assignment;
    final double[] squaredDistances;
    private final double cost;
    // Per-center sum of positive weights and of w * squaredDistance, or null when not supplied.
    final double[] clusterWeights;
    final double[] clusterCosts;

    Clustering(List<SparseVector> centers, int[] assignment, double[] squaredDistances, double cost) {
        this(centers, assignment, squaredDistances, cost, null, null);
    }

    Clustering(List<SparseVector> centers, int[] assignment, double[] squaredDistances, double cost,
               double[] clusterWeights, double[] clusterCosts) {
        this.centers = List.copyOf(centers);
        this.assignment = assignment;
        this.squaredDistances = squaredDistances;
        this.cost = cost;
        this.clusterWeights = clusterWeights;
        this.clusterCosts = clusterCosts;
    }

    public List<SparseVector> centers() {
        return centers;
    }

    /** Index into {@link #centers()} per input point, or -1 for every point when there are no centers; a copy. */
    public int[] assignment() {
        return assignment.clone();
    }

    /** Squared distance from each input point to its assigned center; a copy. */
    public double[] squaredDistances() {
        return squaredDistances.clone();
    }

    /** {@code sum_p w_p * squaredDistance_p}. */
    public double cost() {
        return cost;
    }
}
