package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.SparseVector;
import java.util.Arrays;
import java.util.List;

/**
 * Outcome of a clustering run.
 *
 * @param centers    final centers (may be fewer than requested k when the input has fewer distinct weighted points)
 * @param assignment index into {@code centers} for each input point, or -1 when {@code centers} is empty; accessor
 *                   returns a copy
 * @param cost       weighted cost of {@code assignment} as computed by the clustering loop
 * @param iterations update steps performed
 */
public record KMeansResult(List<SparseVector> centers, int[] assignment, double cost, int iterations) {

    public KMeansResult {
        centers = List.copyOf(centers);
        assignment = assignment.clone();
    }

    @Override
    public int[] assignment() {
        return assignment.clone();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof KMeansResult r
                && centers.equals(r.centers)
                && Arrays.equals(assignment, r.assignment)
                && Double.compare(cost, r.cost) == 0
                && iterations == r.iterations;
    }

    @Override
    public int hashCode() {
        return 31 * (31 * (31 * centers.hashCode() + Arrays.hashCode(assignment)) + Double.hashCode(cost))
                + iterations;
    }

    @Override
    public String toString() {
        return "KMeansResult[centers=" + centers.size() + ", n=" + assignment.length
                + ", cost=" + cost + ", iterations=" + iterations + "]";
    }
}
