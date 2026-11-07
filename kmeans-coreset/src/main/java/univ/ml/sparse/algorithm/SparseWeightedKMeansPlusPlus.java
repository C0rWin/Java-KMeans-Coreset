package univ.ml.sparse.algorithm;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.math3.exception.NotStrictlyPositiveException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.linear.OpenMapRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseClusterer;
import univ.ml.sparse.SparseWeightableVector;

/**
 * Weighted k-means: k-means++ seeding (first center drawn proportionally to weight, the next ones
 * proportionally to weight times squared distance to the nearest center) followed by weighted Lloyd
 * iterations.
 *
 * <p>Guarantees on the returned clusters:
 * <ul>
 * <li>they partition the input and hold the input objects themselves;</li>
 * <li>none is empty: a cluster left without points is reseeded with the point that currently adds the
 * most to the cost, or dropped when every point already sits on its center;</li>
 * <li>each center is the weighted mean of its points, or the point the cluster was seeded with if all of
 * its points weigh zero, so centers are never NaN;</li>
 * <li>there are fewer than k of them when the input has fewer than k distinct points.</li>
 * </ul>
 */
public class SparseWeightedKMeansPlusPlus implements SparseClusterer, Serializable {

    private static final long serialVersionUID = 2186434757431342771L;

    private final int k;

    private final int maxIterations;

    private final RandomGenerator random = new JDKRandomGenerator();

    /**
     * @param k number of clusters
     */
    public SparseWeightedKMeansPlusPlus(final int k) {
        this(k, -1);
    }

    /**
     * @param k             number of clusters
     * @param maxIterations maximal number of Lloyd iterations after seeding; a negative value iterates
     *                      until no point changes cluster
     */
    public SparseWeightedKMeansPlusPlus(final int k, final int maxIterations) {
        if (k <= 0) {
            throw new NotStrictlyPositiveException(k);
        }
        this.k = k;
        this.maxIterations = maxIterations;
    }

    @Override
    public List<SparseCentroidCluster> cluster(final List<SparseWeightableVector> points) {
        if (points.isEmpty()) {
            throw new NumberIsTooSmallException(0, 1, true);
        }

        final SparseWeightableVector[] centers = seed(points);
        final int n = points.size();
        final int[] assignment = new int[n];
        // Weighted squared distance of each point to the center it is assigned to.
        final double[] cost = new double[n];
        final double[] mass = new double[centers.length];

        Arrays.fill(assignment, -1);
        assign(points, centers, assignment, cost);
        for (int iteration = 0; ; iteration++) {
            updateCenters(points, centers, assignment, mass);
            if (maxIterations >= 0 && iteration >= maxIterations) {
                break;
            }
            if (!assign(points, centers, assignment, cost)) {
                break;
            }
        }

        final List<SparseCentroidCluster> clusters = new ArrayList<>(centers.length);
        final SparseCentroidCluster[] byCenter = new SparseCentroidCluster[centers.length];
        for (int j = 0; j < centers.length; j++) {
            if (centers[j] != null) {
                // A zero-mass cluster still points at an input object; hand out a copy instead.
                final SparseWeightableVector center = mass[j] > 0 ? centers[j] : new SparseWeightableVector(centers[j], 1d);
                byCenter[j] = new SparseCentroidCluster(center);
                clusters.add(byCenter[j]);
            }
        }
        for (int i = 0; i < n; i++) {
            byCenter[assignment[i]].addPoint(points.get(i));
        }
        return clusters;
    }

    private SparseWeightableVector[] seed(final List<SparseWeightableVector> points) {
        final int n = points.size();
        final List<SparseWeightableVector> centers = new ArrayList<>(Math.min(k, n));
        final double[] mass = new double[n];

        double total = 0d;
        for (int i = 0; i < n; i++) {
            mass[i] = points.get(i).getWeight();
            total += mass[i];
        }
        SparseWeightableVector center = points.get(total > 0 ? sample(mass, total) : random.nextInt(n));
        centers.add(center);

        total = 0d;
        for (int i = 0; i < n; i++) {
            final SparseWeightableVector p = points.get(i);
            mass[i] = p.getWeight() * squaredDistance(p, center);
            total += mass[i];
        }
        // Points on a center carry no mass, so once it is all gone only duplicates would remain.
        while (centers.size() < k && total > 0) {
            center = points.get(sample(mass, total));
            centers.add(center);

            total = 0d;
            for (int i = 0; i < n; i++) {
                final SparseWeightableVector p = points.get(i);
                mass[i] = Math.min(mass[i], p.getWeight() * squaredDistance(p, center));
                total += mass[i];
            }
        }
        return centers.toArray(new SparseWeightableVector[0]);
    }

    /** Draws index i with probability mass[i] / total; total must be positive. */
    private int sample(final double[] mass, final double total) {
        final double target = random.nextDouble() * total;
        double cdf = 0d;
        int last = -1;
        for (int i = 0; i < mass.length; i++) {
            if (mass[i] <= 0) {
                continue;
            }
            last = i;
            cdf += mass[i];
            if (cdf >= target) {
                return i;
            }
        }
        // Rounding can leave cdf just below target.
        return last;
    }

    /**
     * Moves every point to its nearest center, then repairs empty clusters.
     *
     * @return whether any assignment or center changed
     */
    private boolean assign(final List<SparseWeightableVector> points, final SparseWeightableVector[] centers,
                           final int[] assignment, final double[] cost) {
        boolean changed = false;
        final int[] size = new int[centers.length];
        for (int i = 0; i < points.size(); i++) {
            final SparseWeightableVector p = points.get(i);
            int best = assignment[i];
            double bestDist = best < 0 ? Double.POSITIVE_INFINITY : squaredDistance(p, centers[best]);
            for (int j = 0; j < centers.length; j++) {
                if (j == assignment[i] || centers[j] == null) {
                    continue;
                }
                final double d = squaredDistance(p, centers[j]);
                // Switch only when strictly closer, so ties cannot make points oscillate.
                if (d < bestDist) {
                    bestDist = d;
                    best = j;
                }
            }
            if (best != assignment[i]) {
                assignment[i] = best;
                changed = true;
            }
            cost[i] = p.getWeight() * bestDist;
            size[best]++;
        }

        for (int j = 0; j < centers.length; j++) {
            if (centers[j] == null || size[j] > 0) {
                continue;
            }
            int farthest = -1;
            for (int i = 0; i < points.size(); i++) {
                if (cost[i] > 0 && size[assignment[i]] > 1 && (farthest < 0 || cost[i] > cost[farthest])) {
                    farthest = i;
                }
            }
            if (farthest < 0) {
                centers[j] = null;
            } else {
                size[assignment[farthest]]--;
                assignment[farthest] = j;
                size[j] = 1;
                cost[farthest] = 0d;
                centers[j] = points.get(farthest);
            }
            changed = true;
        }
        return changed;
    }

    /** Sets each center to the weighted mean of its points; clusters of zero mass keep their center. */
    private static void updateCenters(final List<SparseWeightableVector> points, final SparseWeightableVector[] centers,
                                      final int[] assignment, final double[] mass) {
        final int dimension = points.get(0).getDimension();
        final WeightedSum[] sums = new WeightedSum[centers.length];
        Arrays.fill(mass, 0d);

        for (int i = 0; i < points.size(); i++) {
            final SparseWeightableVector p = points.get(i);
            final double w = p.getWeight();
            if (w == 0) {
                continue;
            }
            final int j = assignment[i];
            if (sums[j] == null) {
                sums[j] = new WeightedSum(dimension);
            }
            mass[j] += w;
            sums[j].add(p, w);
        }

        for (int j = 0; j < centers.length; j++) {
            if (centers[j] != null && mass[j] > 0) {
                centers[j] = sums[j].divide(mass[j]);
            }
        }
    }

    /** Sparse running sum; subclassing gives access to the sparse entry iterator type. */
    private static final class WeightedSum extends OpenMapRealVector {

        private static final long serialVersionUID = 1L;

        WeightedSum(final int dimension) {
            super(dimension);
        }

        void add(final RealVector p, final double weight) {
            for (Iterator<Entry> it = p.sparseIterator(); it.hasNext(); ) {
                final Entry e = it.next();
                addToEntry(e.getIndex(), weight * e.getValue());
            }
        }

        SparseWeightableVector divide(final double divisor) {
            final SparseWeightableVector result = new SparseWeightableVector(getDimension());
            for (Iterator<Entry> it = sparseIterator(); it.hasNext(); ) {
                final Entry e = it.next();
                result.setEntry(e.getIndex(), e.getValue() / divisor);
            }
            return result;
        }
    }

    private static double squaredDistance(final SparseWeightableVector p, final SparseWeightableVector center) {
        final double d = p.getDistance(center);
        return d * d;
    }
}
