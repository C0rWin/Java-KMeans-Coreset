/*
 * BEGIN_COPYRIGHT
 *
 *   IBM Confidential
 *   OCO Source Materials
 *
 *   5727-I17
 *   (C) Copyright IBM Corp. 2011, 2016 All Rights Reserved.
 *
 *   The source code for this program is not published or otherwise
 *   divested of its trade secrets, irrespective of what has been
 *   deposited with the U.S. Copyright Office.
 *
 *  END_COPYRIGHT
 *
 */

package univ.ml.sparse.algorithm;

import java.io.Serializable;
import java.util.Collection;
import java.util.List;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.MathUtils;

import com.google.common.collect.Lists;
import com.google.common.collect.Sets;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.SparseClusterable;
import univ.ml.sparse.SparseClusterer;
import univ.ml.sparse.SparseWeightableVector;

public class SparseWeightedKMeansPlusPlus implements SparseClusterer, Serializable {

    private static final long serialVersionUID = -4153678824717933803L;

    private final int maxIterations;

    private final int k;

    /**
     * Random generator for choosing initial centers.
     */
    private final RandomGenerator random = new JDKRandomGenerator();

    private final SparseSeedingAlgorithm seedAlg;


    public SparseWeightedKMeansPlusPlus(final int k) {
        this(k, Integer.MAX_VALUE);
    }

    public SparseWeightedKMeansPlusPlus(final int k, final int maxIter) {
//        System.out.println("KKKK: Initialize kmeans w/ K == " + k);
        this.k = k;
        this.maxIterations = maxIter;
        seedAlg = new KMeansPlusPlusSeed(k);
    }

    @Override
    public List<SparseCentroidCluster> cluster(List<SparseWeightableVector> points)
            throws MathIllegalArgumentException, ConvergenceException {
        // sanity checks
        MathUtils.checkNotNull(points);

        // number of clusters has to be smaller or equal the number of data points
        if (points.size() < k) {
            throw new NumberIsTooSmallException(points.size(), k, false);
        }

        // Once passing previous check we know that we have at least k points
        int D = points.get(0).getDimension();

        if (Sets.newHashSet(points).size() < k) {
            throw new NumberIsTooSmallException(points.size(), k, false);
        }

        // create the initial clusters
        List<SparseCentroidCluster> clusters = seedAlg.seed(points);

        // create an array containing the latest assignment of a point to a cluster
        // no need to initialize the array, as it will be filled with the first assignment
        int[] assignments = new int[points.size()];
        final int max = (maxIterations < 0) ? Integer.MAX_VALUE : maxIterations;

        for (int count = 0; count < max; count++) {
/*
            double energy = 0d;

            for (final SparseCentroidCluster cluster : clusters) {
                for (SparseWeightableVector vector : cluster.getPoints()) {
                    energy += Math.pow(cluster.getDistanceToCenter(vector), 2);
                }
            }

            System.out.println("Energy at round #" + count + " is " + energy
                    + ", first cluster center is = " + clusters.get(0).getCenter().getVector().toString()
                    + ", cluster size is " + clusters.get(0).getPoints().size());
*/

            final List<SparseCentroidCluster> newClusters = Lists.newArrayList();

            for (SparseCentroidCluster cluster : clusters) {
                newClusters.add(new SparseCentroidCluster(centroidOf(cluster.getPoints(), D)));
            }

            int changes = assignPointsToClusters(newClusters, points, assignments);
            clusters = newClusters;

            // if there were no more changes in the point-to-cluster assignment
            if (changes == 0) {
                break;
            }
        }
        return clusters;
    }

    private SparseClusterable centroidOf(List<SparseWeightableVector> points, int dimension) {
        SparseWeightableVector centroid = new SparseWeightableVector(dimension);
        double totalWeight = 0;
        for (final SparseWeightableVector p : points) {
            SparseWeightableVector newCentroid = new SparseWeightableVector(centroid.add(p.mapMultiply(p.getWeight())));
            centroid = newCentroid;
            totalWeight += p.getWeight();
        }
        return new SparseWeightableVector(centroid.mapDivideToSelf(totalWeight));
    }

    private int assignPointsToClusters(List<SparseCentroidCluster> clusters,
                                       Collection<SparseWeightableVector> points, int[] assignments) {
        int assignedDifferently = 0;
        int pointIndex = 0;
        for (final SparseWeightableVector p : points) {
            int clusterIndex = getNearestCluster(clusters, p);
            if (clusterIndex != assignments[pointIndex]) {
                assignedDifferently++;
            }

            SparseCentroidCluster cluster = clusters.get(clusterIndex);
            cluster.addPoint(p);
            assignments[pointIndex++] = clusterIndex;
        }

        return assignedDifferently;

    }

    private int getNearestCluster(List<SparseCentroidCluster> clusters, SparseWeightableVector point) {
        double minDistance = Double.MAX_VALUE;
        int minCluster = 0;

        for (int i = 0; i < clusters.size(); i++) {
            final RealVector center = clusters.get(i).getCenter().getVector();
            final double d = center.getDistance(point.getVector()) * point.getWeight();
            if (minDistance > d) {
                minDistance = d;
                minCluster = i;
            }
        }
        return minCluster;
    }
}