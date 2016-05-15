package univ.ml.sparse;

import org.apache.commons.math3.exception.ConvergenceException;
import org.apache.commons.math3.exception.MathIllegalArgumentException;
import org.apache.commons.math3.exception.NumberIsTooSmallException;
import org.apache.commons.math3.exception.util.LocalizedFormats;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;
import org.apache.commons.math3.util.FastMath;
import org.apache.commons.math3.util.MathUtils;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SparseWeightedKMeansPlusPlus implements SparseClusterer {

    private final int maxIterations;

    private final int k;

    /**
     * Random generator for choosing initial centers.
     */
    private final RandomGenerator random = new JDKRandomGenerator();


    public SparseWeightedKMeansPlusPlus(final int k) {
        this(k, Integer.MAX_VALUE);
    }

    public SparseWeightedKMeansPlusPlus(final int k, final int maxIter) {
        this.k = k;
        this.maxIterations = maxIter;
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

        System.out.println(String.format("input for SparseWeightedKMeansPlusPlus::cluster are k=%d and pnts=%s", k, points.toString()));

        // create the initial clusters
        Map<Integer, SparseCentroidCluster> clusters = chooseInitialCenters(points);

        // create an array containing the latest assignment of a point to a cluster
        // no need to initialize the array, as it will be filled with the first assignment
        int[] assignments = new int[points.size()];
        assignPointsToClusters(clusters, points, assignments);

//        final SparseWSSE wsse = new SparseWSSE();
//        double cost = wsse.getCost(clusters.values());

//?System.out.println("Initial cost = " + cost);

        final int max = (maxIterations < 0) ? Integer.MAX_VALUE : maxIterations;
        for (int count = 0; count < max; count++) {
            boolean emptyCluster = false;
            Map<Integer, SparseCentroidCluster> newClusters = new HashMap<>();
            for (final Map.Entry<Integer, SparseCentroidCluster> each : clusters.entrySet()) {
                final SparseClusterable newCenter;
                final List<SparseWeightableVector> clusterPoints = each.getValue().getPoints();
                if (clusterPoints.isEmpty()) {
//                    System.out.println("XXXXX: ACHTUNG! Empty cluster detected.");
                    newCenter = getPointFromLargestVarianceCluster(clusters);
                } else {
                    final SparseClusterable center = each.getValue().getCenter();
                    newCenter = centroidOf(clusterPoints, center.getVector().getDimension());
                }
                newClusters.put(each.getKey(), new SparseCentroidCluster(newCenter));
            }
            int changes = assignPointsToClusters(newClusters, points, assignments);

//            double newCost = wsse.getCost(newClusters.values());

//            System.out.println("Round #" + count + ", changes #" + changes + ", cost = " + newCost + ", delta = " + (cost - newCost));
//            cost = newCost;
            System.out.println("Round #" + count + ", changes #" + changes);
            clusters = newClusters;

            // if there were no more changes in the point-to-cluster assignment
            // and there are no empty clusters left, return the current clusters
            if (changes == 0 && !emptyCluster) {
                return new ArrayList<>(clusters.values());
            }
        }
        return new ArrayList<>(clusters.values());

    }

    private SparseClusterable centroidOf(List<SparseWeightableVector> points, int dimension) {
        SparseWeightableVector centroid = new SparseWeightableVector(dimension);
        double overallWeight = 0;
        for (final SparseWeightableVector p : points) {
            centroid.combineToSelf(1, p.getWeight(), p);
            overallWeight += p.getWeight();
        }
        return new SparseWeightableVector(centroid.mapDivideToSelf(overallWeight));
    }

    private SparseClusterable getPointFromLargestVarianceCluster(Map<Integer, SparseCentroidCluster> clusters) {
        double maxVariance = Double.NEGATIVE_INFINITY;
        SparseCluster selected = null;
        for (final SparseCentroidCluster cluster : clusters.values()) {
            if (maxVariance < cluster.getClusterVariance()) {
                maxVariance = cluster.getClusterVariance();
                selected = cluster;
            }
        }

        // did we find at least one non-empty cluster ?
        if (selected == null) {
            throw new ConvergenceException(LocalizedFormats.EMPTY_CLUSTER_IN_K_MEANS);
        }

        // extract a random point from the cluster
        final List<SparseWeightableVector> selectedPoints = selected.getPoints();
        return selectedPoints.remove(random.nextInt(selectedPoints.size()));
    }

    private int assignPointsToClusters(Map<Integer, SparseCentroidCluster> clusters,
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

    private int getNearestCluster(Map<Integer, SparseCentroidCluster> clusters, SparseWeightableVector point) {
        double minDistance = Double.MAX_VALUE;
        int minCluster = 0;
        for (final Map.Entry<Integer, SparseCentroidCluster> each : clusters.entrySet()) {
            final SparseClusterable center = each.getValue().getCenter();
            double distance = point.getDistance(center.getVector()) * FastMath.sqrt(point.getWeight());
            if (distance < minDistance) {
                minDistance = distance;
                minCluster = each.getKey();
            }
        }
        return minCluster;
    }

    private Map<Integer, SparseCentroidCluster> chooseInitialCenters(List<SparseWeightableVector> points) {
        // Convert to list for indexed access. Make it unmodifiable, since removal of items
        // would screw up the logic of this method.
        final List<SparseWeightableVector> pointList = Collections.unmodifiableList(points);

        // The number of points in the list.
        final int numPoints = pointList.size();

        // Set the corresponding element in this array to indicate when
        // elements of pointList are no longer available.
        final boolean[] taken = new boolean[numPoints];

        // The resulting list of initial centers.
        final Map<Integer, SparseCentroidCluster> resultSet = new HashMap<>();

        // Choose one center uniformly at random from among the data points.
        final int firstPointIndex = random.nextInt(numPoints);

        final SparseWeightableVector firstPoint = pointList.get(firstPointIndex);

        resultSet.put(0, new SparseCentroidCluster(firstPoint));

        // Must mark it as taken
        taken[firstPointIndex] = true;

        // To keep track of the minimum distance squared of elements of
        // pointList to elements of resultSet.
        final double[] minDistSquared = new double[numPoints];

        // Initialize the elements.  Since the only point in resultSet is firstPoint,
        // this is very easy.
        for (int i = 0; i < numPoints; i++) {
            if (i != firstPointIndex) { // That point isn't considered
                double d = firstPoint.getDistance(pointList.get(i));
                minDistSquared[i] = firstPoint.getWeight() * d * d;
            }
        }

        int index = 1;
        while (resultSet.size() < k) {

            // Sum up the squared distances for the points in pointList not
            // already taken.
            double distSqSum = 0.0;

            for (int i = 0; i < numPoints; i++) {
                if (!taken[i]) {
                    distSqSum += minDistSquared[i];
                }
            }

            // Add one new data point as a center. Each point x is chosen with
            // probability proportional to D(x)2
            final double r = random.nextDouble() * distSqSum;

            // The index of the next point to be added to the resultSet.
            int nextPointIndex = -1;

            // Sum through the squared min distances again, stopping when
            // sum >= r.
            double sum = 0.0;
            for (int i = 0; i < numPoints; i++) {
                if (!taken[i]) {
                    sum += minDistSquared[i];
                    if (sum >= r) {
                        nextPointIndex = i;
                        break;
                    }
                }
            }

            // If it's not set to >= 0, the point wasn't found in the previous
            // for loop, probably because distances are extremely small.  Just pick
            // the last available point.
            if (nextPointIndex == -1) {
                for (int i = numPoints - 1; i >= 0; i--) {
                    if (!taken[i]) {
                        nextPointIndex = i;
                        break;
                    }
                }
            }

            // We found one.
            if (nextPointIndex >= 0) {

                final SparseWeightableVector p = pointList.get(nextPointIndex);

                resultSet.put(index++, new SparseCentroidCluster((p)));

                // Mark it as taken.
                taken[nextPointIndex] = true;

                if (resultSet.size() < k) {
                    // Now update elements of minDistSquared.  We only have to compute
                    // the distance to the new center to do this.
                    for (int j = 0; j < numPoints; j++) {
                        // Only have to worry about the points still not taken.
                        if (!taken[j]) {
                            double d = p.getDistance(pointList.get(j));
                            d *= p.getWeight() * d;
                            minDistSquared[j] = FastMath.min(d, minDistSquared[j]);
                        }
                    }
                }

            } else {
                // None found --
                // Break from the while loop to prevent
                // an infinite loop.
                break;
            }
        }

        return resultSet;
    }
}
