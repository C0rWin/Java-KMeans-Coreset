package univ.ml.distributed;

import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.util.FastMath;
import org.junit.Assert;
import org.junit.Test;

import com.google.common.collect.Lists;
import com.google.common.collect.Maps;

import univ.ml.distributed.coreset.CoresetUtils;
import univ.ml.sparse.SparseWeightableVector;

public class CoresetUtilsTest {

    @Test
    public void testEnergyComputation() {

        final Random rnd = new Random(System.currentTimeMillis());
        final List<SparseWeightableVector> centers = Lists.newArrayList();

        // Generate random centers
        for (int i = 0; i < 10; i++) {
            final Map<Integer, Double> coords = Maps.newHashMap();
            for (int k = 0; k < 3; k++) {
                coords.put(k, rnd.nextDouble());
            }
            centers.add(new SparseWeightableVector(coords, 1d, 3));
        }

        final List<SparseWeightableVector> points = Lists.newArrayList();

        // Create random dataset
        for (int i = 0; i < 1_000; i++) {
            final Map<Integer, Double> coords = Maps.newHashMap();
            for (int k = 0; k < 3; k++) {
                coords.put(k, rnd.nextDouble());
            }
            points.add(new SparseWeightableVector(coords, 1d, 3));
        }

        // Compute energy
        final double totalEnergyV1 = CoresetUtils.getEnergy(centers, points);

        final Map<Integer, List<SparseWeightableVector>> clusters = Maps.newHashMap();
        for (int i = 0; i < centers.size(); i++) {
            clusters.put(i, Lists.newArrayList());
        }

        for (final SparseWeightableVector point : points) {
            double minDist = Double.MAX_VALUE;
            int pointClusterIndx = -1;
            for (int i = 0; i < centers.size(); i++) {
                final SparseWeightableVector center = centers.get(i);
                double dist = FastMath.sqrt(center.getWeight()) * center.getDistance(point.getVector());
                if (dist < minDist) {
                    minDist = dist;
                    pointClusterIndx = i;
                }
            }
            clusters.get(pointClusterIndx).add(point);
        }

        double totalEnergyV2 = 0d;

        for (Map.Entry<Integer, List<SparseWeightableVector>> each : clusters.entrySet()) {
            final List<SparseWeightableVector> clusterPoints = each.getValue();
            for (SparseWeightableVector point : clusterPoints) {
                totalEnergyV2 += point.getWeight() * FastMath.pow(point.getDistance(centers.get(each.getKey()).getVector()), 2);
            }
        }

        Assert.assertEquals(totalEnergyV1, totalEnergyV2, 10e-10);
    }
}
