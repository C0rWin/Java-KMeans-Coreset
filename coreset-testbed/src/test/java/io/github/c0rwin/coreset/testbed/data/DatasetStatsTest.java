package io.github.c0rwin.coreset.testbed.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class DatasetStatsTest {

    @Test
    void scaleIncludesImplicitZerosAndSeedingFindsBothClusters() {
        double[][] rows = {{0, 0, 0}, {0, 1, 0}, {1, 0, 0},
                {100, 100, 0}, {100, 101, 0}, {101, 100, 0}, {101, 101, 0}, {100.5, 100.5, 0}};
        List<WeightedPoint> pts = new ArrayList<>();
        for (double[] r : rows) {
            pts.add(new WeightedPoint(SparseVector.dense(r), 1.0));
        }
        PointSet ps = PointSet.of(PointBlock.of(pts));
        DatasetStats s = DatasetStats.of(ps, new int[] {2}, 1L);

        double[] var = new double[3];
        for (int c = 0; c < 3; c++) {
            double mean = 0.0;
            for (double[] r : rows) {
                mean += r[c] / rows.length;
            }
            for (double[] r : rows) {
                var[c] += (r[c] - mean) * (r[c] - mean) / rows.length;
            }
        }
        assertEquals(8, s.n());
        assertEquals(3, s.d());
        assertEquals(8.0, s.weightSum());
        assertEquals(var[0] + var[1], s.totalVariance(), 1e-9);
        assertEquals(0.0, s.minStd());
        assertEquals(Math.sqrt(var[0]), s.medianStd(), 1e-9);
        assertEquals(Math.sqrt(var[1]), s.maxStd(), 1e-9);

        DatasetStats.Seeding k2 = s.seeding().get(0);
        assertEquals(2, k2.k());
        assertEquals(3.0, k2.minClusterWeight());
        assertEquals(5.0, k2.maxClusterWeight());
        assertEquals(5.0 / 3.0, k2.balance());
        assertTrue(k2.cost() > 0 && k2.cost() < 10, "cost " + k2.cost()); // within-cluster only
    }
}
