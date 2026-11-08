package io.github.c0rwin.coreset.core.kmeans;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class MeanCellsTest {

    private static PointSet points(int n, SplittableRandom rnd) {
        List<WeightedPoint> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(new WeightedPoint(SparseVector.dense(new double[] {rnd.nextGaussian() * 3, rnd.nextGaussian()}),
                    0.5 + rnd.nextDouble()));
        }
        return PointSet.of(PointBlock.of(out));
    }

    @Test
    void perCellOffsetsSumToTheMeanCoresetOffsetBitForBit() {
        PointSet p = points(300, new SplittableRandom(1));
        int[] assignment = new int[p.size()];
        for (int i = 0; i < assignment.length; i++) {
            assignment[i] = i % 7;
        }
        MeanCells cells = WeightedLloyd.meanCells(p, assignment, 7);
        Coreset means = WeightedLloyd.meansOf(p, assignment, 7);
        double sum = 0.0;
        for (double o : cells.offsets()) {
            sum += o;
        }
        assertEquals(means.offset(), sum, 0.0);
        assertEquals(means.pointSet().compact(), cells.means());
    }

    /** prop:cf-triples (a): sum_{p in cell} w ||p - x||^2 = n_g ||mu_g - x||^2 + offset_g for every x. */
    @Test
    void everyCellIsExactForASingleCenter() {
        SplittableRandom rnd = new SplittableRandom(2);
        PointSet p = points(200, rnd);
        int[] assignment = new int[p.size()];
        for (int i = 0; i < assignment.length; i++) {
            assignment[i] = rnd.nextInt(5);
        }
        MeanCells cells = WeightedLloyd.meanCells(p, assignment, 5);
        SparseVector x = SparseVector.dense(new double[] {4.0, -2.0});
        for (int g = 0; g < cells.means().size(); g++) {
            double direct = 0.0;
            for (int i = 0; i < p.size(); i++) {
                if (assignment[i] == g) {
                    direct += p.weight(i) * p.vector(i).squaredDistance(x);
                }
            }
            double viaCell = cells.means().weight(g) * cells.means().vector(g).squaredDistance(x) + cells.offsets()[g];
            assertEquals(direct, viaCell, 1e-9 * direct);
        }
    }

    @Test
    void emptyCellsAreSkippedAndOffsetsMatchRows() {
        PointSet p = points(10, new SplittableRandom(3));
        int[] assignment = new int[p.size()]; // all in part 3 of 6
        java.util.Arrays.fill(assignment, 3);
        MeanCells cells = WeightedLloyd.meanCells(p, assignment, 6);
        assertEquals(1, cells.means().size());
        assertEquals(1, cells.offsets().length);
        assertThrows(IllegalArgumentException.class, () -> new MeanCells(cells.means(), new double[2]));
    }
}
