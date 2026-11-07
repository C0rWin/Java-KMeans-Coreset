package univ.ml.distributed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

import univ.ml.distributed.coreset.CoresetPoints;
import univ.ml.distributed.coreset.CoresetUtils;
import univ.ml.distributed.coreset.CoresetWeightedPoint;
import univ.ml.sparse.SparseWeightableVector;

class CoresetUtilsSparseConversionTest {

    @Test
    void conversionCostDependsOnNonZerosNotOnDimension() {
        final int dim = Integer.MAX_VALUE;
        final List<SparseWeightableVector> points = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            points.add(new SparseWeightableVector(Map.of(dim - 1 - i, i + 0.5), i + 1d, dim));
        }

        // A loop over all dimensions needs about 2^31 lookups per point (seconds each);
        // walking the stored entries takes microseconds.
        final CoresetPoints converted = assertTimeoutPreemptively(Duration.ofSeconds(1),
                () -> CoresetUtils.toCoresetPoints(points));

        assertEquals(8, converted.getPoints().size());
        for (int i = 0; i < 8; i++) {
            final CoresetWeightedPoint point = converted.getPoints().get(i);
            assertEquals(Map.of(dim - 1 - i, i + 0.5), point.getPoint().getCoords());
            assertEquals(dim, point.getPoint().getDim());
            assertEquals(i + 1d, point.getWeight());
        }
    }

    @Test
    void convertedCoordsAreExactlyTheNonZeroEntries() {
        final Map<Integer, Double> coords = Map.of(0, 1.5, 3, -2d, 9, 1e-3);
        final SparseWeightableVector vector = new SparseWeightableVector(coords, 2d, 10);

        final CoresetWeightedPoint point = CoresetUtils.toCoresetPoints(List.of(vector)).getPoints().get(0);

        assertEquals(coords, point.getPoint().getCoords());
        assertEquals(10, point.getPoint().getDim());
        assertEquals(2d, point.getWeight());
    }
}
