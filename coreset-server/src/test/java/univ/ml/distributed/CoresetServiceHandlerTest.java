package univ.ml.distributed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.apache.thrift.TApplicationException;
import org.apache.thrift.TException;
import org.junit.jupiter.api.Test;

import univ.ml.distributed.coreset.CoresetAlgorithm;
import univ.ml.distributed.coreset.CoresetPoint;
import univ.ml.distributed.coreset.CoresetPoints;
import univ.ml.distributed.coreset.CoresetServiceHandler;
import univ.ml.distributed.coreset.CoresetWeightedPoint;

class CoresetServiceHandlerTest {

    private static final int DIM = 3;

    @Test
    void batchesEndUpInTheTotalCoreset() throws TException {
        final CoresetServiceHandler handler = new CoresetServiceHandler();
        // Sample size above the input size, so the uniform coreset keeps every point.
        assertTrue(handler.initialize(2, 100, CoresetAlgorithm.UNIFORM));
        handler.compressPoints(batch(1, 0, 4));
        handler.compressPoints(batch(2, 4, 4));
        assertTrue(handler.isDone());

        final Set<Map<Integer, Double>> expected = new HashSet<>();
        batch(0, 0, 8).getPoints().forEach(p -> expected.add(p.getPoint().getCoords()));
        final Set<Map<Integer, Double>> actual = handler.getTotalCoreset().getPoints().stream()
                .map(p -> p.getPoint().getCoords())
                .collect(Collectors.toSet());
        assertEquals(expected, actual);
    }

    @Test
    void compressPointsBeforeInitializeFailsWithAClearError() throws TException {
        final CoresetServiceHandler handler = new CoresetServiceHandler();
        assertThrows(TApplicationException.class, () -> handler.compressPoints(batch(1, 0, 4)));
        assertTrue(handler.isDone());
    }

    @Test
    void getTotalCoresetBeforeInitializeFailsWithAClearError() {
        final CoresetServiceHandler handler = new CoresetServiceHandler();
        assertThrows(TApplicationException.class, handler::getTotalCoreset);
    }

    @Test
    void failedBatchDoesNotLeaveIsDoneFalse() throws TException {
        final CoresetServiceHandler handler = new CoresetServiceHandler();
        handler.initialize(2, 100, CoresetAlgorithm.UNIFORM);
        assertThrows(RuntimeException.class, () -> handler.compressPoints(malformedBatch()));
        assertTrue(handler.isDone());
        // The batch is lost, so returning a coreset now would silently under-count the input.
        assertThrows(TApplicationException.class, handler::getTotalCoreset);
    }

    private static CoresetPoints batch(final long id, final int first, final int count) {
        final List<CoresetWeightedPoint> points = new ArrayList<>();
        for (int i = first; i < first + count; i++) {
            final Map<Integer, Double> coords = new HashMap<>();
            coords.put(0, i + 1d);
            coords.put(1, 1d);
            points.add(new CoresetWeightedPoint(new CoresetPoint(coords, DIM), 1d));
        }
        return new CoresetPoints(id, points);
    }

    // Coordinate index outside the declared dimension: building the vector throws.
    private static CoresetPoints malformedBatch() {
        final Map<Integer, Double> coords = new HashMap<>();
        coords.put(DIM + 2, 1d);
        return new CoresetPoints(1, List.of(new CoresetWeightedPoint(new CoresetPoint(coords, DIM), 1d)));
    }
}
