package univ.ml.distributed;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.time.Duration;
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
import univ.ml.distributed.coreset.CoresetWeightedPoint;
import univ.ml.distributed.coreset.StreamingCoresetHandler;

class StreamingCoresetHandlerTest {

    // Generous bound: hitting it means a hang, not a slow machine.
    private static final Duration HANG = Duration.ofSeconds(10);

    private static final int DIM = 3;

    @Test
    void everyBatchEndsUpInTheTotalCoreset() throws TException {
        final StreamingCoresetHandler handler = new StreamingCoresetHandler();
        // Sample size above the input size, so the uniform coreset keeps every point.
        handler.initialize(2, 100, CoresetAlgorithm.UNIFORM);
        handler.compressPoints(batch(1, 0, 4));
        handler.compressPoints(batch(2, 4, 4));
        handler.compressPoints(batch(3, 8, 4));
        awaitDone(handler);

        final Set<Map<Integer, Double>> expected = new HashSet<>();
        batch(0, 0, 12).getPoints().forEach(p -> expected.add(p.getPoint().getCoords()));
        final Set<Map<Integer, Double>> actual = handler.getTotalCoreset().getPoints().stream()
                .map(p -> p.getPoint().getCoords())
                .collect(Collectors.toSet());
        assertEquals(expected, actual);
    }

    @Test
    void getTotalCoresetReleasesItsLock() throws TException {
        final StreamingCoresetHandler handler = new StreamingCoresetHandler();
        handler.initialize(2, 100, CoresetAlgorithm.UNIFORM);
        handler.getTotalCoreset();
        // assertTimeoutPreemptively runs the second call on another thread.
        assertTimeoutPreemptively(HANG, handler::getTotalCoreset);
    }

    @Test
    void compressPointsBeforeInitializeIsRejected() throws TException {
        final StreamingCoresetHandler handler = new StreamingCoresetHandler();
        assertThrows(TApplicationException.class, () -> handler.compressPoints(batch(1, 0, 4)));
        assertTrue(handler.isDone());
    }

    @Test
    void failedBatchIsReportedInsteadOfHangingIsDone() throws TException {
        final StreamingCoresetHandler handler = new StreamingCoresetHandler();
        handler.initialize(2, 100, CoresetAlgorithm.UNIFORM);
        handler.compressPoints(malformedBatch());
        awaitDone(handler);
        assertThrows(TApplicationException.class, handler::getTotalCoreset);
    }

    @Test
    void workerThreadsDoNotKeepTheJvmAlive() throws TException {
        final Set<Thread> before = Thread.getAllStackTraces().keySet();
        final StreamingCoresetHandler handler = new StreamingCoresetHandler();
        handler.initialize(2, 100, CoresetAlgorithm.UNIFORM);
        handler.compressPoints(batch(1, 0, 4));
        awaitDone(handler);

        final List<String> nonDaemon = Thread.getAllStackTraces().keySet().stream()
                .filter(t -> !before.contains(t) && !t.isDaemon())
                .map(Thread::getName)
                .sorted()
                .collect(Collectors.toList());
        assertEquals(List.of(), nonDaemon);
    }

    @Test
    void closeStopsTheWorkerThreads() throws Exception {
        final Set<Thread> before = Thread.getAllStackTraces().keySet();
        final StreamingCoresetHandler handler = new StreamingCoresetHandler();
        handler.initialize(2, 100, CoresetAlgorithm.UNIFORM);
        handler.compressPoints(batch(1, 0, 4));
        awaitDone(handler);

        final List<Thread> workers = Thread.getAllStackTraces().keySet().stream()
                .filter(t -> !before.contains(t) && t.getName().startsWith("coreset-"))
                .collect(Collectors.toList());
        assertFalse(workers.isEmpty());
        handler.close();
        for (Thread worker : workers) {
            worker.join(HANG.toMillis());
            assertFalse(worker.isAlive(), worker.getName() + " still running after close()");
        }
    }

    // isDone() never blocks, so a deadline on the test thread is enough to detect a hang.
    private static void awaitDone(final StreamingCoresetHandler handler) throws TException {
        final long deadline = System.nanoTime() + HANG.toNanos();
        while (!handler.isDone()) {
            if (System.nanoTime() > deadline) {
                fail("isDone() still false after " + HANG);
            }
            Thread.onSpinWait();
        }
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

    // Coordinate index outside the declared dimension: building the vector throws on the worker.
    private static CoresetPoints malformedBatch() {
        final Map<Integer, Double> coords = new HashMap<>();
        coords.put(DIM + 2, 1d);
        return new CoresetPoints(1, List.of(new CoresetWeightedPoint(new CoresetPoint(coords, DIM), 1d)));
    }
}
