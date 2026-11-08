package io.github.c0rwin.coreset.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.flink.runtime.checkpoint.OperatorSubtaskState;
import org.apache.flink.streaming.api.operators.OneInputStreamOperator;
import org.apache.flink.streaming.runtime.streamrecord.StreamRecord;
import org.apache.flink.streaming.util.OneInputStreamOperatorTestHarness;
import org.junit.jupiter.api.Test;

/** Single-operator behaviour under Flink's test harness: triggers, snapshot/restore, guards. */
class OperatorHarnessTest {

    private static final Mixture MIX = new Mixture(3, 4, 10.0, 11, true);
    private static final Map<String, String> SENSITIVITY = Map.of("k", "3", "m", "20");

    private static CoresetConfig.Builder config(long seed) {
        return CoresetConfig.builder("sensitivity", SENSITIVITY, seed).chunkSize(40).chunksPerPartial(4);
    }

    private static <I, O> OneInputStreamOperatorTestHarness<I, O> harness(OneInputStreamOperator<I, O> op,
            int parallelism, int subtask, OperatorSubtaskState restore) throws Exception {
        OneInputStreamOperatorTestHarness<I, O> h =
                new OneInputStreamOperatorTestHarness<>(op, parallelism, parallelism, subtask);
        if (restore != null) {
            h.initializeState(restore);
        }
        h.open();
        return h;
    }

    private static OneInputStreamOperatorTestHarness<WeightedPoint, PartialCoreset> local(CoresetConfig c,
            int parallelism, int subtask, OperatorSubtaskState restore) throws Exception {
        return harness(new LocalCoresetOperator(c), parallelism, subtask, restore);
    }

    private static void feed(OneInputStreamOperatorTestHarness<WeightedPoint, ?> h, long from, long to)
            throws Exception {
        for (long i = from; i < to; i++) {
            h.processElement(new StreamRecord<>(MIX.point(i)));
        }
    }

    private static List<PartialCoreset> localRun(CoresetConfig c, int parallelism, int subtask, long from, long to)
            throws Exception {
        try (OneInputStreamOperatorTestHarness<WeightedPoint, PartialCoreset> h = local(c, parallelism, subtask,
                null)) {
            feed(h, from, to);
            h.endInput();
            return new ArrayList<>(h.extractOutputValues());
        }
    }

    @Test
    void countTriggerCutsPartialsEveryChunksPerPartialChunks() throws Exception {
        List<PartialCoreset> out = localRun(config(1).build(), 2, 1, 0, 1000);
        // 25 chunks of 40: six partials of 4 chunks, then the last chunk at end of input.
        assertEquals(7, out.size());
        for (int i = 0; i < 6; i++) {
            PartialCoreset p = out.get(i);
            assertEquals(1, p.subtask());
            assertEquals(2, p.parallelism());
            assertEquals(i, p.sequence());
            assertEquals(160, p.records());
            assertEquals(2, p.depth()); // 4 chunks: one level-2 coreset, no final reduce
            assertTrue(p.coreset().size() <= 20);
            assertEquals(MIX.coreset(160L * i, 160L * (i + 1)).totalWeight(), p.weight(), 1e-9);
        }
        PartialCoreset last = out.get(6);
        assertEquals(40, last.records());
        assertEquals(0, last.depth());
        assertEquals(MIX.coreset(960, 1000), last.coreset()); // a lone raw chunk is passed through
    }

    @Test
    void restoreFromSnapshotReproducesUninterruptedPartials() throws Exception {
        CoresetConfig c = config(1).build();
        List<PartialCoreset> expected = localRun(c, 2, 0, 0, 1000);
        List<PartialCoreset> actual = new ArrayList<>();
        OperatorSubtaskState snapshot;
        try (OneInputStreamOperatorTestHarness<WeightedPoint, PartialCoreset> h = local(c, 2, 0, null)) {
            feed(h, 0, 530); // mid-chunk and mid-epoch: buffer and tree both carry state
            snapshot = h.snapshot(1, 0);
            actual.addAll(h.extractOutputValues());
        }
        try (OneInputStreamOperatorTestHarness<WeightedPoint, PartialCoreset> h = local(c, 2, 0, snapshot)) {
            feed(h, 530, 1000);
            h.endInput();
            actual.addAll(h.extractOutputValues());
        }
        assertEquals(expected, actual);
        assertNotEquals(expected, localRun(config(2).build(), 2, 0, 0, 1000));
    }

    @Test
    void restoreRejectsRescalingAndChangedConfiguration() throws Exception {
        CoresetConfig c = config(1).build();
        OperatorSubtaskState snapshot;
        try (OneInputStreamOperatorTestHarness<WeightedPoint, PartialCoreset> h = local(c, 2, 0, null)) {
            feed(h, 0, 100);
            snapshot = h.snapshot(1, 0);
        }
        assertMessage(assertThrows(Exception.class, () -> local(c, 3, 0, snapshot)), "rescaling");
        assertMessage(assertThrows(Exception.class, () -> local(c, 2, 1, snapshot)), "rescaling");
        assertMessage(assertThrows(Exception.class, () -> local(config(9).build(), 2, 0, snapshot)),
                "different algorithm");
        assertMessage(assertThrows(Exception.class, () -> local(config(1).chunkSize(41).build(), 2, 0, snapshot)),
                "different algorithm");
    }

    private static void assertMessage(Throwable t, String fragment) {
        for (Throwable c = t; c != null; c = c.getCause()) {
            if (c.getMessage() != null && c.getMessage().contains(fragment)) {
                return;
            }
        }
        throw new AssertionError("no cause mentions '" + fragment + "'", t);
    }

    @Test
    void processingTimeTriggerFlushesPartialChunks() throws Exception {
        CoresetConfig c = config(1).partialInterval(Duration.ofMillis(100)).build();
        OneInputStreamOperatorTestHarness<WeightedPoint, PartialCoreset> h =
                new OneInputStreamOperatorTestHarness<>(new LocalCoresetOperator(c), 1, 1, 0);
        h.setProcessingTime(0);
        h.open();
        feed(h, 0, 50);
        h.setProcessingTime(100);
        h.setProcessingTime(200); // nothing new: no empty partial
        List<PartialCoreset> out = h.extractOutputValues();
        assertEquals(1, out.size());
        assertEquals(50, out.get(0).records());
        assertEquals(1, out.get(0).depth()); // full and partial chunk merged into one level-1 coreset
        h.close();
    }

    @Test
    void eventTimeTriggerEmitsWhenTheWatermarkCrossesABoundary() throws Exception {
        CoresetConfig c = config(1).partialEventTimeInterval(Duration.ofSeconds(1)).build();
        try (OneInputStreamOperatorTestHarness<WeightedPoint, PartialCoreset> h = local(c, 1, 0, null)) {
            h.processWatermark(500);
            feed(h, 0, 30);
            h.processWatermark(999);
            assertEquals(0, h.extractOutputValues().size());
            h.processWatermark(1000);
            feed(h, 30, 45);
            h.processWatermark(1500);
            h.processWatermark(2000);
            List<PartialCoreset> out = h.extractOutputValues();
            assertEquals(List.of(30L, 15L), out.stream().map(PartialCoreset::records).toList());
            assertEquals(2000, h.getCurrentWatermark());
        }
    }

    // ---- merge stage

    private static List<PartialCoreset> partialsOf(CoresetConfig c, int subtask) throws Exception {
        return localRun(c, 2, subtask, subtask * 1000L, (subtask + 1) * 1000L);
    }

    private static OneInputStreamOperatorTestHarness<PartialCoreset, CoresetSnapshot> merge(CoresetConfig c,
            OperatorSubtaskState restore) throws Exception {
        return harness(new GlobalCoresetOperator(c), 1, 0, restore);
    }

    private static CoresetSnapshot mergeAll(CoresetConfig c, List<PartialCoreset> order) throws Exception {
        try (OneInputStreamOperatorTestHarness<PartialCoreset, CoresetSnapshot> h = merge(c, null)) {
            for (PartialCoreset p : order) {
                h.processElement(new StreamRecord<>(p));
            }
            h.endInput();
            List<CoresetSnapshot> out = h.extractOutputValues();
            assertEquals(order.size() + 1, out.size());
            assertTrue(out.get(out.size() - 1).last());
            return out.get(out.size() - 1);
        }
    }

    @Test
    void mergeResultIsIndependentOfInterleavingAndSurvivesRestore() throws Exception {
        CoresetConfig c = config(1).compressSnapshots(true).build();
        List<PartialCoreset> a = partialsOf(c, 0);
        List<PartialCoreset> b = partialsOf(c, 1);
        List<PartialCoreset> sequential = new ArrayList<>(a);
        sequential.addAll(b);
        List<PartialCoreset> alternating = new ArrayList<>();
        for (int i = 0; i < a.size(); i++) {
            alternating.add(b.get(i));
            alternating.add(a.get(i));
        }
        CoresetSnapshot expected = mergeAll(c, sequential);
        assertEquals(expected, mergeAll(c, alternating));
        assertEquals(2000, expected.records());
        assertEquals(14, expected.partials());
        assertEquals(MIX.coreset(0, 2000).totalWeight(), expected.weight(), 1e-9);
        assertTrue(expected.compressed().size() <= 20);
        // 7 partials per subtask: levels 0, 1, 2 occupied in each tree.
        assertEquals(6, expected.levels());

        OperatorSubtaskState snapshot;
        try (OneInputStreamOperatorTestHarness<PartialCoreset, CoresetSnapshot> h = merge(c, null)) {
            for (PartialCoreset p : alternating.subList(0, 5)) {
                h.processElement(new StreamRecord<>(p));
            }
            snapshot = h.snapshot(1, 0);
        }
        try (OneInputStreamOperatorTestHarness<PartialCoreset, CoresetSnapshot> h = merge(c, snapshot)) {
            for (PartialCoreset p : alternating.subList(5, alternating.size())) {
                h.processElement(new StreamRecord<>(p));
            }
            h.endInput();
            List<CoresetSnapshot> out = h.extractOutputValues();
            assertEquals(expected, out.get(out.size() - 1));
        }
    }

    @Test
    void mergeRejectsGapsDuplicatesAndRescaledInput() throws Exception {
        CoresetConfig c = config(1).build();
        List<PartialCoreset> a = partialsOf(c, 0);
        try (OneInputStreamOperatorTestHarness<PartialCoreset, CoresetSnapshot> h = merge(c, null)) {
            h.processElement(new StreamRecord<>(a.get(0)));
            assertMessage(assertThrows(IllegalStateException.class, () -> h.processElement(new StreamRecord<>(
                    a.get(0)))), "lost or duplicated");
            assertMessage(assertThrows(IllegalStateException.class, () -> h.processElement(new StreamRecord<>(
                    a.get(2)))), "lost or duplicated");
            PartialCoreset p = a.get(1);
            assertMessage(assertThrows(IllegalStateException.class, () -> h.processElement(new StreamRecord<>(
                    new PartialCoreset(p.subtask(), 3, p.sequence(), p.coreset(), p.records(), p.weight(),
                            p.depth())))), "rescaling");
        }
    }

    @Test
    void snapshotCountTriggerAndEmptyInput() throws Exception {
        CoresetConfig every3 = config(1).snapshotEveryPartials(3).build();
        List<PartialCoreset> a = partialsOf(every3, 0);
        try (OneInputStreamOperatorTestHarness<PartialCoreset, CoresetSnapshot> h = merge(every3, null)) {
            for (PartialCoreset p : a) {
                h.processElement(new StreamRecord<>(p));
            }
            h.endInput();
            List<CoresetSnapshot> out = h.extractOutputValues();
            assertEquals(List.of(3L, 6L, 7L), out.stream().map(CoresetSnapshot::partials).toList());
            assertNull(out.get(0).compressed());
        }
        try (OneInputStreamOperatorTestHarness<PartialCoreset, CoresetSnapshot> h = merge(every3, null)) {
            h.endInput();
            CoresetSnapshot s = h.extractOutputValues().get(0);
            assertTrue(s.last());
            assertEquals(Coreset.of(List.of()), s.coreset());
            assertEquals(0, s.records());
        }
    }

    // ---- windows

    private static OneInputStreamOperatorTestHarness<WeightedPoint, WindowPartial> window(CoresetConfig c,
            OperatorSubtaskState restore) throws Exception {
        return harness(new WindowCoresetOperator(c, 1000), 2, 1, restore);
    }

    @Test
    void windowStageEmitsOnePartialPerWindowAndDropsLateRecords() throws Exception {
        CoresetConfig c = config(1).build();
        try (OneInputStreamOperatorTestHarness<WeightedPoint, WindowPartial> h = window(c, null)) {
            for (int i = 0; i < 300; i++) {
                h.processElement(MIX.point(i), i * 10L); // 100 records per 1 s window
            }
            h.processWatermark(1999);
            h.processElement(MIX.point(300), 1500); // behind the watermark: dropped
            h.processElement(MIX.point(301), 2500);
            h.endInput();
            List<StreamRecord<? extends WindowPartial>> out = h.extractOutputStreamRecords();
            assertEquals(3, out.size());
            assertEquals(List.of(999L, 1999L, 2999L), out.stream().map(StreamRecord::getTimestamp).toList());
            assertEquals(List.of(0L, 1000L, 2000L), out.stream().map(r -> r.getValue().windowStart()).toList());
            assertEquals(List.of(100L, 100L, 101L), out.stream().map(r -> r.getValue().records()).toList());
            assertEquals(1, out.get(0).getValue().subtask());
            assertEquals(2, out.get(0).getValue().depth()); // 3 chunks: levels 0 and 1, reduced once more
        }
    }

    @Test
    void windowStageRestoresOpenWindows() throws Exception {
        CoresetConfig c = config(1).build();
        List<WindowPartial> expected;
        try (OneInputStreamOperatorTestHarness<WeightedPoint, WindowPartial> h = window(c, null)) {
            for (int i = 0; i < 300; i++) {
                h.processElement(MIX.point(i), i * 10L);
            }
            h.endInput();
            expected = new ArrayList<>(h.extractOutputValues());
        }
        List<WindowPartial> actual = new ArrayList<>();
        OperatorSubtaskState snapshot;
        try (OneInputStreamOperatorTestHarness<WeightedPoint, WindowPartial> h = window(c, null)) {
            for (int i = 0; i < 155; i++) {
                h.processElement(MIX.point(i), i * 10L);
            }
            h.processWatermark(999);
            snapshot = h.snapshot(1, 0);
            actual.addAll(h.extractOutputValues());
        }
        try (OneInputStreamOperatorTestHarness<WeightedPoint, WindowPartial> h = window(c, snapshot)) {
            h.processElement(MIX.point(0), 0); // window 0 was emitted before the snapshot: stays closed
            for (int i = 155; i < 300; i++) {
                h.processElement(MIX.point(i), i * 10L);
            }
            h.endInput();
            actual.addAll(h.extractOutputValues());
        }
        assertEquals(expected, actual);
    }

    @Test
    void windowStageRequiresTimestamps() throws Exception {
        try (OneInputStreamOperatorTestHarness<WeightedPoint, WindowPartial> h = window(config(1).build(), null)) {
            assertMessage(assertThrows(IllegalStateException.class,
                    () -> h.processElement(new StreamRecord<>(MIX.point(0)))), "timestamps");
        }
    }
}
