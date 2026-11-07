package io.github.c0rwin.coreset.flink;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.algo.Algorithms;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import io.github.c0rwin.coreset.core.stream.MergeReduceTree;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ChunkedTreeTest {

    private static final Mixture MIX = new Mixture(3, 5, 10.0, 3, true);

    private static ChunkedTree tree(long seed) {
        return new ChunkedTree(Algorithms.create("sensitivity", Map.of("k", "3", "m", "16")), seed, 2, 32);
    }

    /** Sparse points of varying support, so varint widths and dims of the sizing formula are exercised. */
    private static WeightedPoint sparse(int i) {
        int nnz = 1 + i % 4;
        int[] idx = new int[nnz];
        double[] val = new double[nnz];
        for (int k = 0; k < nnz; k++) {
            idx[k] = (k * 997 + i * 31) % 300 + k * 300;
            val[k] = 1.0 + (i * 7 + k) % 11;
        }
        return new WeightedPoint(SparseVector.of(1500, idx, val), 0.5 + i % 3);
    }

    @Test
    void cachedSizesMatchTheSerializedState() {
        ChunkedTree t = tree(1);
        long peak = 0;
        for (int i = 0; i < 32 * 7 + 5; i++) {
            if (t.add(sparse(i))) {
                long beforeFlush = t.stateBytes();
                assertEquals(t.state().encodedLength(), beforeFlush);
                peak = Math.max(peak, beforeFlush);
                t.flushChunk();
                assertEquals(peak, t.peakStateBytes());
                assertEquals(t.state().encodedLength(), t.stateBytes());
            }
        }
        assertEquals(t.state().encodedLength(), t.stateBytes());
        assertEquals(3, t.levels()); // 7 chunks
        assertEquals(2, t.depth());
    }

    @Test
    void epochsReproduceAPlainTreeWithOffsetSequences() {
        ChunkedTree t = tree(5);
        List<WeightedPoint> second = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            t.add(MIX.point(i));
            if (i % 32 == 31) {
                t.flushChunk();
            }
        }
        ChunkedTree.Partial first = t.takePartial();
        assertEquals(64, first.records());
        assertEquals(1, first.depth());
        for (int i = 64; i < 64 + 96; i++) {
            second.add(MIX.point(i));
            if (t.add(MIX.point(i))) {
                t.flushChunk();
            }
        }
        ChunkedTree.Partial p = t.takePartial();
        // Epoch two starts at chunk 2: its merges use sequences 2, 3, 4 and its final reduce sequence 5.
        MergeReduceTree reference = new MergeReduceTree(Algorithms.create("sensitivity", Map.of("k", "3", "m", "16")),
                (level, seq) -> MergeRandomness.derived(5, 2).forMerge(level, seq + 2), MetricsSink.NOOP);
        for (int c = 0; c < 3; c++) {
            reference.add(Coreset.of(second.subList(32 * c, 32 * (c + 1))));
        }
        assertEquals(reference.compressedSnapshot(), p.coreset());
        assertEquals(2, p.depth()); // levels 0 and 1, reduced once more
        assertEquals(5, t.chunks());
        assertNull(t.takePartial());
        assertTrue(t.isEmpty());
    }
}
