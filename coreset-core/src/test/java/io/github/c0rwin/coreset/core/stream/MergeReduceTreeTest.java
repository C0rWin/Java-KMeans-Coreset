package io.github.c0rwin.coreset.core.stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.Distribution;
import io.github.c0rwin.coreset.core.metrics.MetricNames;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.metrics.MetricsSnapshot;
import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Test;

class MergeReduceTreeTest {

    /** Keeps every point: the tree then behaves as a pure binary counter. */
    private static final CoresetAlgorithm IDENTITY = (input, rng, metrics) -> input;

    /** Keeps a uniform half (at least one point) and rescales weights so total weight is preserved. */
    private static final CoresetAlgorithm HALVING = new CoresetAlgorithm() {
        @Override
        public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
            List<WeightedPoint> pts = new ArrayList<>(input.points());
            int keep = Math.max(1, pts.size() / 2);
            for (int i = 0; i < keep; i++) {
                int j = i + rng.nextInt(pts.size() - i);
                WeightedPoint t = pts.get(i);
                pts.set(i, pts.get(j));
                pts.set(j, t);
            }
            List<WeightedPoint> kept = pts.subList(0, keep);
            double in = input.totalWeight();
            double sub = 0.0;
            for (WeightedPoint p : kept) {
                sub += p.weight();
            }
            double scale = sub > 0 ? in / sub : 0.0;
            List<WeightedPoint> out = new ArrayList<>(keep);
            for (WeightedPoint p : kept) {
                out.add(new WeightedPoint(p.vector(), p.weight() * scale));
            }
            return new Coreset(out, input.offset());
        }
    };

    private static Coreset batch(SplittableRandom rnd, int size) {
        List<WeightedPoint> pts = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            pts.add(new WeightedPoint(SparseVector.dense(new double[] {rnd.nextDouble(), rnd.nextDouble()}),
                    0.5 + rnd.nextDouble()));
        }
        return new Coreset(pts, 0.125);
    }

    @Test
    void occupiedLevelsFollowBinaryCounter() {
        RecordingMetrics m = new RecordingMetrics();
        MergeReduceTree tree = new MergeReduceTree(IDENTITY, new SplittableRandom(1), m);
        SplittableRandom rnd = new SplittableRandom(2);
        int batchSize = 3;
        for (int b = 1; b <= 37; b++) {
            tree.add(batch(rnd, batchSize));
            assertEquals(b, tree.batchesAdded());
            assertEquals(Long.bitCount(b), tree.levels(), "after " + b + " batches");
            assertEquals(b * batchSize, tree.snapshot().size());
        }
        MetricsSnapshot s = m.snapshot();
        long b = 37;
        assertEquals(b - Long.bitCount(b), s.counters().get(MetricNames.MRTREE_MERGES));
        for (int level = 0; (1L << (level + 1)) <= b; level++) {
            assertEquals(b >> (level + 1), s.counters().get(MetricNames.mrtreeLevelMerges(level)), "level " + level);
        }
        assertNull(s.counters().get(MetricNames.mrtreeLevelMerges(5)));
        Distribution chain = s.distributions().get(MetricNames.MRTREE_CARRY_CHAIN_LENGTH);
        assertEquals(b, chain.count());
        assertEquals(b - Long.bitCount(b), chain.sum());
        assertEquals(5.0, chain.max()); // the 32nd add carries through levels 0..4
        // Level 5 only ever receives the single 32-batch coreset.
        assertEquals(32.0 * batchSize, s.distributions().get(MetricNames.mrtreeLevelPoints(5)).max());
    }

    @Test
    void weightIsConservedByWeightPreservingAlgorithm() {
        RecordingMetrics m = new RecordingMetrics();
        MergeReduceTree tree = new MergeReduceTree(HALVING, new SplittableRandom(3), m);
        SplittableRandom rnd = new SplittableRandom(4);
        double totalWeight = 0.0;
        int batches = 100;
        for (int i = 0; i < batches; i++) {
            Coreset c = batch(rnd, 16);
            totalWeight += c.totalWeight();
            tree.add(c);
        }
        Coreset snapshot = tree.snapshot();
        assertEquals(totalWeight, snapshot.totalWeight(), 1e-9 * totalWeight);
        assertEquals(batches * 0.125, snapshot.offset(), 1e-12);
        Coreset compressed = tree.compressedSnapshot();
        assertEquals(totalWeight, compressed.totalWeight(), 1e-9 * totalWeight);
        assertTrue(compressed.size() < snapshot.size());

        MetricsSnapshot s = m.snapshot();
        Distribution drift = s.distributions().get(MetricNames.MRTREE_MERGE_RELATIVE_WEIGHT_DRIFT);
        assertEquals(batches - Long.bitCount(batches), drift.count());
        assertTrue(Math.abs(drift.min()) < 1e-12 && Math.abs(drift.max()) < 1e-12);
        assertEquals(compressed.size(), s.distributions().get(MetricNames.MRTREE_COMPRESSED_SNAPSHOT_POINTS).max());
        assertEquals(snapshot.estimatedBytes(), s.distributions().get(MetricNames.MRTREE_SNAPSHOT_BYTES).max());
    }

    @Test
    void inputsAreNotMutatedOrAliased() {
        MergeReduceTree tree = new MergeReduceTree(HALVING, new SplittableRandom(5));
        SplittableRandom rnd = new SplittableRandom(6);
        List<Coreset> inputs = new ArrayList<>();
        List<List<WeightedPoint>> before = new ArrayList<>();
        for (int i = 0; i < 9; i++) {
            Coreset c = batch(rnd, 8);
            inputs.add(c);
            before.add(new ArrayList<>(c.points()));
            tree.add(c);
        }
        for (int i = 0; i < inputs.size(); i++) {
            assertEquals(before.get(i), inputs.get(i).points());
        }
        Coreset s1 = tree.snapshot();
        tree.add(batch(rnd, 8));
        assertEquals(9 * 0.125, s1.offset(), 1e-12); // earlier snapshot unaffected by later adds
    }

    @Test
    void emptyTree() {
        MergeReduceTree tree = new MergeReduceTree(IDENTITY, new SplittableRandom(7));
        assertTrue(tree.snapshot().isEmpty());
        assertTrue(tree.compressedSnapshot().isEmpty());
        assertEquals(0, tree.levels());
        assertEquals(0, tree.batchesAdded());
    }

    @Test
    void singleBatchSnapshotKeepsPoints() {
        MergeReduceTree tree = new MergeReduceTree(IDENTITY, new SplittableRandom(8));
        Coreset c = batch(new SplittableRandom(9), 4);
        tree.add(c);
        assertEquals(c, tree.snapshot());
        assertSame(c.points().get(0), tree.snapshot().points().get(0));
    }

    @Test
    void levelsViewSharesLevelCoresetsAndIsImmutable() {
        MergeReduceTree tree = new MergeReduceTree(IDENTITY, new SplittableRandom(10));
        SplittableRandom rnd = new SplittableRandom(11);
        Coreset first = batch(rnd, 3);
        tree.add(first);
        List<Coreset> one = tree.levelsView();
        assertEquals(1, one.size());
        assertSame(first, one.get(0)); // a reference, not a copy of the points

        tree.add(batch(rnd, 3)); // carries into level 1
        List<Coreset> two = tree.levelsView();
        Coreset third = batch(rnd, 3);
        tree.add(third);
        List<Coreset> three = tree.levelsView();
        assertEquals(2, three.size());
        assertSame(third, three.get(0));
        assertSame(two.get(0), three.get(1)); // the untouched level is the same instance, not re-copied
        assertEquals(6, three.get(1).size());

        assertEquals(List.of(first), one); // later adds do not change an earlier view
        assertThrows(UnsupportedOperationException.class, () -> three.add(first));
        assertThrows(UnsupportedOperationException.class, () -> three.remove(0));
        assertEquals(List.of(), new MergeReduceTree(IDENTITY, new SplittableRandom(12)).levelsView());
    }
}
