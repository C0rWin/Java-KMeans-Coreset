package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class CoresetTest {

    private static WeightedPoint wp(double w, double... coords) {
        return new WeightedPoint(SparseVector.dense(coords), w);
    }

    @Test
    void weightedPointValidatesWeightAndVector() {
        SparseVector v = SparseVector.dense(new double[] {1});
        assertThrows(IllegalArgumentException.class, () -> new WeightedPoint(v, -1.0));
        assertThrows(IllegalArgumentException.class, () -> new WeightedPoint(v, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> new WeightedPoint(v, Double.POSITIVE_INFINITY));
        assertThrows(NullPointerException.class, () -> new WeightedPoint(null, 1.0));
        assertEquals(0.0, new WeightedPoint(v, 0.0).weight());
    }

    @Test
    void coresetValidatesOffsetAndDimensions() {
        List<WeightedPoint> pts = List.of(wp(1, 1, 2));
        assertThrows(IllegalArgumentException.class, () -> new Coreset(pts, -0.5));
        assertThrows(IllegalArgumentException.class, () -> new Coreset(pts, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Coreset.of(List.of(wp(1, 1, 2), wp(1, 1, 2, 3))));
    }

    @Test
    void coresetCopiesCallerList() {
        List<WeightedPoint> pts = new ArrayList<>(List.of(wp(1, 1), wp(2, 2)));
        Coreset c = Coreset.of(pts);
        pts.add(wp(3, 3));
        assertEquals(2, c.size());
        assertThrows(UnsupportedOperationException.class, () -> c.points().add(wp(1, 1)));
    }

    @Test
    void unionConcatenatesPointsAndAddsOffsets() {
        Coreset a = new Coreset(List.of(wp(1.5, 1, 0), wp(2.0, 0, 1)), 3.0);
        Coreset b = new Coreset(List.of(wp(4.0, 2, 2)), 0.25);

        Coreset u = Coreset.union(a, b);

        assertEquals(3, u.size());
        assertEquals(7.5, u.totalWeight());
        assertEquals(3.25, u.offset());
        assertEquals(List.of(wp(1.5, 1, 0), wp(2.0, 0, 1), wp(4.0, 2, 2)), u.points());
        assertEquals(2, a.size());
        assertEquals(1, b.size());
    }

    @Test
    void emptyCoreset() {
        Coreset e = Coreset.of(List.of());
        assertTrue(e.isEmpty());
        assertEquals(0.0, e.totalWeight());
        Coreset u = Coreset.union(e, new Coreset(List.of(), 1.0));
        assertTrue(u.isEmpty());
        assertEquals(1.0, u.offset());
    }

    @Test
    void unionOfManyConcatenatesInOrderAndChecksDimensions() {
        Coreset a = new Coreset(List.of(wp(1, 1, 0)), 1.0);
        Coreset b = Coreset.of(List.of());
        Coreset c = new Coreset(List.of(wp(2, 0, 1), wp(3, 1, 1)), 0.5);
        Coreset u = Coreset.union(List.of(a, b, c));
        assertEquals(List.of(wp(1, 1, 0), wp(2, 0, 1), wp(3, 1, 1)), u.points());
        assertEquals(1.5, u.offset());
        assertEquals(Coreset.union(Coreset.union(a, b), c), u);
        Coreset d3 = Coreset.of(List.of(wp(1, 1, 2, 3)));
        assertThrows(IllegalArgumentException.class, () -> Coreset.union(List.of(a, b, d3)));
    }

    @Test
    void builderChecksDimensionsAndBuildsOnce() {
        Coreset.Builder b = Coreset.builder(1).add(wp(1, 1, 2)).add(SparseVector.dense(new double[] {3, 4}), 2.0);
        assertThrows(IllegalArgumentException.class, () -> b.add(wp(1, 1, 2, 3)));
        Coreset c = b.build(0.5);
        assertEquals(new Coreset(List.of(wp(1, 1, 2), wp(2, 3, 4)), 0.5), c);
        assertThrows(IllegalStateException.class, () -> b.add(wp(1, 1, 2)));
        assertThrows(UnsupportedOperationException.class, () -> c.points().set(0, wp(1, 1, 2)));
    }

    /** Guard: a union allocates its point array once (the old path copied it twice). */
    @Test
    void unionAllocatesThePointArrayOnce() {
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 20_000; i++) {
            pts.add(wp(1, i, 1));
        }
        Coreset a = Coreset.of(pts.subList(0, 10_000));
        Coreset b = Coreset.of(pts.subList(10_000, 20_000));
        Coreset expected = Coreset.union(a, b);
        long union = Allocations.bytes(() -> Coreset.union(a, b));
        // Exactly the result's five block arrays (and a few small objects): one concatenation, no staging copy.
        assertTrue(union <= expected.estimatedBytes() + 256,
                "union allocated " + union + " bytes, its block is " + expected.block().estimatedBytes());
    }

    @Test
    void builtCoresetsSerialiseAndRevalidate() throws Exception {
        Coreset c = Coreset.union(new Coreset(List.of(wp(1, 1, 2)), 0.5), Coreset.of(List.of(wp(2, 3, 4))));
        Coreset back = (Coreset) deserialize(serialize(c));
        assertEquals(c, back);
    }

    @Test
    void estimatedBytesGrowsWithContent() {
        Coreset small = Coreset.of(List.of(wp(1, 1, 2)));
        Coreset large = Coreset.of(List.of(wp(1, 1, 2), wp(1, 3, 4), wp(1, 5, 6)));
        WeightedPoint p = wp(1, 1, 2);
        // header 12 + ref 4 + double 8 = 24
        assertEquals(24 + p.vector().estimatedBytes(), p.estimatedBytes());
        // Points live in one CSR block: record 24 + block object 48 + its five arrays at 8-byte alignment.
        assertEquals(24 + small.block().estimatedBytes(), small.estimatedBytes());
        assertEquals(24 + 48 + align(16 + 4 * 4) + align(16 + 4 * 6) + align(16 + 8 * 6) + 2 * align(16 + 8 * 3),
                large.estimatedBytes());
        assertTrue(large.estimatedBytes() > small.estimatedBytes());
    }

    private static long align(long bytes) {
        return (bytes + 7) & ~7L;
    }

    private static byte[] serialize(Object o) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(o);
        }
        return bytes.toByteArray();
    }

    private static Object deserialize(byte[] b) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(b))) {
            return in.readObject();
        }
    }

    @Test
    void javaSerializationRoundTripsCoresets() throws Exception {
        Coreset c = new Coreset(List.of(wp(1.5, 0, 2, 0, -3), wp(0.25, 1, 0, 0, 0),
                new WeightedPoint(SparseVector.of(4, new int[0], new double[0]), 2)), 0.75);
        Coreset back = (Coreset) deserialize(serialize(c));
        assertEquals(c, back);
        assertEquals(c.points().get(0).vector().hashCode(), back.points().get(0).vector().hashCode());
    }

    @Test
    void deserializationRevalidatesSparseVectors() throws Exception {
        double marker = 1.2345678901234567;
        byte[] b = serialize(SparseVector.of(3, new int[] {1}, new double[] {marker}));
        byte[] pattern = ByteBuffer.allocate(8).putDouble(marker).array();
        int at = -1;
        for (int i = 0; i + 8 <= b.length && at < 0; i++) {
            if (ByteBuffer.wrap(b, i, 8).equals(ByteBuffer.wrap(pattern))) {
                at = i;
            }
        }
        assertTrue(at >= 0);
        ByteBuffer.wrap(b, at, 8).putDouble(Double.NaN);
        assertThrows(InvalidObjectException.class, () -> deserialize(b));
    }
}
