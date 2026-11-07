package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class PointSetTest {

    private static List<WeightedPoint> points(int n, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        List<WeightedPoint> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            int nnz = rnd.nextInt(4);
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int k = 0; k < nnz; k++) {
                idx[k] = k * 7 + rnd.nextInt(7);
                val[k] = rnd.nextGaussian();
            }
            out.add(new WeightedPoint(SparseVector.of(40, idx, val), rnd.nextDouble()));
        }
        return out;
    }

    @Test
    void blockSlicesShareArraysAndEqualCopies() {
        List<WeightedPoint> pts = points(50, 1);
        PointBlock b = PointBlock.of(pts);
        PointBlock s = b.slice(10, 30);
        assertEquals(PointBlock.of(pts.subList(10, 30)), s);
        assertEquals(PointBlock.of(pts.subList(10, 30)).hashCode(), s.hashCode());
        assertEquals(b.estimatedBytes(), s.estimatedBytes()); // the slice retains its parent's arrays
        assertEquals(PointBlock.of(pts.subList(15, 20)), s.slice(5, 10));
        assertSame(b, b.slice(0, 50));
        assertSame(PointBlock.EMPTY, b.slice(7, 7));
        assertThrows(IndexOutOfBoundsException.class, () -> b.slice(20, 10));
        long bytes = Allocations.bytes(() -> b.slice(10, 30));
        assertTrue(bytes <= 64, "slice allocated " + bytes + " bytes");
        // Copies of a slice hold only its rows.
        assertEquals(PointBlock.of(pts.subList(10, 30)), PointBlock.concat(List.of(s, PointBlock.EMPTY)));
        assertEquals(PointBlock.of(pts.subList(10, 40)), PointBlock.concat(List.of(s, b.slice(30, 40))));
        PointBlock.Builder copy = PointBlock.builder(40, 0, 0).addAll(s);
        assertEquals(s, copy.build());
    }

    @Test
    void slicesSpanSegments() {
        List<WeightedPoint> pts = points(60, 2);
        PointSet set = PointSet.concat(List.of(PointSet.of(PointBlock.of(pts.subList(0, 25))),
                PointSet.of(PointBlock.of(pts.subList(25, 40))), PointSet.of(PointBlock.of(pts.subList(40, 60)))));
        for (int[] r : new int[][] {{0, 60}, {3, 22}, {20, 45}, {24, 25}, {25, 40}, {39, 60}, {5, 5}}) {
            PointSet s = set.slice(r[0], r[1]);
            assertEquals(PointSet.of(PointBlock.of(pts.subList(r[0], r[1]))), s, r[0] + ".." + r[1]);
            assertEquals(r[1] - r[0], s.size());
        }
        assertEquals(3, set.slice(20, 45).segmentCount());
        assertSame(set, set.slice(0, 60));
    }

    @Test
    void slicesSerialiseAsTheirRowsOnly() throws Exception {
        List<WeightedPoint> pts = points(40, 3);
        PointSet s = PointSet.of(PointBlock.of(pts)).slice(12, 31);
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(new Coreset(s, 0.5));
        }
        Coreset back;
        try (ObjectInputStream in = new ObjectInputStream(new ByteArrayInputStream(bytes.toByteArray()))) {
            back = (Coreset) in.readObject();
        }
        assertEquals(new Coreset(pts.subList(12, 31), 0.5), back);
        assertTrue(back.estimatedBytes() < new Coreset(pts, 0.5).estimatedBytes()); // the parent's arrays stay behind
    }
}
