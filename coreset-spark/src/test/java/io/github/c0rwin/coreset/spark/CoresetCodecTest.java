package io.github.c0rwin.coreset.spark;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class CoresetCodecTest {

    /** m points of dimension dim with about nnz non-zeros each, values of full double precision. */
    static Coreset sparseCoreset(int m, int dim, int nnz, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        List<WeightedPoint> pts = new ArrayList<>(m);
        for (int i = 0; i < m; i++) {
            int[] idx = rnd.ints(nnz, 0, dim).toArray();
            double[] val = rnd.doubles(nnz, -1e3, 1e3).toArray();
            pts.add(new WeightedPoint(SparseVector.of(dim, idx, val), rnd.nextDouble() * 50));
        }
        return new Coreset(pts, rnd.nextDouble() * 1e6);
    }

    private static long javaSerializedSize(Object o) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(o);
        }
        return bytes.size();
    }

    @Test
    void roundTripIsBitExact() {
        for (long seed = 1; seed <= 5; seed++) {
            Coreset c = sparseCoreset(50, 100_000, (int) (seed * 7), seed);
            Coreset back = CoresetCodec.decode(CoresetCodec.encode(c));
            assertEquals(c, back);
            for (int i = 0; i < c.size(); i++) {
                assertEquals(Double.doubleToRawLongBits(c.points().get(i).weight()),
                        Double.doubleToRawLongBits(back.points().get(i).weight()));
            }
        }
        Coreset empty = new Coreset(List.of(), 3.5);
        assertEquals(empty, CoresetCodec.decode(CoresetCodec.encode(empty)));
        assertEquals(9, CoresetCodec.encode(empty).length);
    }

    @Test
    void standaloneVectorsAndPointsRoundTrip() {
        SparseVector v = SparseVector.of(1 << 30, new int[] {0, 5, (1 << 30) - 1}, new double[] {-0.5, 1e-300, 7});
        assertEquals(v, CoresetCodec.decodeVector(CoresetCodec.encode(v)));
        SparseVector zero = SparseVector.of(4, new int[0], new double[0]);
        assertEquals(zero, CoresetCodec.decodeVector(CoresetCodec.encode(zero)));
        WeightedPoint p = new WeightedPoint(v, 0.125);
        assertEquals(p, CoresetCodec.decodePoint(CoresetCodec.encode(p)));
    }

    @Test
    void malformedInputIsRejected() {
        byte[] good = CoresetCodec.encode(sparseCoreset(3, 10, 4, 9));
        for (int cut = 0; cut < good.length; cut++) {
            byte[] truncated = Arrays.copyOf(good, cut);
            assertThrows(IllegalArgumentException.class, () -> CoresetCodec.decode(truncated));
        }
        byte[] trailing = Arrays.copyOf(good, good.length + 1);
        assertThrows(IllegalArgumentException.class, () -> CoresetCodec.decode(trailing));
        // offset 0, one point claimed, dim 2, weight 1, nnz 2 with a repeated index (delta 0)
        byte[] repeated = new byte[8 + 1 + 1 + 8 + 1 + 2 + 16];
        repeated[8] = 1;
        repeated[9] = 2;
        repeated[17] = 0x3F;
        repeated[16] = (byte) 0xF0; // little-endian 1.0 has 0x3FF0 in its top bytes
        repeated[18] = 2;
        repeated[19] = 1;
        repeated[20] = 0;
        assertThrows(IllegalArgumentException.class, () -> CoresetCodec.decode(repeated));
        // a huge claimed point count must fail before allocating
        byte[] huge = {0, 0, 0, 0, 0, 0, 0, 0, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, (byte) 0xFF, 0x07};
        assertThrows(IllegalArgumentException.class, () -> CoresetCodec.decode(huge));
    }

    @Test
    void encodingIsSmallerThanJavaSerializationAndHeap() throws IOException {
        // research notes §4.1: about m * (11 s + 8) bytes for m points with s non-zeros (varint index ~3 B)
        for (int[] shape : new int[][] {{2000, 784, 100}, {2000, 100_000, 100}, {500, 20, 20}}) {
            Coreset c = sparseCoreset(shape[0], shape[1], shape[2], 11);
            long nnz = c.points().stream().mapToLong(p -> p.vector().nnz()).sum();
            long encoded = CoresetCodec.encode(c).length;
            long java = javaSerializedSize(c);
            long heap = c.estimatedBytes();
            long dense = 8L * shape[0] * (shape[1] + 1);
            System.out.printf("codec size m=%d dim=%d nnz/pt=%.1f: codec %d B (%.2f B/nnz), java %d B, heap %d B,"
                            + " dense %d B, research-formula %d B%n", shape[0], shape[1], (double) nnz / shape[0],
                    encoded, (double) encoded / nnz, java, heap, dense, shape[0] * (11L * nnz / shape[0] + 8));
            assertTrue(encoded < java, "codec " + encoded + " vs java " + java);
            assertTrue(encoded < heap, "codec " + encoded + " vs heap " + heap);
            assertTrue(encoded <= shape[0] * 30L + nnz * 11L, "codec bytes beyond 11 B per non-zero");
        }
    }

    @Test
    void sameCoresetEncodesToSameBytes() {
        Coreset c = sparseCoreset(20, 50, 5, 4);
        assertArrayEquals(CoresetCodec.encode(c), CoresetCodec.encode(CoresetCodec.decode(CoresetCodec.encode(c))));
    }

    @Test
    void decodesFromOffsetsAndIntoListsWithExactSizing() {
        Coreset a = sparseCoreset(30, 1 << 20, 40, 5);
        Coreset b = sparseCoreset(10, 1 << 20, 3, 6);
        byte[] ea = CoresetCodec.encode(a);
        assertEquals(ea.length, CoresetCodec.encodedLength(a));
        byte[] framed = new byte[ea.length + 7];
        System.arraycopy(ea, 0, framed, 3, ea.length);
        assertEquals(a, CoresetCodec.decode(framed, 3, ea.length));
        assertThrows(IllegalArgumentException.class, () -> CoresetCodec.decode(framed, 3, ea.length - 1));
        assertThrows(IndexOutOfBoundsException.class, () -> CoresetCodec.decode(framed, 8, ea.length));

        List<WeightedPoint> union = new ArrayList<>();
        double offset = CoresetCodec.decodeInto(ea, union) + CoresetCodec.decodeInto(CoresetCodec.encode(b), union);
        assertEquals(Coreset.union(a, b), new Coreset(union, offset));
    }
}
