package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class SparseVectorTest {

    @Test
    void ofSortsSumsDuplicatesAndDropsZeros() {
        SparseVector v = SparseVector.of(10, new int[] {7, 2, 7, 5, 2, 9}, new double[] {1.5, 3.0, 2.5, 0.0, -3.0, -1.0});

        assertEquals(10, v.dim());
        assertEquals(2, v.nnz());
        assertEquals(7, v.indexAt(0));
        assertEquals(4.0, v.valueAt(0));
        assertEquals(9, v.indexAt(1));
        assertEquals(-1.0, v.valueAt(1));
        assertEquals(0.0, v.get(2));
        assertEquals(0.0, v.get(5));
        assertEquals(17.0, v.squaredNorm());
    }

    @Test
    void ofCopiesInputArrays() {
        int[] idx = {1, 3};
        double[] val = {2.0, 4.0};
        SparseVector v = SparseVector.of(5, idx, val);
        idx[0] = 4;
        val[0] = 100.0;
        assertEquals(2.0, v.get(1));
        assertEquals(0.0, v.get(4));
    }

    @Test
    void ofOwnedCanonicalisesAndValidatesLikeOf() {
        int[][] indices = {{1, 3}, {3, 1, 3}, {2, 0, 4}, {1, 2}};
        double[][] values = {{2.0, 4.0}, {1.0, 2.0, -1.0}, {0.0, 5.0, 6.0}, {0.0, 0.0}};
        for (int t = 0; t < indices.length; t++) {
            assertEquals(SparseVector.of(5, indices[t], values[t]),
                    SparseVector.ofOwned(5, indices[t].clone(), values[t].clone()));
        }
        assertThrows(IllegalArgumentException.class, () -> SparseVector.ofOwned(3, new int[] {3}, new double[] {1}));
        assertThrows(IllegalArgumentException.class, () -> SparseVector.ofOwned(3, new int[] {0}, new double[] {
            Double.NaN}));
        assertThrows(IllegalArgumentException.class, () -> SparseVector.ofOwned(3, new int[] {0, 1}, new double[] {1}));
    }

    /** Guard: decoders hand over fresh arrays, so a canonical vector must not be copied again. */
    @Test
    void ofOwnedDoesNotCopyCanonicalArrays() {
        int n = 10_000;
        int[] idx = new int[n];
        double[] val = new double[n];
        for (int k = 0; k < n; k++) {
            idx[k] = 2 * k;
            val[k] = k + 1.0;
        }
        long owned = Allocations.bytes(() -> SparseVector.ofOwned(2 * n, idx, val));
        long copied = Allocations.bytes(() -> SparseVector.of(2 * n, idx, val));
        assertTrue(owned < 256, "ofOwned allocated " + owned + " bytes");
        assertTrue(copied >= 12L * n, "of allocated " + copied + " bytes");
    }

    @Test
    void ofValidatesBoundsLengthsAndFiniteness() {
        assertThrows(IllegalArgumentException.class, () -> SparseVector.of(3, new int[] {3}, new double[] {1}));
        assertThrows(IllegalArgumentException.class, () -> SparseVector.of(3, new int[] {-1}, new double[] {1}));
        assertThrows(IllegalArgumentException.class, () -> SparseVector.of(3, new int[] {0, 1}, new double[] {1}));
        assertThrows(IllegalArgumentException.class, () -> SparseVector.of(-1, new int[0], new double[0]));
        assertThrows(IllegalArgumentException.class,
                () -> SparseVector.of(3, new int[] {0}, new double[] {Double.NaN}));
        assertThrows(IllegalArgumentException.class,
                () -> SparseVector.of(3, new int[] {0}, new double[] {Double.POSITIVE_INFINITY}));
        assertThrows(IllegalArgumentException.class,
                () -> SparseVector.of(3, new int[] {1, 1}, new double[] {Double.MAX_VALUE, Double.MAX_VALUE}));
        assertThrows(IllegalArgumentException.class, () -> SparseVector.dense(new double[] {1, Double.NaN}));
        assertThrows(IndexOutOfBoundsException.class, () -> SparseVector.dense(new double[] {1, 2}).get(2));
    }

    @Test
    void equalityIsValueBased() {
        SparseVector a = SparseVector.of(4, new int[] {3, 1}, new double[] {2.0, 1.0});
        SparseVector b = SparseVector.dense(new double[] {0.0, 1.0, 0.0, 2.0});
        SparseVector c = SparseVector.of(4, new int[] {1, 3, 2}, new double[] {1.0, 2.0, -0.0});
        assertEquals(a, b);
        assertEquals(a, c);
        assertEquals(a.hashCode(), b.hashCode());
        assertNotEquals(a, SparseVector.dense(new double[] {0.0, 1.0, 0.0, 2.0, 0.0}));
        assertNotEquals(a, SparseVector.dense(new double[] {0.0, 1.0, 0.0, 2.5}));
    }

    @Test
    void zeroVectorHasNoEntries() {
        SparseVector z = SparseVector.of(5, new int[] {2, 2}, new double[] {1.0, -1.0});
        assertEquals(0, z.nnz());
        assertEquals(0.0, z.squaredNorm());
        assertEquals(SparseVector.dense(new double[5]), z);
        assertArrayEquals(new double[5], z.toDense());
    }

    @Test
    void dotAndDistanceMatchDenseReferenceOnRandomVectors() {
        SplittableRandom rnd = new SplittableRandom(42);
        for (int trial = 0; trial < 500; trial++) {
            int dim = 1 + rnd.nextInt(40);
            double[] a = randomDense(rnd, dim);
            double[] b = randomDense(rnd, dim);
            SparseVector sa = randomlyEncoded(rnd, a);
            SparseVector sb = randomlyEncoded(rnd, b);

            double dot = 0.0;
            double dist = 0.0;
            double normB = 0.0;
            for (int i = 0; i < dim; i++) {
                dot += a[i] * b[i];
                dist += (a[i] - b[i]) * (a[i] - b[i]);
                normB += b[i] * b[i];
            }
            assertEquals(dot, sa.dot(sb), 1e-9 * (1 + Math.abs(dot)));
            assertEquals(dot, sa.dot(b), 1e-9 * (1 + Math.abs(dot)));
            assertEquals(dist, sa.squaredDistance(sb), 1e-9 * (1 + dist));
            assertEquals(dist, sb.squaredDistance(sa), 1e-9 * (1 + dist));
            assertEquals(dist, sa.squaredDistance(b, normB), 1e-9 * (1 + dist + normB));
            assertArrayEquals(a, sa.toDense());
        }
    }

    @Test
    void sparseDistanceIsExactWhereNormExpansionCancels() {
        double big = 1e8;
        SparseVector a = SparseVector.dense(new double[] {big, 1.0});
        SparseVector b = SparseVector.dense(new double[] {big, 1.0 + 1e-6});
        double expected = 1e-12;
        assertEquals(expected, a.squaredDistance(b), 1e-20);
        assertEquals(0.0, a.squaredDistance(a));
        assertEquals(0.0, a.squaredDistance(SparseVector.dense(new double[] {big, 1.0})));
    }

    @Test
    void denseFastPathIsNonNegative() {
        double[] c = {1e8, 3.0};
        SparseVector p = SparseVector.dense(c.clone());
        double norm = c[0] * c[0] + c[1] * c[1];
        assertTrue(p.squaredDistance(c, norm) >= 0.0);
        assertTrue(p.squaredDistance(c, norm * (1 - 1e-15)) >= 0.0);
    }

    /** Review F1: the dense-center distance must stay accurate when ||c||^2 dwarfs the distance itself. */
    @Test
    void denseDistanceDoesNotCancelFarFromTheOrigin() {
        double base = 1e9;
        double[] c = {base, base};
        double norm = c[0] * c[0] + c[1] * c[1];
        SparseVector full = SparseVector.dense(new double[] {base + 0.5, base - 0.25});
        assertEquals(0.3125, full.squaredDistance(c, norm), 1e-12);

        // Only coordinate 0 stored: the off-support term base^2 is large and must survive as is.
        SparseVector partial = SparseVector.of(2, new int[] {0}, new double[] {base + 0.5});
        assertEquals(0.25 + base * base, partial.squaredDistance(c, norm), 1e-15 * base * base);

        // Almost all of ||c||^2 sits on the point's support; the small off-support mass 1e-6 must not vanish.
        double[] skewed = {base, 1e-3};
        double skewedNorm = skewed[0] * skewed[0] + skewed[1] * skewed[1];
        assertEquals(0.25 + 1e-6, partial.squaredDistance(skewed, skewedNorm), 1e-12);
    }

    @Test
    void addToAccumulatesScaledEntries() {
        double[] acc = {1.0, 1.0, 1.0};
        SparseVector.of(3, new int[] {0, 2}, new double[] {2.0, -1.0}).addTo(acc, 3.0);
        assertArrayEquals(new double[] {7.0, 1.0, -2.0}, acc);
    }

    @Test
    void dimensionMismatchIsRejected() {
        SparseVector a = SparseVector.dense(new double[] {1, 2});
        SparseVector b = SparseVector.dense(new double[] {1, 2, 3});
        assertThrows(IllegalArgumentException.class, () -> a.dot(b));
        assertThrows(IllegalArgumentException.class, () -> a.squaredDistance(b));
        assertThrows(IllegalArgumentException.class, () -> a.squaredDistance(new double[3], 0.0));
        assertThrows(IllegalArgumentException.class, () -> a.addTo(new double[3], 1.0));
    }

    @Test
    void estimatedBytesFollowsDocumentedModel() {
        // 40 (object) + align(16 + 4*3) + align(16 + 8*3)
        SparseVector v = SparseVector.of(100, new int[] {1, 2, 3}, new double[] {1, 2, 3});
        assertEquals(40 + 32 + 40, v.estimatedBytes());
    }

    private static double[] randomDense(SplittableRandom rnd, int dim) {
        double[] v = new double[dim];
        for (int i = 0; i < dim; i++) {
            if (rnd.nextDouble() < 0.4) {
                v[i] = rnd.nextDouble(-10, 10);
            }
        }
        return v;
    }

    /** Encodes {@code dense} as shuffled entries with some values split into duplicates and explicit zeros. */
    private static SparseVector randomlyEncoded(SplittableRandom rnd, double[] dense) {
        int[] idx = new int[3 * dense.length];
        double[] val = new double[idx.length];
        int n = 0;
        for (int i = 0; i < dense.length; i++) {
            if (dense[i] != 0.0 && rnd.nextBoolean()) {
                // Halving is exact for normal doubles, so the duplicates sum back to dense[i] bit-for-bit.
                double half = dense[i] / 2;
                idx[n] = i;
                val[n++] = half;
                idx[n] = i;
                val[n++] = half;
            } else if (dense[i] != 0.0) {
                idx[n] = i;
                val[n++] = dense[i];
            } else if (rnd.nextBoolean()) {
                idx[n] = i;
                val[n++] = 0.0;
            }
        }
        for (int k = n - 1; k > 0; k--) {
            int j = rnd.nextInt(k + 1);
            int ti = idx[k];
            idx[k] = idx[j];
            idx[j] = ti;
            double tv = val[k];
            val[k] = val[j];
            val[j] = tv;
        }
        return SparseVector.of(dense.length, Arrays.copyOf(idx, n), Arrays.copyOf(val, n));
    }
}
