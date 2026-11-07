package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class RowCandidatesTest {

    /** Rows with shared and private coordinates, one far from the origin so the cancellation guard trips. */
    private static PointBlock rows(long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        List<WeightedPoint> pts = new ArrayList<>();
        for (int i = 0; i < 60; i++) {
            int nnz = 1 + rnd.nextInt(5);
            int[] idx = new int[nnz];
            double[] val = new double[nnz];
            for (int k = 0; k < nnz; k++) {
                idx[k] = k < 2 ? k : 2 + rnd.nextInt(200);
                val[k] = (i % 7 == 0 && k == 0 ? 1e8 : 1.0) * rnd.nextGaussian();
            }
            pts.add(new WeightedPoint(SparseVector.of(300, idx, val), 1.0));
        }
        pts.add(new WeightedPoint(SparseVector.of(300, new int[] {0, 1}, new double[] {1e8, 2.0}), 1.0));
        pts.add(new WeightedPoint(SparseVector.of(300, new int[] {0}, new double[] {1e8}), 1.0));
        return PointBlock.of(pts);
    }

    @Test
    void fusedDistancesEqualThePerCandidateKernels() {
        PointBlock b = rows(1);
        PointBlock other = rows(2);
        for (boolean map : new boolean[] {true, false}) {
            RowCandidates c = new RowCandidates(300, 4, map);
            int[] cand = {0, 7, b.size() - 1, b.size() - 2};
            double[] scratch = new double[300];
            double[] out = new double[4];
            for (int round = 0; round < 2; round++) {
                c.clear();
                for (int t = 0; t < cand.length; t++) {
                    c.add(b, (cand[t] + round) % b.size());
                }
                assertEquals(4, c.size());
                for (PointBlock p : new PointBlock[] {b, other}) {
                    for (int i = 0; i < p.size(); i++) {
                        c.squaredDistances(p, i, out);
                        for (int t = 0; t < cand.length; t++) {
                            int j = (cand[t] + round) % b.size();
                            double expected;
                            if (map) {
                                b.scatterRow(j, scratch);
                                expected = p.squaredDistanceScatteredToRow(i, scratch, b, j);
                                b.clearRow(j, scratch);
                            } else {
                                expected = p.squaredDistance(i, b, j);
                            }
                            assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(out[t]),
                                    "map=" + map + " point " + i + " candidate " + j);
                        }
                    }
                }
            }
            assertThrows(IllegalStateException.class, () -> c.add(b, 1));
        }
    }

    @Test
    void maskedDistancesEqualTheFullPass() {
        PointBlock b = rows(3);
        for (boolean map : new boolean[] {true, false}) {
            RowCandidates c = new RowCandidates(300, 5, map);
            for (int t = 0; t < 5; t++) {
                c.add(b, (11 * t) % b.size());
            }
            double[] full = new double[5];
            double[] out = new double[5];
            for (int i = 0; i < b.size(); i++) {
                c.squaredDistances(b, i, full);
                for (int mask = 0; mask < 32; mask++) {
                    Arrays.fill(out, -1.0);
                    int computed = c.squaredDistances(b, i, out, mask | 0xffffffe0); // bits above size() are ignored
                    boolean all = map && mask != 0; // with a map one fused pass computes every candidate
                    assertEquals(all ? 5 : Integer.bitCount(mask), computed);
                    for (int t = 0; t < 5; t++) {
                        double expected = all || (mask & (1 << t)) != 0 ? full[t] : -1.0;
                        assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(out[t]),
                                "map=" + map + " point " + i + " mask " + mask);
                    }
                }
            }
        }
    }
}
