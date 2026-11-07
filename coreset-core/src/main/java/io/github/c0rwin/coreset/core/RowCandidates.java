package io.github.c0rwin.coreset.core;

import java.util.Arrays;

/**
 * Up to {@code capacity} block rows (candidate centers) whose squared distances to a point are computed together, in
 * one pass over the point's entries. With a coordinate map (worth its O(dim) ints on sparse input, see
 * {@link SparseVector#scatterPays}) the candidates' coordinates index one small value table, so a point entry costs
 * one map read whatever the number of candidates; without it every distance merges supports. Each distance is
 * bit-identical to {@link PointBlock#squaredDistanceScatteredToRow} (with the map) or
 * {@link PointBlock#squaredDistance(int, PointBlock, int)} (without). Not thread-safe.
 */
public final class RowCandidates {

    private final int capacity;
    private final int[] slot; // coordinate -> 1 + table row, 0 when no candidate stores it; null without a map
    private double[] table = new double[0]; // [row * capacity + t] = candidate t's value at the row's coordinate
    private int[] coords = new int[0]; // coordinates holding a slot, to clear them
    private int used;
    private final PointBlock[] blocks;
    private final int[] rows;
    private int count;
    private final double[] on;
    private final double[] centerOn;
    private final int[] hits;

    /** @param map whether to keep an O(dim) coordinate map (sparse input) instead of merging supports */
    public RowCandidates(int dim, int capacity, boolean map) {
        if (capacity < 1) {
            throw new IllegalArgumentException("capacity must be >= 1: " + capacity);
        }
        this.capacity = capacity;
        this.slot = map ? new int[dim] : null;
        this.blocks = new PointBlock[capacity];
        this.rows = new int[capacity];
        this.on = new double[capacity];
        this.centerOn = new double[capacity];
        this.hits = new int[capacity];
    }

    public int size() {
        return count;
    }

    /** Appends row {@code row} of {@code b} as the next candidate. */
    public void add(PointBlock b, int row) {
        if (count == capacity) {
            throw new IllegalStateException("already holds " + capacity + " candidates");
        }
        int t = count++;
        blocks[t] = b;
        rows[t] = row;
        if (slot == null) {
            return;
        }
        for (int e = b.rowStart(row), end = b.rowEnd(row); e < end; e++) {
            int c = b.index(e);
            int s = slot[c];
            if (s == 0) {
                if (used == coords.length) {
                    int cap = Math.max(16, 2 * coords.length);
                    coords = Arrays.copyOf(coords, cap);
                    table = Arrays.copyOf(table, cap * capacity);
                }
                coords[used] = c;
                Arrays.fill(table, used * capacity, (used + 1) * capacity, 0.0);
                s = ++used;
                slot[c] = s;
            }
            table[(s - 1) * capacity + t] = b.value(e);
        }
    }

    /** Removes every candidate; O(their stored entries). */
    public void clear() {
        if (slot != null) {
            for (int u = 0; u < used; u++) {
                slot[coords[u]] = 0;
            }
        }
        used = 0;
        count = 0;
    }

    /**
     * Squared distances from row {@code i} of {@code p} to at least the candidates {@code t < 32} whose bit
     * {@code 1 << t} is set in {@code mask}, written to {@code out[t]} bit-identically to the unmasked call; returns how
     * many were computed. With a coordinate map a non-zero mask computes all of them, since the fused pass costs about
     * the same; without one only the selected candidates are merged and the other entries of {@code out} stay unchanged.
     */
    public int squaredDistances(PointBlock p, int i, double[] out, int mask) {
        int selected = count >= Integer.SIZE ? mask : mask & ((1 << count) - 1);
        if (slot != null) {
            if (selected == 0) {
                return 0;
            }
            squaredDistances(p, i, out);
            return count;
        }
        int computed = 0;
        for (int bits = selected; bits != 0; bits &= bits - 1) {
            int t = Integer.numberOfTrailingZeros(bits);
            out[t] = p.squaredDistance(i, blocks[t], rows[t]);
            computed++;
        }
        return computed;
    }

    /** {@code out[t]} = squared distance from row {@code i} of {@code p} to candidate {@code t}, for every candidate. */
    public void squaredDistances(PointBlock p, int i, double[] out) {
        final int n = count;
        if (slot == null) {
            for (int t = 0; t < n; t++) {
                out[t] = p.squaredDistance(i, blocks[t], rows[t]);
            }
            return;
        }
        for (int t = 0; t < n; t++) {
            on[t] = 0.0;
            centerOn[t] = 0.0;
            hits[t] = 0;
        }
        final int from = p.rowStart(i);
        final int to = p.rowEnd(i);
        for (int k = from; k < to; k++) {
            double v = p.value(k);
            int s = slot[p.index(k)];
            if (s == 0) { // no candidate stores it: d = v - 0.0, and 0.0 * 0.0 adds nothing
                for (int t = 0; t < n; t++) {
                    on[t] += v * v;
                }
            } else {
                int row = (s - 1) * capacity;
                for (int t = 0; t < n; t++) {
                    double x = table[row + t];
                    double d = v - x;
                    on[t] += d * d;
                    centerOn[t] += x * x;
                    hits[t] += x != 0.0 ? 1 : 0;
                }
            }
        }
        for (int t = 0; t < n; t++) {
            PointBlock c = blocks[t];
            int r = rows[t];
            double norm = c.squaredNorm(r);
            double off = norm - centerOn[t];
            if (off > SparseVector.CANCELLATION_GUARD * norm) {
                out[t] = on[t] + off;
            } else if (hits[t] == c.nnz(r)) {
                out[t] = on[t];
            } else {
                out[t] = on[t] + c.offSupport(r, p, i);
            }
        }
    }
}
