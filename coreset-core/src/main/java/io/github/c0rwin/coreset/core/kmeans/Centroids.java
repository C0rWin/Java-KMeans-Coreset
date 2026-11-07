package io.github.c0rwin.coreset.core.kmeans;

import io.github.c0rwin.coreset.core.Guards;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.SparseVector;
import java.util.Arrays;
import java.util.List;

/**
 * The centers of one Lloyd run and its assignment step. {@link Dense} keeps m x dim arrays; {@link Sparse} keeps every
 * center as its sorted non-zero coordinates, so memory and update time follow the clusters' stored entries instead of
 * m * dim. Both perform the same floating-point operations in the same order and return bit-identical results;
 * {@link #layoutFor} picks one. Points are streamed from their {@link PointSet} segments.
 *
 * <p>Bounded runs keep Hamerly's bounds per point: {@code upper} on the distance to its own center and {@code lower}
 * on the distance to every other center, loosened by how far the centers move. A point whose bounds separate is not
 * looked at; otherwise its own distance tightens {@code upper}, and only if the bounds still overlap are all centers
 * scanned. The bounds carry a margin above the kernels' worst-case rounding ({@link Bounds}), so a skipped scan could
 * not have changed the assignment: assignments, centers and costs equal those of full scans bit for bit. A skipped
 * point whose center moved has a stale distance (NaN) until {@link #refresh}. Not thread-safe: one instance per run.
 * Per point: 16 bytes of bounds in bounded runs, plus 8 for the work lists of the sparse layout.
 */
abstract class Centroids {

    /** Dense arrays are always allowed up to this many center cells (8 MiB per copy, two copies). */
    static final long DENSE_CELLS = 1L << 20;

    private static final double ROUND_UP = 1.0 + 0x1p-50;
    private static final double ROUND_DOWN = 1.0 - 0x1p-50;
    /** Distances below this are never trusted to separate bounds (underflow, see {@link Bounds#TINY_SQUARED}). */
    private static final double TINY = 0x1p-500;

    enum Layout { DENSE, SPARSE }

    final int dim;
    final int m;
    final double[] norms;
    /** Whether the bounds are kept; false runs scan every point in full each step. */
    final boolean bounded;
    /** Per center: an upper bound on how far it moved since the last assignment step (bounded runs). */
    final double[] moved;
    final double margin; // relative, on distances
    final double[] upper; // per point: >= distance to its own center
    final double[] lower; // per point: <= distance to every other center
    /** Per center: at most half the distance to its nearest other center, or null when not kept. */
    double[] half;
    private final int[] positive; // positive-weight points per center
    private boolean started;
    private double move1; // the largest entry of moved, at the start of a bounded step
    private double move2; // the second largest
    private int fastest = -1; // the center that moved by move1
    /** Point-center distances evaluated so far. */
    long evaluations;
    /** Point rows read so far by assignment and update steps. */
    long pointsScanned;
    /** Point-steps whose full scan the bounds made unnecessary. */
    long skipped;

    Centroids(int m, int dim, PointSet points, boolean bounds) {
        final int n = points.size();
        this.dim = dim;
        this.m = m;
        this.norms = new double[m];
        double err = Bounds.relativeError(2L * Math.min(dim, points.storedEntries()));
        this.bounded = bounds && Bounds.usable(err);
        this.margin = 2.0 * err;
        this.moved = new double[m];
        this.upper = bounded ? new double[n] : null;
        this.lower = bounded ? new double[n] : null;
        this.positive = new int[m];
    }

    /**
     * Dense when {@code m * dim <= max(DENSE_CELLS, 2 * totalNnz)}: the two dense copies then stay within a small
     * multiple of the input size and their O(m * dim) sweep per update costs no more than a pass over the input.
     */
    static Layout layoutFor(int m, int dim, long totalNnz) {
        return (long) m * dim <= Math.max(DENSE_CELLS, 2 * totalNnz) ? Layout.DENSE : Layout.SPARSE;
    }

    static Centroids create(Layout layout, List<SparseVector> initial, int dim, PointSet points, boolean bounds) {
        Centroids c = layout == Layout.DENSE
                ? new Dense(initial.size(), dim, points, bounds)
                : new Sparse(initial.size(), dim, points, bounds);
        for (int j = 0; j < initial.size(); j++) {
            c.set(j, initial.get(j));
        }
        return c;
    }

    /** {@code sum w dist} over the positive-weight points, in point order. */
    static double cost(PointSet points, double[] dist) {
        double total = 0.0;
        for (int t = 0, g = 0; t < points.segmentCount(); t++) {
            PointBlock b = points.segment(t);
            for (int i = 0, n = b.size(); i < n; i++, g++) {
                double w = b.weight(i);
                if (w > 0.0) { // a zero weight times an overflowed distance would be NaN
                    total += w * dist[g];
                }
            }
        }
        return total;
    }

    /**
     * Assigns every point to its nearest center (ties: lowest index); returns whether any assignment changed.
     * {@code dist[g]} receives the squared distance to it, or NaN when the bounds skipped g after its center moved.
     */
    final boolean assign(PointSet points, int[] assignment, double[] dist) {
        if (bounded) {
            prepare();
        }
        boolean changed;
        if (bounded && started) {
            summarizeMoves();
            changed = boundedStep(points, assignment, dist);
        } else {
            changed = fullStep(points, assignment, dist);
        }
        started = true;
        Arrays.fill(moved, 0.0);
        return changed;
    }

    /** Bounded runs: called before each assignment step, once the centers are in place. */
    void prepare() {
    }

    /**
     * Scans every point against every center (ties: lowest index), setting {@code dist}, the assignment and, in bounded
     * runs, both bounds of every point; returns whether any assignment changed.
     */
    abstract boolean fullStep(PointSet points, int[] assignment, double[] dist);

    /**
     * The bounded step: {@link #loosen}, then for the points whose bounds overlap the own distance and {@link #tighten},
     * then a scan of the points whose bounds still overlap; returns whether any assignment changed.
     */
    abstract boolean boundedStep(PointSet points, int[] assignment, double[] dist);

    private void summarizeMoves() {
        move1 = 0.0;
        move2 = 0.0;
        fastest = -1;
        for (int j = 0; j < m; j++) {
            if (moved[j] > move1) {
                move2 = move1;
                move1 = moved[j];
                fastest = j;
            } else if (moved[j] > move2) {
                move2 = moved[j];
            }
        }
    }

    /**
     * Loosens point {@code g}'s bounds by the center moves since the last step ({@code a} its center), marking its
     * distance stale if {@code a} moved; returns whether the bounds overlap, i.e. the point needs a closer look.
     */
    final boolean loosen(int g, int a, double[] dist) {
        double u = upper[g];
        double mv = moved[a];
        if (mv > 0.0) {
            u = (u + mv) * ROUND_UP;
            upper[g] = u;
            dist[g] = Double.NaN;
        }
        double l = lower[g];
        double others = a == fastest ? move2 : move1;
        if (others > 0.0) {
            l -= others;
            l = l > 0.0 ? l * ROUND_DOWN : 0.0;
            lower[g] = l;
        }
        return !separated(u, bound(l, a));
    }

    /** Tightens point {@code g}'s upper bound to its own distance (computed square {@code d2}); returns overlap. */
    final boolean tighten(int g, int a, double d2) {
        double u = upperOf(d2);
        upper[g] = u;
        return !separated(u, bound(lower[g], a));
    }

    /** Assigns point {@code g} of weight {@code w} to center {@code b}; returns whether its assignment changed. */
    final boolean reassign(int g, int b, double w, int[] assignment) {
        int a = assignment[g];
        if (a == b) {
            return false;
        }
        assignment[g] = b;
        if (w > 0.0) {
            if (a >= 0) {
                positive[a]--;
            }
            positive[b]++;
        }
        return true;
    }

    /** Whether {@code d(own) <= upper} and {@code d(other) >= lower} prove the own center strictly nearest. */
    final boolean separated(double upper, double lower) {
        return upper * (1.0 + margin) < lower && lower > TINY;
    }

    /**
     * The better of a point's lower bound {@code l} and Hamerly's center bound: a point within half the distance from
     * its center {@code a} to every other center is nearer to {@code a}.
     */
    final double bound(double l, int a) {
        return half != null ? Math.max(l, half[a]) : l;
    }

    /** An upper bound on the true distance whose computed square is {@code d2}. */
    final double upperOf(double d2) {
        return Math.sqrt(d2) * (1.0 + margin);
    }

    /**
     * A lower bound on the true distance whose computed square is {@code d2}, the second-smallest of a full scan;
     * infinite there only for one center or by overflow, which gives no bound.
     */
    final double lowerOf(double d2) {
        return d2 < Double.POSITIVE_INFINITY ? Math.sqrt(d2) * (1.0 - margin) : m == 1 ? d2 : 0.0;
    }

    /** Whether some center has no positive-weight point assigned (it will be re-seeded after the update). */
    final boolean hasEmptyCluster() {
        for (int j = 0; j < m; j++) {
            if (positive[j] == 0) {
                return true;
            }
        }
        return false;
    }

    /** Computes the stale (NaN) distances, so that {@code dist} holds every point's exact squared distance. */
    final void refresh(PointSet points, int[] assignment, double[] dist) {
        if (bounded) {
            evaluations += refreshStale(points, assignment, dist);
        }
    }

    /** {@code dist[g]} = squared distance to the assigned center wherever {@code dist[g]} is NaN; returns the count. */
    abstract long refreshStale(PointSet points, int[] assignment, double[] dist);

    /**
     * Moves every center whose cluster has positive weight to the cluster's weighted mean and leaves the others in
     * place; fills {@code clusterWeight}, summed in point order. Bounded runs add each move to {@link #moved}.
     */
    abstract void moveToMeans(PointSet points, int[] assignment, double[] clusterWeight);

    /** Places center {@code j} at {@code v}. */
    final void set(int j, SparseVector v) {
        if (bounded && started) {
            addMove(j, squaredDistanceTo(j, v));
        }
        place(j, v);
    }

    /** Records that center {@code j} moved by a distance whose computed square is {@code d2}. */
    final void addMove(int j, double d2) {
        moved[j] = (moved[j] + upperOf(d2)) * ROUND_UP;
    }

    abstract void place(int j, SparseVector v);

    /** Exact squared distance between center {@code j} and {@code v}. */
    abstract double squaredDistanceTo(int j, SparseVector v);

    abstract SparseVector center(int j);

    /**
     * m x dim arrays; the spare copy receives the next update's sums. Assignment is point-major and sequential: a point
     * is read once while all centers, small by {@link #layoutFor}, stay cached. Bounded runs also keep lower bounds on
     * the center-center distances (while m^2 stays within the dense-cell budget), which give Hamerly's center bound and
     * let a scan skip center j whenever {@code d(c_best, c_j) > 2 d(p, c_best)} proves j farther than the best so far.
     */
    static final class Dense extends Centroids {
        private final double[][] centers;
        private final double[][] spare;
        private final double[] cc; // [a * m + j]: lower bound on d(c_a, c_j), bounded runs only

        Dense(int m, int dim, PointSet points, boolean bounds) {
            super(m, dim, points, bounds);
            centers = new double[m][dim];
            spare = new double[m][dim];
            boolean keep = bounded && (long) m * m <= Math.max(DENSE_CELLS, 2 * points.storedEntries());
            cc = keep ? new double[m * m] : null;
            half = keep ? new double[m] : null;
        }

        /** O(m^2 dim): the center-center bounds and {@code half}. */
        @Override
        void prepare() {
            if (cc == null) {
                return;
            }
            Arrays.fill(half, Double.POSITIVE_INFINITY);
            for (int a = 0; a < m; a++) {
                double[] ca = centers[a];
                for (int j = a + 1; j < m; j++) {
                    double[] cj = centers[j];
                    double s = 0.0;
                    for (int d = 0; d < dim; d++) {
                        double diff = ca[d] - cj[d];
                        s += diff * diff;
                    }
                    double low = Math.sqrt(s) * (1.0 - margin);
                    cc[a * m + j] = low;
                    cc[j * m + a] = low;
                    half[a] = Math.min(half[a], 0.5 * low);
                    half[j] = Math.min(half[j], 0.5 * low);
                }
            }
        }

        /**
         * Nearest center of row {@code i} = global row {@code g} (ties: lowest index), starting from center {@code a}
         * at computed squared distance {@code da}, or from scratch when {@code a < 0}; returns it and sets
         * {@code dist[g]} and, with {@code bound}, {@code lower[g]}.
         */
        private int nearest(PointBlock p, int i, int g, int a, double da, double[] dist, boolean bound) {
            int evals = 0;
            int b = a;
            double d1 = da;
            if (a < 0) {
                b = 0;
                d1 = p.squaredDistance(i, centers[0], norms[0]);
                evals = 1;
            }
            final int first = b;
            double d2 = Double.POSITIVE_INFINITY; // smallest computed one besides the best
            boolean otherComputed = false;
            double skippedLow = Double.POSITIVE_INFINITY; // lower bound on every skipped center
            double ub = cc != null ? upperOf(d1) : 0.0;
            double limit = (2.0 + margin) * ub;
            for (int j = 0; j < m; j++) {
                if (j == first) {
                    continue;
                }
                if (cc != null) {
                    double c = cc[b * m + j];
                    if (c > limit) { // d(p, c_j) >= c - d(p, c_b) > (1 + margin) d(p, c_b): computes larger
                        skippedLow = Math.min(skippedLow, (c - ub) * ROUND_DOWN);
                        continue;
                    }
                }
                double d = p.squaredDistance(i, centers[j], norms[j]);
                evals++;
                otherComputed = true;
                if (d < d1 || (d == d1 && j < b)) {
                    d2 = d1;
                    d1 = d;
                    b = j;
                    if (cc != null) {
                        ub = upperOf(d1);
                        limit = (2.0 + margin) * ub;
                    }
                } else if (d < d2) {
                    d2 = d;
                }
            }
            evaluations += evals;
            dist[g] = d1;
            if (bound) {
                double low = otherComputed ? lowerOf(d2) : m == 1 ? Double.POSITIVE_INFINITY : skippedLow;
                lower[g] = Math.min(low, skippedLow);
            }
            return b;
        }

        @Override
        boolean fullStep(PointSet points, int[] assignment, double[] dist) {
            boolean changed = false;
            for (int s = 0, g = 0; s < points.segmentCount(); s++) {
                PointBlock p = points.segment(s);
                for (int i = 0, n = p.size(); i < n; i++, g++) {
                    if ((g & 1023) == 0) {
                        Guards.checkInterrupted("Lloyd assignment");
                    }
                    int b = nearest(p, i, g, -1, 0.0, dist, bounded);
                    changed |= reassign(g, b, p.weight(i), assignment);
                    if (bounded) {
                        upper[g] = upperOf(dist[g]);
                    }
                }
            }
            pointsScanned += points.size();
            return changed;
        }

        /** One sequential pass: bounds, then the own distance, then the other centers, each only when still needed. */
        @Override
        boolean boundedStep(PointSet points, int[] assignment, double[] dist) {
            boolean changed = false;
            long read = 0;
            long scanned = 0;
            for (int s = 0, g = 0; s < points.segmentCount(); s++) {
                PointBlock p = points.segment(s);
                for (int i = 0, n = p.size(); i < n; i++, g++) {
                    if ((g & 1023) == 0) {
                        Guards.checkInterrupted("Lloyd assignment");
                    }
                    int a = assignment[g];
                    if (!loosen(g, a, dist)) {
                        continue;
                    }
                    double own = p.squaredDistance(i, centers[a], norms[a]);
                    read++;
                    dist[g] = own;
                    if (!tighten(g, a, own)) {
                        continue;
                    }
                    scanned++;
                    int b = nearest(p, i, g, a, own, dist, true);
                    changed |= reassign(g, b, p.weight(i), assignment);
                    upper[g] = upperOf(dist[g]);
                }
            }
            evaluations += read;
            pointsScanned += read;
            skipped += points.size() - scanned;
            return changed;
        }

        @Override
        long refreshStale(PointSet points, int[] assignment, double[] dist) {
            long count = 0;
            for (int s = 0, g = 0; s < points.segmentCount(); s++) {
                PointBlock p = points.segment(s);
                for (int i = 0, n = p.size(); i < n; i++, g++) {
                    if (Double.isNaN(dist[g])) {
                        int a = assignment[g];
                        dist[g] = p.squaredDistance(i, centers[a], norms[a]);
                        count++;
                    }
                }
            }
            pointsScanned += count;
            return count;
        }

        @Override
        void moveToMeans(PointSet points, int[] assignment, double[] clusterWeight) {
            for (double[] row : spare) {
                Arrays.fill(row, 0.0);
            }
            Arrays.fill(clusterWeight, 0.0);
            for (int t = 0, g = 0; t < points.segmentCount(); t++) {
                PointBlock p = points.segment(t);
                for (int i = 0, n = p.size(); i < n; i++, g++) {
                    double w = p.weight(i);
                    if (w > 0.0) {
                        int j = assignment[g];
                        p.addTo(i, spare[j], w);
                        clusterWeight[j] += w;
                    }
                }
            }
            pointsScanned += points.size();
            for (int j = 0; j < m; j++) {
                if (clusterWeight[j] > 0.0) {
                    double[] acc = spare[j];
                    double[] old = centers[j];
                    double inv = 1.0 / clusterWeight[j];
                    double norm = 0.0;
                    double move = 0.0;
                    for (int d = 0; d < acc.length; d++) {
                        double v = acc[d] * inv;
                        acc[d] = v;
                        norm += v * v;
                        double diff = v - old[d];
                        move += diff * diff;
                    }
                    spare[j] = old;
                    centers[j] = acc;
                    norms[j] = norm;
                    if (bounded) {
                        addMove(j, move);
                    }
                }
            }
        }

        @Override
        void place(int j, SparseVector v) {
            double[] row = centers[j];
            Arrays.fill(row, 0.0);
            v.addTo(row, 1.0);
            norms[j] = v.squaredNorm();
        }

        @Override
        double squaredDistanceTo(int j, SparseVector v) {
            double[] row = centers[j];
            double s = 0.0;
            for (int d = 0, k = 0, nnz = v.nnz(); d < row.length; d++) {
                double x = k < nnz && v.indexAt(k) == d ? v.valueAt(k++) : 0.0;
                double diff = row[d] - x;
                s += diff * diff;
            }
            return s;
        }

        @Override
        SparseVector center(int j) {
            return SparseVector.dense(centers[j]);
        }
    }

    /**
     * Per center: sorted non-zero coordinates and values in reusable buffers, double-buffered so an update can measure
     * the move. Assignment is center-major: each center is scattered once into an O(dim) scratch when
     * {@link SparseVector#scatterPays} says the buffer pays, so a distance costs one array read per stored point entry;
     * otherwise point entries are looked up in the center's sorted coordinates. Updates accumulate each cluster, in point
     * order, in the same scratch (without it {@link ClusterSums} sorts the cluster's entries), exactly the additions of
     * the dense layout. About 12 bytes per stored center coordinate, twice.
     */
    static final class Sparse extends Centroids {
        private final ClusterSums sums;
        private final double[] scratch; // zero outside the center being scattered or the cluster being summed
        private final long[] touchedBits; // coordinates touched while summing one cluster
        private int[] touched = new int[0];
        private int[][] indices;
        private double[][] values;
        private final int[] nnz;
        private int[][] oldIndices;
        private double[][] oldValues;
        private final int[] oldNnz;
        private final int[] bucketStart; // own(): listed points grouped by assigned center
        private final int[] rows; // global rows to work on, ascending
        private final int[] best; // scan result per global row; bucket scratch in own()

        Sparse(int m, int dim, PointSet points, boolean bounds) {
            super(m, dim, points, bounds);
            this.sums = new ClusterSums(points);
            boolean scatter = SparseVector.scatterPays(dim, points.size(), points.storedEntries());
            this.scratch = scatter ? new double[dim] : null;
            this.touchedBits = scatter ? new long[(dim + 63) >>> 6] : null;
            indices = new int[m][0];
            values = new double[m][0];
            nnz = new int[m];
            oldIndices = new int[m][0];
            oldValues = new double[m][0];
            oldNnz = new int[m];
            bucketStart = new int[m + 1];
            rows = new int[points.size()];
            best = new int[points.size()];
        }

        @Override
        boolean fullStep(PointSet points, int[] assignment, double[] dist) {
            final int n = points.size();
            if (n == 0) {
                return false;
            }
            for (int g = 0; g < n; g++) {
                rows[g] = g;
            }
            evaluations += scan(points, rows, n, dist, bounded ? lower : null);
            return commit(points, n, assignment, dist);
        }

        @Override
        boolean boundedStep(PointSet points, int[] assignment, double[] dist) {
            final int n = points.size();
            int count = 0;
            for (int g = 0; g < n; g++) {
                if (loosen(g, assignment[g], dist)) {
                    rows[count++] = g;
                }
            }
            int full = 0;
            if (count > 0) {
                own(points, rows, count, assignment, dist);
                evaluations += count;
                for (int t = 0; t < count; t++) {
                    int g = rows[t];
                    if (tighten(g, assignment[g], dist[g])) {
                        rows[full++] = g;
                    }
                }
            }
            skipped += n - full;
            if (full == 0) {
                return false;
            }
            evaluations += scan(points, rows, full, dist, lower);
            return commit(points, full, assignment, dist);
        }

        /** Applies a scan of {@code rows[0, count)}: assignments, positive-weight counts and upper bounds. */
        private boolean commit(PointSet points, int count, int[] assignment, double[] dist) {
            boolean changed = false;
            int seg = 0;
            int segStart = 0;
            int segEnd = points.start(1);
            PointBlock p = points.segment(0);
            for (int t = 0; t < count; t++) {
                int g = rows[t];
                while (g >= segEnd) {
                    seg++;
                    segStart = segEnd;
                    segEnd = points.start(seg + 1);
                    p = points.segment(seg);
                }
                changed |= reassign(g, best[g], p.weight(g - segStart), assignment);
                if (bounded) {
                    upper[g] = upperOf(dist[g]);
                }
            }
            return changed;
        }

        @Override
        long refreshStale(PointSet points, int[] assignment, double[] dist) {
            int count = 0;
            for (int g = 0; g < dist.length; g++) {
                if (Double.isNaN(dist[g])) {
                    rows[count++] = g;
                }
            }
            if (count > 0) {
                own(points, rows, count, assignment, dist);
            }
            return count;
        }

        private double distance(PointBlock p, int i, int j) {
            return scratch != null
                    ? p.squaredDistanceScattered(i, scratch, indices[j], values[j], 0, nnz[j], norms[j])
                    : p.squaredDistanceSorted(i, indices[j], values[j], 0, nnz[j], norms[j]);
        }

        private void scatter(int j) {
            if (scratch != null) {
                int[] idx = indices[j];
                double[] val = values[j];
                for (int t = 0, k = nnz[j]; t < k; t++) {
                    scratch[idx[t]] = val[t];
                }
            }
        }

        private void clear(int j) {
            if (scratch != null) {
                int[] idx = indices[j];
                for (int t = 0, k = nnz[j]; t < k; t++) {
                    scratch[idx[t]] = 0.0;
                }
            }
        }

        /** Center-major over the listed points grouped by assigned center (stable counting sort into {@code best}). */
        /** {@code dist[g]} = squared distance to the assigned center for the listed points {@code rows[0, count)}. */
        private void own(PointSet points, int[] rows, int count, int[] assignment, double[] dist) {
            Arrays.fill(bucketStart, 0);
            for (int t = 0; t < count; t++) {
                bucketStart[assignment[rows[t]] + 1]++;
            }
            for (int j = 0; j < m; j++) {
                bucketStart[j + 1] += bucketStart[j];
            }
            for (int t = 0; t < count; t++) {
                int g = rows[t];
                best[bucketStart[assignment[g]]++] = g;
            }
            for (int j = m; j > 0; j--) {
                bucketStart[j] = bucketStart[j - 1];
            }
            bucketStart[0] = 0;
            for (int j = 0; j < m; j++) {
                int from = bucketStart[j];
                int to = bucketStart[j + 1];
                if (from == to) {
                    continue;
                }
                Guards.checkInterrupted("Lloyd assignment");
                scatter(j);
                int seg = 0;
                int segStart = 0;
                int segEnd = points.start(1);
                PointBlock p = points.segment(0);
                for (int t = from; t < to; t++) {
                    int g = best[t];
                    while (g >= segEnd) {
                        seg++;
                        segStart = segEnd;
                        segEnd = points.start(seg + 1);
                        p = points.segment(seg);
                    }
                    dist[g] = distance(p, g - segStart, j);
                }
                clear(j);
            }
            pointsScanned += count;
        }

        /**
         * Center-major: each center's coordinates stay cached while the listed points stream past it. Visiting centers
         * in index order with a strict comparison keeps the lowest index on ties, as the point-major order does. Fills
         * {@code best[g]}, {@code dist[g]} and, if {@code second} is not null, {@link #lowerOf} the second-smallest
         * distance in {@code second[g]}; returns the distances evaluated.
         */
        private long scan(PointSet points, int[] rows, int count, double[] dist, double[] second) {
            for (int j = 0; j < m; j++) {
                Guards.checkInterrupted("Lloyd assignment");
                scatter(j);
                int seg = 0;
                int segStart = 0;
                int segEnd = points.start(1);
                PointBlock p = points.segment(0);
                for (int t = 0; t < count; t++) {
                    int g = rows[t];
                    while (g >= segEnd) {
                        seg++;
                        segStart = segEnd;
                        segEnd = points.start(seg + 1);
                        p = points.segment(seg);
                    }
                    double d = distance(p, g - segStart, j);
                    if (j == 0) {
                        best[g] = 0;
                        dist[g] = d;
                        if (second != null) {
                            second[g] = Double.POSITIVE_INFINITY;
                        }
                    } else if (d < dist[g]) {
                        if (second != null) {
                            second[g] = dist[g];
                        }
                        dist[g] = d;
                        best[g] = j;
                    } else if (second != null && d < second[g]) {
                        second[g] = d;
                    }
                }
                clear(j);
            }
            if (second != null) {
                for (int t = 0; t < count; t++) {
                    int g = rows[t];
                    second[g] = lowerOf(second[g]);
                }
            }
            pointsScanned += (long) count * m;
            return (long) count * m;
        }

        @Override
        void moveToMeans(PointSet points, int[] assignment, double[] clusterWeight) {
            double[] w = sums.group(assignment, m);
            System.arraycopy(w, 0, clusterWeight, 0, m);
            pointsScanned += 2L * points.size() + sums.start(m);
            for (int j = 0; j < m; j++) {
                if (clusterWeight[j] <= 0.0) {
                    continue;
                }
                int[] spareIdx = oldIndices[j];
                double[] spareVal = oldValues[j];
                oldIndices[j] = indices[j];
                oldValues[j] = values[j];
                oldNnz[j] = nnz[j];
                indices[j] = spareIdx;
                values[j] = spareVal;
                if (scratch != null) {
                    sumCluster(points, j, 1.0 / clusterWeight[j]);
                } else {
                    ClusterSums.Mean mean = sums.mean(j, false);
                    install(j, mean.indices, mean.values, mean.indices.length, mean.squaredNorm);
                }
                if (bounded) {
                    addMove(j, mergeDistance(oldIndices[j], oldValues[j], oldNnz[j], indices[j], values[j], nnz[j]));
                }
            }
        }

        /**
         * Accumulates cluster {@code j}'s members, in point order, into the scratch ({@code acc[c] += w * v}, the dense
         * layout's additions), then emits the touched coordinates in increasing order as the new center.
         */
        private void sumCluster(PointSet points, int j, double inv) {
            int[] mem = sums.members();
            long need = Math.min(dim, sums.entries(j));
            if (touched.length < need) {
                touched = new int[(int) need];
            }
            int count = 0;
            int seg = 0;
            int segStart = 0;
            int segEnd = points.start(1);
            PointBlock p = points.segment(0);
            for (int t = sums.start(j), end = sums.start(j + 1); t < end; t++) {
                int g = mem[t];
                while (g >= segEnd) {
                    seg++;
                    segStart = segEnd;
                    segEnd = points.start(seg + 1);
                    p = points.segment(seg);
                }
                int i = g - segStart;
                double w = p.weight(i);
                for (int e = p.rowStart(i), stop = p.rowEnd(i); e < stop; e++) {
                    int c = p.index(e);
                    long bit = 1L << c;
                    if ((touchedBits[c >>> 6] & bit) == 0) {
                        touchedBits[c >>> 6] |= bit;
                        touched[count++] = c;
                    }
                    scratch[c] += w * p.value(e);
                }
            }
            if (touchedBits.length <= 4L * count) { // a sweep of the bitmap is cheaper than sorting
                int k = 0;
                for (int word = 0; word < touchedBits.length; word++) {
                    long bits = touchedBits[word];
                    while (bits != 0) {
                        touched[k++] = (word << 6) | Long.numberOfTrailingZeros(bits);
                        bits &= bits - 1;
                    }
                    touchedBits[word] = 0L;
                }
            } else {
                Arrays.sort(touched, 0, count);
                for (int k = 0; k < count; k++) {
                    touchedBits[touched[k] >>> 6] = 0L;
                }
            }
            ensure(j, count);
            int[] idx = indices[j];
            double[] val = values[j];
            int out = 0;
            double norm = 0.0;
            for (int k = 0; k < count; k++) {
                int c = touched[k];
                double mu = scratch[c] * inv;
                scratch[c] = 0.0;
                norm += mu * mu;
                if (mu != 0.0) {
                    idx[out] = c;
                    val[out++] = mu;
                }
            }
            nnz[j] = out;
            norms[j] = norm;
        }

        private void ensure(int j, int capacity) {
            if (indices[j].length < capacity) {
                indices[j] = new int[capacity];
                values[j] = new double[capacity];
            }
        }

        /** Copies the non-zero entries of {@code (idx, val)[0, count)} into center {@code j}'s buffers. */
        private void install(int j, int[] idx, double[] val, int count, double norm) {
            ensure(j, count);
            int out = 0;
            for (int t = 0; t < count; t++) {
                if (val[t] != 0.0) {
                    indices[j][out] = idx[t];
                    values[j][out++] = val[t];
                }
            }
            nnz[j] = out;
            norms[j] = norm;
        }

        @Override
        void place(int j, SparseVector v) {
            int k = v.nnz();
            ensure(j, k);
            for (int t = 0; t < k; t++) {
                indices[j][t] = v.indexAt(t);
                values[j][t] = v.valueAt(t);
            }
            nnz[j] = k;
            norms[j] = v.squaredNorm();
        }

        @Override
        double squaredDistanceTo(int j, SparseVector v) {
            int[] ai = indices[j];
            double[] av = values[j];
            int an = nnz[j];
            int bn = v.nnz();
            double s = 0.0;
            int a = 0;
            int b = 0;
            while (a < an || b < bn) {
                int x = a < an ? ai[a] : Integer.MAX_VALUE;
                int y = b < bn ? v.indexAt(b) : Integer.MAX_VALUE;
                double d = x == y ? av[a++] - v.valueAt(b++) : x < y ? av[a++] : v.valueAt(b++);
                s += d * d;
            }
            return s;
        }

        /** Exact {@code ||a - b||^2} of two sorted coordinate lists by merging them. */
        private static double mergeDistance(int[] ai, double[] av, int an, int[] bi, double[] bv, int bn) {
            double s = 0.0;
            int a = 0;
            int b = 0;
            while (a < an && b < bn) {
                double d;
                if (ai[a] == bi[b]) {
                    d = av[a++] - bv[b++];
                } else if (ai[a] < bi[b]) {
                    d = av[a++];
                } else {
                    d = bv[b++];
                }
                s += d * d;
            }
            for (; a < an; a++) {
                s += av[a] * av[a];
            }
            for (; b < bn; b++) {
                s += bv[b] * bv[b];
            }
            return s;
        }

        @Override
        SparseVector center(int j) {
            return SparseVector.ofOwned(dim, Arrays.copyOf(indices[j], nnz[j]), Arrays.copyOf(values[j], nnz[j]));
        }
    }
}
