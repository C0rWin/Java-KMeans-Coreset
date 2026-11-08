package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import java.util.Arrays;
import java.util.Locale;
import java.util.random.RandomGenerator;

/** Assignments of the rows of a dataset to nodes; every row lands on exactly one node, rows ascending per node. */
final class Partitions {

    enum Kind {
        /** Shuffle, then deal equal contiguous parts. */
        RANDOM,
        /** Rows grouped by reference cluster (cluster order shuffled), then equal contiguous parts: the adversary. */
        CLUSTER_SORTED,
        /** Rows sorted by their projection on a random Gaussian direction: the geometric adversary. */
        PROJECTION_SORTED,
        /** Random assignment with node sizes proportional to {@code 1/(i+1)}: unequal parts. */
        ZIPF_SIZES,
        /** Dataset order as given, equal contiguous parts. */
        INPUT;

        String id() {
            return name().toLowerCase(Locale.ROOT).replace('_', '-');
        }
    }

    private Partitions() {
    }

    static int[][] split(Kind kind, int[] cluster, PointSet points, int nodes, RandomGenerator rng) {
        int n = cluster.length;
        if (nodes < 1 || nodes > n) {
            throw new IllegalArgumentException("need 1 <= nodes <= rows: " + nodes);
        }
        int[] order = order(kind, cluster, points, rng);
        int[] sizes = kind == Kind.ZIPF_SIZES ? zipfSizes(n, nodes, rng) : equalSizes(n, nodes);
        int[][] parts = new int[nodes][];
        for (int v = 0, at = 0; v < nodes; v++) {
            parts[v] = Arrays.copyOfRange(order, at, at + sizes[v]);
            Arrays.sort(parts[v]);
            at += sizes[v];
        }
        return parts;
    }

    /** The row order a kind deals from: a stream order for merge-reduce, or the input of {@link #split}. */
    static int[] order(Kind kind, int[] cluster, PointSet points, RandomGenerator rng) {
        return switch (kind) {
            case RANDOM, ZIPF_SIZES -> permutation(cluster.length, rng);
            case CLUSTER_SORTED -> clusterOrder(cluster, rng);
            case PROJECTION_SORTED -> projectionOrder(points, rng);
            case INPUT -> identity(cluster.length);
        };
    }

    static int[] equalSizes(int n, int nodes) {
        int[] s = new int[nodes];
        for (int v = 0; v < nodes; v++) {
            s[v] = (int) ((long) n * (v + 1) / nodes - (long) n * v / nodes);
        }
        return s;
    }

    /** Sizes proportional to {@code 1/(i+1)} in shuffled node order, each at least 1, summing to n. */
    static int[] zipfSizes(int n, int nodes, RandomGenerator rng) {
        double h = 0.0;
        for (int i = 0; i < nodes; i++) {
            h += 1.0 / (i + 1);
        }
        int[] s = new int[nodes];
        int assigned = 0;
        for (int i = 0; i < nodes; i++) {
            s[i] = Math.max(1, (int) Math.floor(n / ((i + 1) * h)));
            assigned += s[i];
        }
        for (int i = 0; assigned < n; i = (i + 1) % nodes) {
            s[i]++;
            assigned++;
        }
        for (int i = nodes - 1; assigned > n; i = (i + nodes - 1) % nodes) {
            if (s[i] > 1) {
                s[i]--;
                assigned--;
            }
        }
        shuffle(s, rng);
        return s;
    }

    static int[] permutation(int n, RandomGenerator rng) {
        int[] p = identity(n);
        shuffle(p, rng);
        return p;
    }

    static int[] identity(int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        return p;
    }

    static void shuffle(int[] a, RandomGenerator rng) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    private static int[] clusterOrder(int[] cluster, RandomGenerator rng) {
        int labels = 0;
        for (int c : cluster) {
            labels = Math.max(labels, c + 1);
        }
        int[] rank = permutation(labels, rng);
        int[] start = new int[labels + 1];
        for (int c : cluster) {
            start[rank[c] + 1]++;
        }
        for (int c = 0; c < labels; c++) {
            start[c + 1] += start[c];
        }
        int[] order = new int[cluster.length];
        for (int r = 0; r < cluster.length; r++) {
            order[start[rank[cluster[r]]]++] = r;
        }
        return order;
    }

    private static int[] projectionOrder(PointSet points, RandomGenerator rng) {
        int d = points.dim();
        double[] dir = new double[d];
        for (int t = 0; t < d; t++) {
            dir[t] = rng.nextGaussian();
        }
        int n = points.size();
        double[] proj = new double[n];
        for (int g = 0; g < n; g++) {
            int s = points.segmentOf(g);
            PointBlock b = points.segment(s);
            proj[g] = b.vector(g - points.start(s)).dot(dir);
        }
        Integer[] idx = new Integer[n];
        for (int g = 0; g < n; g++) {
            idx[g] = g;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(proj[a], proj[b]) != 0 ? Double.compare(proj[a], proj[b])
                : Integer.compare(a, b));
        int[] order = new int[n];
        for (int g = 0; g < n; g++) {
            order[g] = idx[g];
        }
        return order;
    }

    /** Node of every row. */
    static int[] owner(int[][] parts, int n) {
        int[] owner = new int[n];
        for (int v = 0; v < parts.length; v++) {
            for (int r : parts[v]) {
                owner[r] = v;
            }
        }
        return owner;
    }
}
