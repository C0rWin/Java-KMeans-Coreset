package io.github.c0rwin.coreset.testbed.faults.framework;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/** Per-thread scratch arrays (grown on demand) and the packing permutations of a message. */
final class Buffers {

    private static final ThreadLocal<Buffers> LOCAL = ThreadLocal.withInitial(Buffers::new);

    private int[] ints = new int[0];
    private int[] ints2 = new int[0];
    private boolean[] flags = new boolean[0];
    private long[] keys = new long[0];

    static Buffers get() {
        return LOCAL.get();
    }

    int[] ints(int n) {
        if (ints.length < n) {
            ints = new int[n];
        }
        return ints;
    }

    int[] ints2(int n) {
        if (ints2.length < n) {
            ints2 = new int[n];
        }
        return ints2;
    }

    boolean[] flags(int n) {
        if (flags.length < n) {
            flags = new boolean[n];
        }
        return flags;
    }

    long[] keys(int n) {
        if (keys.length < n) {
            keys = new long[n];
        }
        return keys;
    }

    /** Fisher-Yates on {@code a[0..len)}: random packing. */
    static void shuffle(int[] a, int len, RandomGenerator rng) {
        for (int i = len - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    /** Sorts draws {@code a[0..len)} by (class, member): content-dependent packing. */
    void sortByClass(int[] a, int len, CostSpace space) {
        long[] k = keys(len);
        for (int i = 0; i < len; i++) {
            k[i] = ((long) space.cls[a[i]] << 32) | a[i];
        }
        Arrays.sort(k, 0, len);
        for (int i = 0; i < len; i++) {
            a[i] = (int) k[i];
        }
    }
}
