package io.github.c0rwin.coreset.node;

import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/** Deterministic derivation of every generator in a session from the session seed. */
final class Seeds {

    private static final long GOLDEN = 0x9E3779B97F4A7C15L;
    private static final long TREE_SALT = 0x5DEECE66DL;

    private Seeds() {
    }

    /** SplitMix64 finaliser: a bijection with good avalanche, so nearby (seed, seq) pairs give unrelated streams. */
    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Generator for compressing batch {@code seq}; independent of worker, arrival order and retries. */
    static RandomGenerator batch(long sessionSeed, long seq) {
        return new SplittableRandom(mix64(sessionSeed ^ mix64(seq * GOLDEN + 1)));
    }

    /** One merge generator per worker, split in index order from a root derived from the session seed. */
    static RandomGenerator[] workers(long sessionSeed, int workers) {
        SplittableRandom root = new SplittableRandom(mix64(sessionSeed ^ TREE_SALT));
        RandomGenerator[] out = new RandomGenerator[workers];
        for (int i = 0; i < workers; i++) {
            out[i] = root.split();
        }
        return out;
    }
}
