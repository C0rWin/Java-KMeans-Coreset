package io.github.c0rwin.coreset.node;

import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/** Deterministic derivation of every generator in a session from the session seed. */
final class Seeds {

    private static final long GOLDEN = 0x9E3779B97F4A7C15L;
    private static final long TREE_SALT = 0x5DEECE66DL;
    private static final long PARTITION_SALT = 0x2545F4914F6CDD1DL;
    private static final long MESSAGE_SHARED_SALT = 0x6A09E667F3BCC909L;
    private static final long MESSAGE_DRAW_SALT = 0xBB67AE8584CAA73BL;
    private static final long MESSAGE_PACK_SALT = 0x3C6EF372FE94F82BL;

    private Seeds() {
    }

    /** SplitMix64 finaliser: a bijection with good avalanche, so nearby (seed, seq) pairs give unrelated streams. */
    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /**
     * Root of every generator of one node's share of a session. Distinct partitions get unrelated roots, so nodes
     * fed by one coordinator never sample with the same stream; partition 0 keeps the session seed as is.
     */
    static long partitionSeed(long sessionSeed, long partition) {
        return partition == 0 ? sessionSeed : mix64(sessionSeed ^ PARTITION_SALT ^ mix64(partition * GOLDEN));
    }

    /**
     * Generator for compressing batch {@code seq} of the producer with {@code producerSalt}; a function of
     * {@code (partitionSeed, producer, seq)} only, so it is independent of worker, arrival order and retries.
     * Salt 0 (the default producer) reproduces the unsalted stream.
     */
    static RandomGenerator batch(long partitionSeed, long producerSalt, long seq) {
        return new SplittableRandom(mix64(partitionSeed ^ producerSalt ^ mix64(seq * GOLDEN + 1)));
    }

    /** One merge generator per worker, split in index order from a root derived from the partition seed. */
    static RandomGenerator[] workers(long partitionSeed, int workers) {
        SplittableRandom root = new SplittableRandom(mix64(partitionSeed ^ TREE_SALT));
        RandomGenerator[] out = new RandomGenerator[workers];
        for (int i = 0; i < workers; i++) {
            out[i] = root.split();
        }
        return out;
    }

    /**
     * Message randomness every replica of one partition shares: the sensitivity seeding (hence q) or the cell
     * partition, so replicas holding the same summary agree on it.
     */
    static RandomGenerator messageShared(long partitionSeed) {
        return new SplittableRandom(mix64(partitionSeed ^ MESSAGE_SHARED_SALT));
    }

    /** The draws of sampling stream {@code stream}; distinct streams sample independently (thm:rateless). */
    static RandomGenerator messageDraws(long partitionSeed, long stream) {
        return new SplittableRandom(mix64(partitionSeed ^ MESSAGE_DRAW_SALT ^ mix64(stream * GOLDEN + 3)));
    }

    /** The record order of stream {@code stream}: independent of the draws, so packing is draw-oblivious. */
    static RandomGenerator messagePacking(long partitionSeed, long stream) {
        return new SplittableRandom(mix64(partitionSeed ^ MESSAGE_PACK_SALT ^ mix64(stream * GOLDEN + 5)));
    }
}
