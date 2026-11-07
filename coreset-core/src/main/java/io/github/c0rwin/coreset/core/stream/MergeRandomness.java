package io.github.c0rwin.coreset.core.stream;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Supplies the generator for each compression a {@link MergeReduceTree} performs. A merge is identified by the level
 * it carries out of and the 0-based index of the batch whose insertion triggered it; the final reduce of a snapshot
 * uses level {@link #FINAL_LEVEL} and the number of batches added so far.
 *
 * <p>With {@link #derived} every merge's generator is a pure function of its position, so a tree rebuilt from the
 * same batches, or {@linkplain MergeReduceTree#restore restored} from checkpointed levels, compresses bit-identically
 * no matter which process, thread or retry attempt runs it.
 */
@FunctionalInterface
public interface MergeRandomness {

    /** Level passed for the final reduce of a snapshot. */
    int FINAL_LEVEL = -1;

    RandomGenerator forMerge(int level, long sequence);

    /** One generator for every merge: results depend on all compressions since construction, in order. */
    static MergeRandomness shared(RandomGenerator rng) {
        Objects.requireNonNull(rng, "rng");
        return (level, sequence) -> rng;
    }

    /** Fresh generator per merge, seeded by {@link #derive derive(seed, stream, level, sequence)}. */
    static MergeRandomness derived(long seed, long stream) {
        return (level, sequence) -> new SplittableRandom(derive(seed, stream, level, sequence));
    }

    /**
     * Mixes the four coordinates into one seed with chained SplitMix64 finalisers, so neighbouring coordinates give
     * unrelated streams. Callers pick {@code stream} to separate independent consumers of one global seed (shards,
     * reduce rounds, solver restarts).
     */
    static long derive(long seed, long stream, long level, long sequence) {
        long h = mix64(seed ^ 0x9E3779B97F4A7C15L);
        h = mix64(h ^ mix64(stream + 0xD1B54A32D192ED03L));
        h = mix64(h ^ mix64(level + 0x8CB92BA72F3D8DD7L));
        return mix64(h ^ mix64(sequence + 0xABC98388FB8FAC03L));
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
