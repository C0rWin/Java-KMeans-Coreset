package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Per-level encoded size and compression depth of a {@code MergeReduceTree}, kept in step with its binary-counter
 * carries so metrics and depth never need a pass over the tree. Level {@code i} is occupied iff bit {@code i} of the
 * tree's batch count is set.
 */
final class LevelShadow {

    private final long[] bytes = new long[64];
    private final int[] depth = new int[64];
    private long totalBytes;

    /**
     * Mirrors one {@code add}: a batch of depth {@code batchDepth} was carried to level
     * {@code numberOfTrailingZeros(batchesAfter)}, merging (one compression each) with every level below it.
     *
     * @param carryBytes encoded size of the coreset that landed (the tree's lowest occupied level after the add)
     * @return its depth
     */
    int onAdd(long batchesAfter, int batchDepth, long carryBytes) {
        int level = Long.numberOfTrailingZeros(batchesAfter);
        int d = batchDepth;
        for (int i = 0; i < level; i++) {
            d = Math.max(depth[i], d) + 1;
            totalBytes -= bytes[i];
            bytes[i] = 0;
            depth[i] = 0;
        }
        bytes[level] = carryBytes;
        depth[level] = d;
        totalBytes += carryBytes;
        return d;
    }

    /** Rebuilds the shadow of occupied {@code levels} (lowest first); null {@code depths} means depth = level index. */
    void restore(long batches, List<Coreset> levels, List<Integer> depths) {
        clear();
        int next = 0;
        for (int level = 0; level < 64; level++) {
            if ((batches & (1L << level)) != 0) {
                bytes[level] = PointCodec.coresetLength(levels.get(next));
                depth[level] = depths == null ? level : depths.get(next);
                totalBytes += bytes[level];
                next++;
            }
        }
    }

    void clear() {
        Arrays.fill(bytes, 0);
        Arrays.fill(depth, 0);
        totalBytes = 0;
    }

    /** Sum of the encoded sizes of the occupied levels' coresets. */
    long totalBytes() {
        return totalBytes;
    }

    /** Largest depth among the occupied levels of a tree holding {@code batches} batches; 0 if empty. */
    int maxDepth(long batches) {
        int max = 0;
        for (long b = batches; b != 0; b &= b - 1) {
            max = Math.max(max, depth[Long.numberOfTrailingZeros(b)]);
        }
        return max;
    }

    /** Depths of the occupied levels, lowest first. */
    List<Integer> depths(long batches) {
        List<Integer> out = new ArrayList<>(Long.bitCount(batches));
        for (long b = batches; b != 0; b &= b - 1) {
            out.add(depth[Long.numberOfTrailingZeros(b)]);
        }
        return out;
    }
}
