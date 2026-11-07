package io.github.c0rwin.coreset.node;

import java.util.Map;
import java.util.Objects;

/**
 * Configuration of one ingestion session.
 *
 * @param algorithmId   id passed to the node's {@link AlgorithmResolver}
 * @param params        algorithm parameters (copied)
 * @param seed          root of every random choice in the session: batch {@code seq} of a producer is compressed with a
 *                      generator derived from {@code (seed, partition, producer, seq)}, worker {@code i} merges with
 *                      one derived from {@code (seed, partition, i)}, where the partition is set per node at creation.
 *                      Batch coresets are therefore reproducible; the snapshot is reproducible
 *                      only in {@link TreeMode#SINGLE_WRITER} with a fixed per-worker arrival order (in
 *                      {@link TreeMode#CAS} the merging worker depends on timing)
 * @param workers       worker threads, {@code >= 1}; batch {@code seq} goes to worker {@code seq % workers}
 * @param queueCapacity bound of each worker queue; a power of two {@code >= 2} so it is exact for the ring buffer
 * @param treeMode      merge-reduce organisation
 */
public record SessionSpec(String algorithmId, Map<String, String> params, long seed, int workers, int queueCapacity,
                          TreeMode treeMode) {

    public static final int MAX_WORKERS = 1024;

    public SessionSpec {
        Objects.requireNonNull(algorithmId, "algorithmId");
        params = Map.copyOf(params);
        Objects.requireNonNull(treeMode, "treeMode");
        if (workers < 1 || workers > MAX_WORKERS) {
            throw new IllegalArgumentException("workers must be in [1, " + MAX_WORKERS + "]: " + workers);
        }
        if (queueCapacity < 2 || Integer.bitCount(queueCapacity) != 1) {
            throw new IllegalArgumentException("queueCapacity must be a power of two >= 2: " + queueCapacity);
        }
    }

    public SessionSpec(String algorithmId, Map<String, String> params, long seed, int workers, int queueCapacity) {
        this(algorithmId, params, seed, workers, queueCapacity, TreeMode.SINGLE_WRITER);
    }
}
