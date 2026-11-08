package io.github.c0rwin.coreset.experiments.data;

import java.io.IOException;
import java.io.Serializable;
import java.util.Map;

/**
 * A dataset every system reads the same way: {@link #size()} rows of dimension {@link #dim()}, addressable by row
 * range, so a gRPC shard thread, a Spark partition and a Flink subtask each stream exactly their own rows. Sources are
 * serialized into Spark tasks and Flink operators and reopen the underlying storage there.
 */
public interface PointSource extends Serializable {

    String id();

    long size();

    int dim();

    /** Streams rows {@code [from, to)}; cheap to open, so every partition or subtask opens its own reader. */
    ChunkReader open(long from, long to) throws IOException;

    /** Where the rows come from (file, generator, checksum), recorded with every result. */
    Map<String, Object> provenance();

    /** First row of shard {@code i} of {@code shards} contiguous, near-equal shards. */
    static long shardStart(long n, int i, int shards) {
        return (n / shards) * i + Math.min(i, n % shards);
    }
}
