package io.github.c0rwin.coreset.flink;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.algo.Algorithms;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.Objects;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Immutable configuration of the Flink coreset pipelines. The algorithm is resolved with {@link Algorithms#create}
 * when the configuration is built (so bad parameters fail at job construction) and again in every operator.
 *
 * <p>Restoring a checkpoint requires the same algorithm, parameters, seed and chunk size; triggers may change, at the
 * price of bit-identical results.
 */
public final class CoresetConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    private final String algorithmId;
    private final TreeMap<String, String> params;
    private final long seed;
    private final int chunkSize;
    private final int chunksPerPartial;
    private final long partialIntervalMillis;
    private final long partialEventTimeIntervalMillis;
    private final int snapshotEveryPartials;
    private final long snapshotIntervalMillis;
    private final boolean compressSnapshots;
    private final String uidPrefix;

    private CoresetConfig(Builder b) {
        this.algorithmId = b.algorithmId;
        this.params = new TreeMap<>(b.params);
        this.seed = b.seed;
        this.chunkSize = b.chunkSize > 0 ? b.chunkSize : defaultChunkSize(b.params);
        this.chunksPerPartial = b.chunksPerPartial;
        this.partialIntervalMillis = b.partialIntervalMillis;
        this.partialEventTimeIntervalMillis = b.partialEventTimeIntervalMillis;
        this.snapshotEveryPartials = b.snapshotEveryPartials;
        this.snapshotIntervalMillis = b.snapshotIntervalMillis;
        this.compressSnapshots = b.compressSnapshots;
        this.uidPrefix = b.uidPrefix;
    }

    /**
     * @param algorithmId {@code uniform}, {@code sensitivity} or {@code mean}
     * @param params      algorithm parameters, see {@link Algorithms#create}
     * @param seed        global seed: equal per-subtask input, configuration and seed give bit-identical coresets
     */
    public static Builder builder(String algorithmId, Map<String, String> params, long seed) {
        return new Builder(algorithmId, params, seed);
    }

    /**
     * Twice the target size ({@code m}, or {@code maxM} for adaptive {@code mean}), so the first merge halves two raw
     * chunks down to m; the same rule as {@code coreset-spark}.
     *
     * @throws IllegalArgumentException if neither parameter is present or parsable
     */
    public static int defaultChunkSize(Map<String, String> params) {
        String m = params.containsKey("m") ? params.get("m") : params.get("maxM");
        if (m == null) {
            throw new IllegalArgumentException("params carry neither m nor maxM; set chunkSize explicitly");
        }
        try {
            return Math.toIntExact(2L * Integer.parseInt(m.trim()));
        } catch (NumberFormatException | ArithmeticException e) {
            throw new IllegalArgumentException("cannot derive chunk size from m/maxM '" + m + "'", e);
        }
    }

    /**
     * 64-bit FNV-1a hash of what determines the checkpointed trees (algorithm, parameters, seed, chunk size); stored
     * in operator state so that a restore under a different configuration fails instead of mixing trees.
     */
    public long fingerprint() {
        String key = algorithmId + '\u0000' + params + '\u0000' + seed + '\u0000' + chunkSize;
        long h = 0xcbf29ce484222325L;
        for (byte b : key.getBytes(StandardCharsets.UTF_8)) {
            h ^= b & 0xFF;
            h *= 0x100000001b3L;
        }
        return h;
    }

    CoresetAlgorithm algorithm() {
        return Algorithms.create(algorithmId, params);
    }

    public String algorithmId() {
        return algorithmId;
    }

    public SortedMap<String, String> params() {
        return new TreeMap<>(params);
    }

    public long seed() {
        return seed;
    }

    /** Points per level-0 batch of each local tree. */
    public int chunkSize() {
        return chunkSize;
    }

    /** Full chunks after which a local subtask emits a partial; the count trigger. */
    public int chunksPerPartial() {
        return chunksPerPartial;
    }

    /** Processing-time period of extra partials, 0 if off. */
    public long partialIntervalMillis() {
        return partialIntervalMillis;
    }

    /** Event-time period of extra partials (emitted when the watermark crosses a multiple of it), 0 if off. */
    public long partialEventTimeIntervalMillis() {
        return partialEventTimeIntervalMillis;
    }

    /** Partials after which the merge stage emits a snapshot, 0 if off. */
    public int snapshotEveryPartials() {
        return snapshotEveryPartials;
    }

    /** Processing-time period of merge-stage snapshots, 0 if off. */
    public long snapshotIntervalMillis() {
        return snapshotIntervalMillis;
    }

    /** Whether snapshots carry a compressed coreset besides the union of levels. */
    public boolean compressSnapshots() {
        return compressSnapshots;
    }

    /** Prefix of the operator uids, which map checkpointed state to operators across job versions. */
    public String uidPrefix() {
        return uidPrefix;
    }

    @Override
    public String toString() {
        return "CoresetConfig{" + algorithmId + " " + params + ", seed=" + seed + ", chunkSize=" + chunkSize
                + ", chunksPerPartial=" + chunksPerPartial + ", partialIntervalMillis=" + partialIntervalMillis
                + ", partialEventTimeIntervalMillis=" + partialEventTimeIntervalMillis + ", snapshotEveryPartials="
                + snapshotEveryPartials + ", snapshotIntervalMillis=" + snapshotIntervalMillis
                + ", compressSnapshots=" + compressSnapshots + ", uidPrefix=" + uidPrefix + "}";
    }

    /** Builder of {@link CoresetConfig}; every setter validates its argument. */
    public static final class Builder {
        private final String algorithmId;
        private final Map<String, String> params;
        private final long seed;
        private int chunkSize;
        private int chunksPerPartial = 8;
        private long partialIntervalMillis;
        private long partialEventTimeIntervalMillis;
        private int snapshotEveryPartials = 1;
        private long snapshotIntervalMillis;
        private boolean compressSnapshots;
        private String uidPrefix = "coreset";

        private Builder(String algorithmId, Map<String, String> params, long seed) {
            this.algorithmId = Objects.requireNonNull(algorithmId, "algorithmId");
            this.params = new TreeMap<>(Objects.requireNonNull(params, "params"));
            this.seed = seed;
            Algorithms.create(algorithmId, params);
        }

        /** Points per level-0 batch; default {@link #defaultChunkSize}. */
        public Builder chunkSize(int chunkSize) {
            this.chunkSize = positive("chunkSize", chunkSize);
            return this;
        }

        /** Count trigger: a partial after this many full chunks per subtask; default 8. */
        public Builder chunksPerPartial(int chunks) {
            this.chunksPerPartial = positive("chunksPerPartial", chunks);
            return this;
        }

        /** Processing-time trigger for partials; {@link Duration#ZERO} turns it off (the default). */
        public Builder partialInterval(Duration interval) {
            this.partialIntervalMillis = millis("partialInterval", interval);
            return this;
        }

        /** Event-time trigger for partials; {@link Duration#ZERO} turns it off (the default). */
        public Builder partialEventTimeInterval(Duration interval) {
            this.partialEventTimeIntervalMillis = millis("partialEventTimeInterval", interval);
            return this;
        }

        /** Merge-stage snapshot after this many partials; 0 turns the count trigger off. Default 1. */
        public Builder snapshotEveryPartials(int partials) {
            if (partials < 0) {
                throw new IllegalArgumentException("snapshotEveryPartials must be >= 0: " + partials);
            }
            this.snapshotEveryPartials = partials;
            return this;
        }

        /** Processing-time trigger for merge-stage snapshots; {@link Duration#ZERO} turns it off (the default). */
        public Builder snapshotInterval(Duration interval) {
            this.snapshotIntervalMillis = millis("snapshotInterval", interval);
            return this;
        }

        /** Also emit each snapshot compressed once; default false. */
        public Builder compressSnapshots(boolean compress) {
            this.compressSnapshots = compress;
            return this;
        }

        /** Prefix of operator uids; default {@code coreset}. Use distinct prefixes for two pipelines in one job. */
        public Builder uidPrefix(String prefix) {
            if (prefix == null || prefix.isBlank()) {
                throw new IllegalArgumentException("uidPrefix must not be blank");
            }
            this.uidPrefix = prefix;
            return this;
        }

        public CoresetConfig build() {
            return new CoresetConfig(this);
        }

        private static int positive(String name, int v) {
            if (v < 1) {
                throw new IllegalArgumentException(name + " must be >= 1: " + v);
            }
            return v;
        }

        private static long millis(String name, Duration d) {
            Objects.requireNonNull(d, name);
            if (d.isNegative()) {
                throw new IllegalArgumentException(name + " must not be negative: " + d);
            }
            return d.toMillis();
        }
    }
}
