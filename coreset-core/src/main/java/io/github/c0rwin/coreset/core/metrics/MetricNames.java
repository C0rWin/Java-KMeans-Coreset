package io.github.c0rwin.coreset.core.metrics;

/**
 * Names of every metric emitted by coreset-core. "Span" names are prefixes: closing the span emits
 * {@code <prefix>.wall_ns}, {@code <prefix>.cpu_ns} (nanoseconds) and {@code <prefix>.alloc_bytes} (bytes).
 */
public final class MetricNames {

    private MetricNames() {
    }

    /** Span suffix: wall-clock duration, nanoseconds. */
    public static final String WALL_NS_SUFFIX = ".wall_ns";
    /** Span suffix: CPU time of the calling thread, nanoseconds. */
    public static final String CPU_NS_SUFFIX = ".cpu_ns";
    /** Span suffix: bytes allocated by the calling thread. */
    public static final String ALLOC_BYTES_SUFFIX = ".alloc_bytes";
    /** Counter suffix: NaN samples a distribution received and dropped (see {@code RecordingMetrics#record}). */
    public static final String NAN_SAMPLES_SUFFIX = ".nan_samples";

    /** Span around one k-means++ seeding call. */
    public static final String KMEANSPP_SEED = "kmeanspp.seed";
    /** Counter: squared-distance evaluations performed by k-means++ seeding. */
    public static final String KMEANSPP_DISTANCE_EVALUATIONS = "kmeanspp.distance_evaluations";

    /** Span around one weighted Lloyd call (includes seeding when seeding is performed). */
    public static final String LLOYD_CLUSTER = "lloyd.cluster";
    /** Distribution: Lloyd update steps per call. */
    public static final String LLOYD_ITERATIONS = "lloyd.iterations";
    /** Counter: squared-distance evaluations performed by Lloyd assignment steps (excludes seeding). */
    public static final String LLOYD_DISTANCE_EVALUATIONS = "lloyd.distance_evaluations";
    /** Counter: Lloyd calls that kept sparse centers because m * dim was large next to the stored entries. */
    public static final String LLOYD_SPARSE_CENTERS = "lloyd.sparse_centers";
    /** Counter: points read by Lloyd passes (input read, assignment and update steps; excludes seeding). */
    public static final String LLOYD_POINTS_SCANNED = "lloyd.points_scanned";
    /** Distribution: weighted cost after each assignment step, in squared input units. */
    public static final String LLOYD_ITERATION_COST = "lloyd.iteration_cost";
    /** Counter: centers re-seeded because their cluster had no positive weight. */
    public static final String LLOYD_EMPTY_CLUSTER_RESEEDS = "lloyd.empty_cluster_reseeds";
    /** Span around one {@code WeightedLloyd.bestOf} call, all restarts included. */
    public static final String LLOYD_BEST_OF = "lloyd.best_of";
    /** Distribution: final cost of each {@code bestOf} restart; its spread shows how bad local optima get. */
    public static final String LLOYD_RESTART_COST = "lloyd.restart_cost";

    /** Span around one {@code MergeReduceTree.add}, including all carries. */
    public static final String MRTREE_ADD = "mrtree.add";
    /** Span around one merge (union + compress) inside the tree. */
    public static final String MRTREE_MERGE = "mrtree.merge";
    /** Counter: merges performed across all levels. */
    public static final String MRTREE_MERGES = "mrtree.merges";
    /** Distribution: merges (carries) triggered by one add. */
    public static final String MRTREE_CARRY_CHAIN_LENGTH = "mrtree.carry_chain_length";
    /** Distribution: {@code (weightOut - weightIn) / weightIn} per merge; 0 for weight-preserving algorithms. */
    public static final String MRTREE_MERGE_RELATIVE_WEIGHT_DRIFT = "mrtree.merge.relative_weight_drift";
    /** Distribution: points in a snapshot. */
    public static final String MRTREE_SNAPSHOT_POINTS = "mrtree.snapshot.points";
    /** Distribution: estimated heap bytes of a snapshot. */
    public static final String MRTREE_SNAPSHOT_BYTES = "mrtree.snapshot.estimated_bytes";
    /** Span around {@code compressedSnapshot}. */
    public static final String MRTREE_COMPRESSED_SNAPSHOT = "mrtree.compressed_snapshot";
    /** Distribution: points in a compressed snapshot. */
    public static final String MRTREE_COMPRESSED_SNAPSHOT_POINTS = "mrtree.compressed_snapshot.points";
    /** Distribution: estimated heap bytes of a compressed snapshot. */
    public static final String MRTREE_COMPRESSED_SNAPSHOT_BYTES = "mrtree.compressed_snapshot.estimated_bytes";

    /** Span around one {@code Seeding.cluster} call (seeding plus optional Lloyd refinement). */
    public static final String SEEDING_CLUSTER = "seeding.cluster";
    /** Counter: squared-distance evaluations performed by D^2 seeding inside {@code Seeding} (excludes Lloyd). */
    public static final String SEEDING_DISTANCE_EVALUATIONS = "seeding.distance_evaluations";
    /** Distribution: centers returned by one seeding call. */
    public static final String SEEDING_CENTERS = "seeding.centers";
    /** Distribution: sample-and-remove rounds performed by one bicriteria seeding call. */
    public static final String SEEDING_BICRITERIA_ROUNDS = "seeding.bicriteria.rounds";
    /**
     * Counter: points visited by full passes of {@code Seeding} (input read, one per center, bicriteria round
     * bookkeeping, final clustering); excludes D^2 draws and Lloyd refinement.
     */
    public static final String SEEDING_POINTS_SCANNED = "seeding.points_scanned";
    /** Counter: points visited by D^2 draws in {@code Seeding}; each draw stops at the drawn point, so at most n. */
    public static final String SEEDING_DRAW_POINTS_SCANNED = "seeding.draw_points_scanned";

    /** Span around one uniform-sampling compress call. */
    public static final String UNIFORM_COMPRESS = "uniform.compress";
    /** Counter: uniform calls that returned the input unchanged because it had at most m points. */
    public static final String UNIFORM_PASSTHROUGH = "uniform.passthrough";
    /** Distribution: distinct points in a uniform-sampling output. */
    public static final String UNIFORM_OUTPUT_POINTS = "uniform.output_points";
    /** Counter: points read by uniform sampling (one pass over the input plus the distinct output rows). */
    public static final String UNIFORM_POINTS_SCANNED = "uniform.points_scanned";

    /** Span around one sensitivity-sampling compress call (includes seeding). */
    public static final String SENSITIVITY_COMPRESS = "sensitivity.compress";
    /** Counter: sensitivity calls that returned the input unchanged because it had at most m points. */
    public static final String SENSITIVITY_PASSTHROUGH = "sensitivity.passthrough";
    /** Distribution: distinct points in a sensitivity-sampling output. */
    public static final String SENSITIVITY_OUTPUT_POINTS = "sensitivity.output_points";
    /** Distribution: actual total sensitivity {@code S = sum_p s_p} used to normalise the sampling distribution. */
    public static final String SENSITIVITY_TOTAL = "sensitivity.total";
    /** Counter: calls whose seeding had zero cost, so only the per-cluster mass term was used. */
    public static final String SENSITIVITY_ZERO_COST_SEEDINGS = "sensitivity.zero_cost_seedings";
    /**
     * Counter: points read by sensitivity sampling after seeding (the bound pass plus the distinct output rows; one
     * more pass when the clustering lacks per-cluster totals).
     */
    public static final String SENSITIVITY_POINTS_SCANNED = "sensitivity.points_scanned";

    /** Span around one mean-coreset compress call (includes clustering). */
    public static final String MEAN_COMPRESS = "mean.compress";
    /** Counter: mean-coreset calls that returned the input unchanged because it had at most m points. */
    public static final String MEAN_PASSTHROUGH = "mean.passthrough";
    /** Distribution: points (non-empty clusters) in a mean-coreset output. */
    public static final String MEAN_OUTPUT_POINTS = "mean.output_points";
    /** Distribution: additive cost {@code sum_i cost(P_i, mean_i)} added to the offset by one call. */
    public static final String MEAN_ADDED_OFFSET = "mean.added_offset";
    /** Distribution: number of means m chosen by the adaptive search. */
    public static final String MEAN_ADAPTIVE_M = "mean.adaptive.m";

    private static final String MRTREE_LEVEL_PREFIX = "mrtree.level.";
    // A tree over at most 2^63 batches has levels 0..63; precomputed so recording builds no strings.
    private static final String[] LEVEL_MERGES = levelNames(".merges");
    private static final String[] LEVEL_POINTS = levelNames(".points");

    private static String[] levelNames(String suffix) {
        String[] out = new String[64];
        for (int i = 0; i < out.length; i++) {
            out[i] = MRTREE_LEVEL_PREFIX + i + suffix;
        }
        return out;
    }

    /** Counter: merges whose inputs lived at {@code level} (result is carried to {@code level + 1}). */
    public static String mrtreeLevelMerges(int level) {
        return level >= 0 && level < 64 ? LEVEL_MERGES[level] : MRTREE_LEVEL_PREFIX + level + ".merges";
    }

    /** Distribution: size (points) of each coreset stored at {@code level}. */
    public static String mrtreeLevelPoints(int level) {
        return level >= 0 && level < 64 ? LEVEL_POINTS[level] : MRTREE_LEVEL_PREFIX + level + ".points";
    }
}
