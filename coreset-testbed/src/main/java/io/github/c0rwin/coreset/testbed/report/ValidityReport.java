package io.github.c0rwin.coreset.testbed.report;

import java.util.List;
import java.util.Map;

/**
 * One run's validity report, {@code schemaVersion} 1: the layout of research notes section 6c plus a few additive
 * fields ({@code construction.perNode}, {@code validation}, per-level distortion). Nullable fields are absent or
 * meaningless for the run's mode or algorithm and serialise as {@code null}.
 */
public record ValidityReport(int schemaVersion, Run run, Config config, DatasetInfo dataset,
                             Construction construction, Integrity integrity, DistortionInfo distortion,
                             Downstream downstream, Validation validation, Verdict verdict) {

    public static final int SCHEMA_VERSION = 1;

    /**
     * @param concurrentRuns runs executing at the same time in this JVM; above 1, heap peaks and process CPU are not
     *                       attributable to a single run and are reported as null
     * @param loadAverage    one-minute system load average when the run was reported, or null where unsupported; a
     *                       value above the core count means wall times include waiting for a CPU
     */
    public record Run(String id, String startedAt, String gitCommit, Boolean gitDirty, Jvm jvm, Host host,
                      Engine engine, int concurrentRuns, Double loadAverage) {
    }

    public record Jvm(String vendor, String version, List<String> flags) {
    }

    public record Host(String os, String arch, String cpu, int cores, long memBytes) {
    }

    public record Engine(String name, String version) {
    }

    /**
     * @param epsilonTarget  target relative error when the algorithm was configured with one, else null
     * @param partitions     distributed: nodes times workers per node
     */
    public record Config(String suite, String problem, double z, int k, Algorithm algorithm, int targetSize,
                         Double epsilonTarget, Double delta, long seed, int replicate, String mode, String modeLabel,
                         Integer blockCount, Integer partitions, String transport, Boolean finalReduce) {
    }

    public record Algorithm(String label, String id, Map<String, String> params) {
    }

    public record DatasetInfo(String id, String sha256, int n, int d, long nnz, double weightSum,
                              List<String> preprocessing, String license, String source) {
    }

    /**
     * Construction cost. CPU and allocation are measured on the constructing thread (offline, stream) or summed over
     * the process's live threads (distributed).
     *
     * @param levels           most compressions any input point went through
     * @param reused           true when this coreset was built for another k of the same grid point and evaluated
     *                         again (k-independent algorithms), so the cost numbers are copied
     * @param bytesSerialized  protobuf size of the output coreset
     * @param bytesOnWire      gRPC payload bytes in both directions (grpc transport only)
     * @param estimatedBytes   modelled heap size of the output coreset
     */
    public record Construction(int outputSize, int levels, long reduceOps, long mergeOps, double additiveOffset,
                               double wallMs, double cpuMs, Double processCpuMs, long allocBytes, Long peakHeapBytes,
                               Long baselineHeapBytes, long gcCount, long gcMs, long distanceEvals, long pointsRead,
                               long bytesSerialized, Long bytesOnWire, long estimatedBytes, boolean reused,
                               List<Level> perLevel, List<Node> perNode) {
    }

    /**
     * Stream mode: the tree's summary of the first {@code 2^level} blocks, measured against exactly those input
     * points with the shared full-data candidates.
     */
    public record Level(int level, int size, long bytes, int inputPoints, Double distortionMax,
                        String argmaxGenerator) {
    }

    public record Node(int node, int size, long bytes, long batches) {
    }

    /**
     * @param oneSidedViolations candidates on which the coreset underestimates the full cost (one-sided algorithms
     *                           only, else null)
     * @param replayHash         SHA-256 of the coreset's canonical encoding
     * @param replayDeterministic rebuilding with the same seed gave the same digest; null when not checked
     */
    public record Integrity(double weightRelErr, double firstMomentRelErr, double secondMomentRelErr,
                            int negativeWeights, int nonFinite, Integer oneSidedViolations, String replayHash,
                            Boolean replayDeterministic) {
    }

    /**
     * @param ciCoverage fraction of confidence intervals containing the full cost (sampled coresets, offline only)
     */
    public record DistortionInfo(boolean isLowerBound, List<Candidate> candidates, double max, double mean,
                                 String argmaxGenerator, Double ciCoverage, DeltaSweep benchmarkDeltaSweep) {
    }

    /** @param ci95 confidence interval of the full cost from the coreset alone, when valid for this candidate */
    public record Candidate(String generator, double costFull, double costCoreset, double ratio, double distortion,
                            List<Double> ci95) {
    }

    public record DeltaSweep(double delta, int clustering, int deficientClusters, double costFull,
                             double costCoreset, double distortion) {
    }

    /** @param relativeCost {@code costFullOfCoresetSolution / costFullOfFullSolution - 1}: optimisation quality */
    public record Downstream(String solver, int restarts, double costFullOfCoresetSolution,
                             double costFullOfFullSolution, double relativeCost, double solveOnCoresetMs,
                             double solveOnFullMs) {
    }

    /**
     * Cost of validating this run, excluding the shared full-data candidates (see the suite's {@code shared.json}).
     *
     * @param distanceEvals  point-to-center distance evaluations spent on full-data and coreset cost passes
     * @param fullDataPasses points read by cost passes divided by the input size (coreset passes add a fraction)
     */
    public record Validation(double wallMs, long distanceEvals, double fullDataPasses) {
    }

    public record Verdict(double distortionMax, double threshold, boolean pass, List<String> notes) {
    }

    /** This report with every timing, memory and environment field zeroed: what must be identical on a replay. */
    public ValidityReport withoutTimings() {
        Run r = new Run("", "", null, null, null, null, null, 0, null);
        Construction c = construction;
        Construction ct = new Construction(c.outputSize(), c.levels(), c.reduceOps(), c.mergeOps(),
                c.additiveOffset(), 0, 0, null, 0, null, null, 0, 0, c.distanceEvals(), c.pointsRead(),
                c.bytesSerialized(), null, c.estimatedBytes(), c.reused(), c.perLevel(), c.perNode());
        Downstream d = downstream;
        Downstream dt = new Downstream(d.solver(), d.restarts(), d.costFullOfCoresetSolution(),
                d.costFullOfFullSolution(), d.relativeCost(), 0, 0);
        return new ValidityReport(schemaVersion, r, config, dataset, ct, integrity, distortion, dt,
                new Validation(0, validation.distanceEvals(), validation.fullDataPasses()), verdict);
    }
}
