package io.github.c0rwin.coreset.node;

import java.time.Duration;
import java.util.Map;
import java.util.Objects;

/**
 * Resource bounds a {@link CoresetNode} (and its transport adapter) enforces before it allocates anything on behalf
 * of a caller. Shape violations are {@link IllegalArgumentException}s; exhausted node capacity or an over-budget
 * request is a {@link NodeLimitException}.
 *
 * @param maxDim                 largest vector dimension; dense working buffers of the algorithms are O(dim)
 * @param maxPointsPerBatch      points per ingested batch, and per raw-point cost shard
 * @param maxSessions            concurrently open sessions
 * @param maxTotalWorkers        worker threads summed over all open sessions
 * @param maxQueueCapacity       per-worker queue capacity; the ring is preallocated
 * @param maxSessionIdLength     session id length; ids are also restricted to {@code [A-Za-z0-9._:-]}
 * @param maxParams              entries in a session's algorithm parameter map
 * @param maxParamLength         length of the algorithm id and of every parameter key and value
 * @param maxCenters             centers per cost evaluation
 * @param maxCostWork            points times centers per cost evaluation
 * @param maxAwait               longest {@code awaitApplied} wait a caller may request
 * @param maxSnapshotPoints      points in a session snapshot; larger snapshots are refused before the union is built
 * @param maxSeqWindow           a producer's seq must be below its committed floor plus this window; bounds the
 *                               dedupe state per producer and must exceed the sender's in-flight batches
 * @param maxProducersPerSession distinct producer ids per session (each keeps a small dedupe state)
 * @param casSnapshotWait        how long a {@link TreeMode#CAS} snapshot may hold off new carries while waiting for
 *                               the ones in flight, before returning a read flagged inconsistent
 * @param sessionIdleTimeout     a session with no ingest progress and no reads for this long is closed; zero keeps
 *                               sessions until they are closed explicitly
 */
public record NodeLimits(int maxDim, int maxPointsPerBatch, int maxSessions, int maxTotalWorkers,
                         int maxQueueCapacity, int maxSessionIdLength, int maxParams, int maxParamLength,
                         int maxCenters, long maxCostWork, Duration maxAwait, int maxSnapshotPoints,
                         int maxSeqWindow, int maxProducersPerSession, Duration casSnapshotWait,
                         Duration sessionIdleTimeout) {

    public NodeLimits {
        Objects.requireNonNull(maxAwait, "maxAwait");
        Objects.requireNonNull(casSnapshotWait, "casSnapshotWait");
        Objects.requireNonNull(sessionIdleTimeout, "sessionIdleTimeout");
        positive("maxDim", maxDim);
        positive("maxPointsPerBatch", maxPointsPerBatch);
        positive("maxSessions", maxSessions);
        positive("maxTotalWorkers", maxTotalWorkers);
        if (maxQueueCapacity < 2 || Integer.bitCount(maxQueueCapacity) != 1) {
            throw new IllegalArgumentException("maxQueueCapacity must be a power of two >= 2: " + maxQueueCapacity);
        }
        positive("maxSessionIdLength", maxSessionIdLength);
        positive("maxParams", maxParams);
        positive("maxParamLength", maxParamLength);
        positive("maxCenters", maxCenters);
        positive("maxCostWork", maxCostWork);
        positive("maxSnapshotPoints", maxSnapshotPoints);
        positive("maxSeqWindow", maxSeqWindow);
        positive("maxProducersPerSession", maxProducersPerSession);
        if (maxAwait.isNegative() || casSnapshotWait.isNegative() || sessionIdleTimeout.isNegative()) {
            throw new IllegalArgumentException("maxAwait, casSnapshotWait and sessionIdleTimeout must be >= 0");
        }
    }

    public static NodeLimits defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder().maxDim(maxDim).maxPointsPerBatch(maxPointsPerBatch).maxSessions(maxSessions)
                .maxTotalWorkers(maxTotalWorkers).maxQueueCapacity(maxQueueCapacity)
                .maxSessionIdLength(maxSessionIdLength).maxParams(maxParams).maxParamLength(maxParamLength)
                .maxCenters(maxCenters).maxCostWork(maxCostWork).maxAwait(maxAwait)
                .maxSnapshotPoints(maxSnapshotPoints).maxSeqWindow(maxSeqWindow)
                .maxProducersPerSession(maxProducersPerSession).casSnapshotWait(casSnapshotWait)
                .sessionIdleTimeout(sessionIdleTimeout);
    }

    /** @throws IllegalArgumentException if {@code id} is empty, too long or has characters outside the id alphabet */
    public void checkSessionId(String id) {
        if (id == null || id.isEmpty()) {
            throw new IllegalArgumentException("session id must not be empty");
        }
        if (id.length() > maxSessionIdLength) {
            throw new IllegalArgumentException("session id longer than " + maxSessionIdLength + " chars");
        }
        for (int i = 0, n = id.length(); i < n; i++) {
            char c = id.charAt(i);
            boolean ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
                    || c == '.' || c == '_' || c == ':' || c == '-';
            if (!ok) {
                throw new IllegalArgumentException("session id may only contain [A-Za-z0-9._:-]");
            }
        }
    }

    /** Checks the algorithm id and parameter map sizes; run before the map is copied. */
    public void checkAlgorithm(String algorithmId, Map<String, String> params) {
        if (algorithmId.length() > maxParamLength) {
            throw new IllegalArgumentException("algorithm id longer than " + maxParamLength + " chars");
        }
        if (params.size() > maxParams) {
            throw new IllegalArgumentException(params.size() + " algorithm params exceed the limit of " + maxParams);
        }
        for (Map.Entry<String, String> e : params.entrySet()) {
            if (e.getKey().length() > maxParamLength || e.getValue().length() > maxParamLength) {
                throw new IllegalArgumentException("algorithm param key or value longer than " + maxParamLength
                        + " chars");
            }
        }
    }

    /** @throws IllegalArgumentException if a batch or cost shard of {@code points} points is too large */
    public void checkPoints(long points) {
        if (points > maxPointsPerBatch) {
            throw new IllegalArgumentException(points + " points exceed the per-batch limit of " + maxPointsPerBatch);
        }
    }

    /** @throws IllegalArgumentException if a vector of dimension {@code dim} is outside {@code [0, maxDim]} */
    public void checkDim(long dim) {
        if (dim < 0 || dim > maxDim) {
            throw new IllegalArgumentException("dim " + dim + " exceeds the node limit of " + maxDim);
        }
    }

    /**
     * @throws IllegalArgumentException if there are too many centers
     * @throws NodeLimitException       if evaluating them on {@code points} points exceeds the work budget
     */
    public void checkCost(long points, long centers) {
        if (centers > maxCenters) {
            throw new IllegalArgumentException(centers + " centers exceed the limit of " + maxCenters);
        }
        if (points * centers > maxCostWork) { // both factors are bounded by int limits: no overflow
            throw new NodeLimitException(points + " points x " + centers + " centers exceed the cost work budget of "
                    + maxCostWork);
        }
    }

    private static void positive(String name, long value) {
        if (value < 1) {
            throw new IllegalArgumentException(name + " must be >= 1: " + value);
        }
    }

    /** Starts from the defaults. */
    public static final class Builder {
        private int maxDim = 1 << 24;
        private int maxPointsPerBatch = 1 << 18;
        private int maxSessions = 64;
        private int maxTotalWorkers = 4096;
        private int maxQueueCapacity = 4096;
        private int maxSessionIdLength = 128;
        private int maxParams = 32;
        private int maxParamLength = 256;
        private int maxCenters = 100_000;
        private long maxCostWork = 1L << 31;
        private Duration maxAwait = Duration.ofMinutes(10);
        private int maxSnapshotPoints = 1 << 22;
        private int maxSeqWindow = 4096;
        private int maxProducersPerSession = 1024;
        private Duration casSnapshotWait = Duration.ofSeconds(1);
        private Duration sessionIdleTimeout = Duration.ofHours(1);

        private Builder() {
        }

        public Builder maxDim(int v) {
            maxDim = v;
            return this;
        }

        public Builder maxPointsPerBatch(int v) {
            maxPointsPerBatch = v;
            return this;
        }

        public Builder maxSessions(int v) {
            maxSessions = v;
            return this;
        }

        public Builder maxTotalWorkers(int v) {
            maxTotalWorkers = v;
            return this;
        }

        public Builder maxQueueCapacity(int v) {
            maxQueueCapacity = v;
            return this;
        }

        public Builder maxSessionIdLength(int v) {
            maxSessionIdLength = v;
            return this;
        }

        public Builder maxParams(int v) {
            maxParams = v;
            return this;
        }

        public Builder maxParamLength(int v) {
            maxParamLength = v;
            return this;
        }

        public Builder maxCenters(int v) {
            maxCenters = v;
            return this;
        }

        public Builder maxCostWork(long v) {
            maxCostWork = v;
            return this;
        }

        public Builder maxAwait(Duration v) {
            maxAwait = v;
            return this;
        }

        public Builder maxSnapshotPoints(int v) {
            maxSnapshotPoints = v;
            return this;
        }

        public Builder maxSeqWindow(int v) {
            maxSeqWindow = v;
            return this;
        }

        public Builder maxProducersPerSession(int v) {
            maxProducersPerSession = v;
            return this;
        }

        public Builder casSnapshotWait(Duration v) {
            casSnapshotWait = v;
            return this;
        }

        public Builder sessionIdleTimeout(Duration v) {
            sessionIdleTimeout = v;
            return this;
        }

        public NodeLimits build() {
            return new NodeLimits(maxDim, maxPointsPerBatch, maxSessions, maxTotalWorkers, maxQueueCapacity,
                    maxSessionIdLength, maxParams, maxParamLength, maxCenters, maxCostWork, maxAwait,
                    maxSnapshotPoints, maxSeqWindow, maxProducersPerSession, casSnapshotWait, sessionIdleTimeout);
        }
    }
}
