package io.github.c0rwin.coreset.node;

/** Names of the metrics emitted by coreset-node, in addition to the coreset-core ones. */
public final class NodeMetricNames {

    private NodeMetricNames() {
    }

    /** Distribution (ns): wall time of one {@code ingest} call, all outcomes. */
    public static final String INGEST_LATENCY = "node.ingest.latency_ns";
    /** Counter: batches accepted. */
    public static final String INGEST_ACCEPTED = "node.ingest.accepted";
    /** Counter: points in accepted batches. */
    public static final String INGEST_ACCEPTED_POINTS = "node.ingest.accepted_points";
    /** Counter: retried seqs acknowledged as duplicates. */
    public static final String INGEST_DUPLICATES = "node.ingest.duplicates";
    /** Counter: BUSY rejections (full worker queue or seq mid-flight). */
    public static final String INGEST_BUSY = "node.ingest.busy";
    /** Counter: FAILED rejections. */
    public static final String INGEST_FAILED = "node.ingest.failed";
    /** Distribution: target worker queue depth right after an accepted enqueue. */
    public static final String QUEUE_DEPTH = "node.queue.depth";
    /** Span around compressing one ingested batch on a worker. */
    public static final String BATCH_COMPRESS = "node.batch.compress";
    /** Distribution: points in a session snapshot. */
    public static final String SNAPSHOT_POINTS = "node.snapshot.points";
    /** Distribution: estimated heap bytes of a session snapshot. */
    public static final String SNAPSHOT_BYTES = "node.snapshot.estimated_bytes";
    /** Distribution: optimistic read attempts per CAS-mode snapshot. */
    public static final String SNAPSHOT_CAS_ATTEMPTS = "node.snapshot.cas_attempts";
    /** Counter: CAS-mode snapshots returned without a validated quiescent read. */
    public static final String SNAPSHOT_INCONSISTENT = "node.snapshot.inconsistent";
    /** Counter: CAS tree slot races (a slot emptied by another worker between a failed place and the take). */
    public static final String CAS_RETRIES = "node.cas.retries";
    /** Distribution: input weight held by in-flight CAS carries, sampled at each snapshot. */
    public static final String CAS_IN_FLIGHT_WEIGHT = "node.cas.in_flight_weight";
    /** Counter: sessions created. */
    public static final String SESSIONS_CREATED = "node.sessions.created";
    /** Counter: sessions closed. */
    public static final String SESSIONS_CLOSED = "node.sessions.closed";
    /** Counter: sessions closed by the idle reaper. */
    public static final String SESSIONS_EXPIRED = "node.sessions.expired";
    /** Counter: sessions that failed. */
    public static final String SESSION_FAILURES = "node.sessions.failed";

    /** Distribution: merge-reduce tree height (levels) observed by worker {@code worker} after each insertion. */
    public static String workerTreeDepth(int worker) {
        return "node.worker." + worker + ".tree_depth";
    }
}
