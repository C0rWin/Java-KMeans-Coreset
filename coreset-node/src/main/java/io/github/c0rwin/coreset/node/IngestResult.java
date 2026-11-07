package io.github.c0rwin.coreset.node;

import java.util.Objects;

/**
 * Outcome of {@link CoresetNode#ingest}.
 *
 * @param status         what happened to the batch
 * @param reason         human-readable detail for {@link Status#BUSY} and {@link Status#FAILED}, empty otherwise
 * @param acceptedPoints points enqueued by this call (non-zero only for {@link Status#ACCEPTED})
 */
public record IngestResult(Status status, String reason, int acceptedPoints) {

    public enum Status {
        /** Enqueued for exactly-once application. */
        ACCEPTED,
        /** This seq was already accepted; nothing was enqueued. Safe outcome of a retry. */
        DUPLICATE,
        /** Not enqueued because the worker queue is full (or the same seq is mid-flight); retry the same seq later. */
        BUSY,
        /** Rejected permanently: unknown/closed/failed session or invalid batch. */
        FAILED
    }

    private static final IngestResult DUPLICATE = new IngestResult(Status.DUPLICATE, "", 0);

    public IngestResult {
        Objects.requireNonNull(status, "status");
        Objects.requireNonNull(reason, "reason");
    }

    public static IngestResult accepted(int points) {
        return new IngestResult(Status.ACCEPTED, "", points);
    }

    public static IngestResult duplicate() {
        return DUPLICATE;
    }

    public static IngestResult busy(String reason) {
        return new IngestResult(Status.BUSY, reason, 0);
    }

    public static IngestResult failed(String reason) {
        return new IngestResult(Status.FAILED, reason, 0);
    }
}
