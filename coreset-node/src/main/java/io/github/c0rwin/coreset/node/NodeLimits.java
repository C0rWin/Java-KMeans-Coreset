package io.github.c0rwin.coreset.node;

import java.time.Duration;
import java.util.Objects;

/**
 * Resource bounds a {@link CoresetNode} enforces before it allocates anything on behalf of a caller.
 *
 * @param maxSeqWindow           a producer's seq must be below its committed floor plus this window; bounds the
 *                               dedupe state per producer and must exceed the sender's in-flight batches
 * @param maxProducersPerSession distinct producer ids per session (each keeps a small dedupe state)
 * @param casSnapshotWait        how long a {@link TreeMode#CAS} snapshot may hold off new carries while waiting for
 *                               the ones in flight, before returning a read flagged inconsistent
 */
public record NodeLimits(int maxSeqWindow, int maxProducersPerSession, Duration casSnapshotWait) {

    public NodeLimits {
        Objects.requireNonNull(casSnapshotWait, "casSnapshotWait");
        positive("maxSeqWindow", maxSeqWindow);
        positive("maxProducersPerSession", maxProducersPerSession);
        if (casSnapshotWait.isNegative()) {
            throw new IllegalArgumentException("casSnapshotWait must be >= 0: " + casSnapshotWait);
        }
    }

    public static NodeLimits defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        Builder b = new Builder();
        b.maxSeqWindow = maxSeqWindow;
        b.maxProducersPerSession = maxProducersPerSession;
        b.casSnapshotWait = casSnapshotWait;
        return b;
    }

    private static void positive(String name, long value) {
        if (value < 1) {
            throw new IllegalArgumentException(name + " must be >= 1: " + value);
        }
    }

    /** Starts from the defaults. */
    public static final class Builder {
        private int maxSeqWindow = 4096;
        private int maxProducersPerSession = 1024;
        private Duration casSnapshotWait = Duration.ofSeconds(1);

        private Builder() {
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

        public NodeLimits build() {
            return new NodeLimits(maxSeqWindow, maxProducersPerSession, casSnapshotWait);
        }
    }
}
