package io.github.c0rwin.coreset.grpc;

import java.time.Duration;
import java.util.Objects;

/**
 * Client-side knobs of {@link CoresetCoordinator}.
 *
 * @param maxInFlightPerNode   unacknowledged batches allowed per node stream; {@code submit} blocks beyond it
 * @param rpcDeadline          deadline of every unary call (added to any snapshot await time)
 * @param streamDeadline       lifetime of one ingest stream; on expiry the stream is reopened and unacknowledged
 *                             batches are resent with their original seqs
 * @param initialBackoff       first delay before resending a BUSY batch or reopening a broken stream; doubles per
 *                             consecutive attempt
 * @param maxBackoff           cap of the backoff
 * @param maxTransportFailures consecutive stream failures without any ack before the ingest is failed
 * @param maxMessageBytes      message limit; inbound it bounds snapshot responses, outbound the batches (which the
 *                             node accepts only up to its own inbound limit)
 */
public record CoordinatorOptions(int maxInFlightPerNode, Duration rpcDeadline, Duration streamDeadline,
                                 Duration initialBackoff, Duration maxBackoff, int maxTransportFailures,
                                 int maxMessageBytes) {

    /** Snapshots of large sessions exceed gRPC's 4 MiB default; the node caps them by point count. */
    public static final int DEFAULT_MAX_MESSAGE_BYTES = 256 * 1024 * 1024;

    public CoordinatorOptions {
        Objects.requireNonNull(rpcDeadline, "rpcDeadline");
        Objects.requireNonNull(streamDeadline, "streamDeadline");
        Objects.requireNonNull(initialBackoff, "initialBackoff");
        Objects.requireNonNull(maxBackoff, "maxBackoff");
        if (maxInFlightPerNode < 1) {
            throw new IllegalArgumentException("maxInFlightPerNode must be >= 1: " + maxInFlightPerNode);
        }
        if (maxTransportFailures < 1) {
            throw new IllegalArgumentException("maxTransportFailures must be >= 1: " + maxTransportFailures);
        }
        if (initialBackoff.isNegative() || maxBackoff.compareTo(initialBackoff) < 0) {
            throw new IllegalArgumentException("need 0 <= initialBackoff <= maxBackoff");
        }
    }

    public static CoordinatorOptions defaults() {
        return new CoordinatorOptions(16, Duration.ofSeconds(30), Duration.ofMinutes(10), Duration.ofMillis(1),
                Duration.ofMillis(250), 8, DEFAULT_MAX_MESSAGE_BYTES);
    }

    public CoordinatorOptions withMaxInFlightPerNode(int n) {
        return new CoordinatorOptions(n, rpcDeadline, streamDeadline, initialBackoff, maxBackoff,
                maxTransportFailures, maxMessageBytes);
    }

    public CoordinatorOptions withStreamDeadline(Duration d) {
        return new CoordinatorOptions(maxInFlightPerNode, rpcDeadline, d, initialBackoff, maxBackoff,
                maxTransportFailures, maxMessageBytes);
    }

    long backoffNanos(int attempt) {
        long base = initialBackoff.toNanos();
        long cap = maxBackoff.toNanos();
        int shift = Math.min(Math.max(attempt - 1, 0), 30);
        long d = base << shift;
        return d < 0 || d > cap ? cap : d;
    }
}
