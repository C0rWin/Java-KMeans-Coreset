package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;
import java.util.Objects;

/**
 * Point-in-time view of a session.
 *
 * <p>{@code accepted*} count every batch acknowledged {@code ACCEPTED} by an {@code ingest} that returned before the
 * snapshot was taken (concurrent ingests may be partially visible); {@code applied*} describe exactly the batches
 * summarised by {@link #coreset()}. Weights are input weights (sum of batch point weights), not coreset weights.
 *
 * @param consistent false only in {@link TreeMode#CAS} when no quiescent instant was observed, in which case the
 *                   coreset and applied counters may miss or double-count batches that were mid-merge
 * @param failure    cause when {@code state == FAILED}, otherwise empty
 */
public record NodeSnapshot(Coreset coreset, SessionState state, String failure, long acceptedBatches,
                           long appliedBatches, double acceptedWeight, double appliedWeight, boolean consistent) {

    public NodeSnapshot {
        Objects.requireNonNull(coreset, "coreset");
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(failure, "failure");
    }

    public long pendingBatches() {
        return Math.max(0L, acceptedBatches - appliedBatches);
    }

    /** Accepted input weight not yet reflected in {@link #coreset()}; 0 when no batch is pending. */
    public double pendingWeight() {
        return pendingBatches() == 0 ? 0.0 : Math.max(0.0, acceptedWeight - appliedWeight);
    }

    /** True when the coreset reflects every accepted batch of an active session. */
    public boolean complete() {
        return state == SessionState.ACTIVE && consistent && appliedBatches == acceptedBatches;
    }
}
