package io.github.c0rwin.coreset.node;

/** Lifecycle of a session. */
public enum SessionState {
    ACTIVE,
    /** A worker failed; the session accepts nothing more and reports the cause. */
    FAILED,
    CLOSED
}
