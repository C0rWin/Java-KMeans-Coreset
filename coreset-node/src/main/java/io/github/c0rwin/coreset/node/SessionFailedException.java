package io.github.c0rwin.coreset.node;

/** The session stopped because a worker failed; {@link #getCause()} is the original failure. */
public class SessionFailedException extends IllegalStateException {

    private static final long serialVersionUID = 1L;

    public SessionFailedException(String sessionId, Throwable cause) {
        super("session " + Messages.bounded(sessionId) + " failed: " + Messages.bounded(cause), cause);
    }
}
