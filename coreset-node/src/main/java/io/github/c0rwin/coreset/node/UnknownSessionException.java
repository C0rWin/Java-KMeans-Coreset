package io.github.c0rwin.coreset.node;

import java.util.NoSuchElementException;

/** No session with the given id exists on this node (never created, or already closed). */
public class UnknownSessionException extends NoSuchElementException {

    private static final long serialVersionUID = 1L;

    public UnknownSessionException(String sessionId) {
        super("unknown session: " + Messages.bounded(sessionId));
    }
}
