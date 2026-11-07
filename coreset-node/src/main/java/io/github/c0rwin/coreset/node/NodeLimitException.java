package io.github.c0rwin.coreset.node;

/** A request would exceed the node's capacity or a per-request work budget (see {@link NodeLimits}). */
public class NodeLimitException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public NodeLimitException(String message) {
        super(message);
    }

    public NodeLimitException(String message, Throwable cause) {
        super(message, cause);
    }
}
