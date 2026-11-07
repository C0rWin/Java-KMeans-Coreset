package io.github.c0rwin.coreset.node;

/** Thrown by an {@link AlgorithmResolver} for an algorithm id it does not provide. */
public class UnknownAlgorithmException extends IllegalArgumentException {

    private static final long serialVersionUID = 1L;

    private final String algorithmId;

    public UnknownAlgorithmException(String algorithmId) {
        super("unknown coreset algorithm: " + algorithmId);
        this.algorithmId = algorithmId;
    }

    public String algorithmId() {
        return algorithmId;
    }
}
