package io.github.c0rwin.coreset.grpc;

/** An ingest stream could not deliver every batch (permanent FAILED ack or exhausted transport retries). */
public class IngestFailedException extends Exception {

    private static final long serialVersionUID = 1L;

    public IngestFailedException(String message, Throwable cause) {
        super(message, cause);
    }
}
