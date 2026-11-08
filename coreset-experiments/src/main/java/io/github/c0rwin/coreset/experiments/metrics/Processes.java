package io.github.c0rwin.coreset.experiments.metrics;

import java.util.List;
import java.util.concurrent.locks.LockSupport;

/** Process-tree enumeration that survives transient failures of the JDK's native call. */
public final class Processes {

    private static final int ATTEMPTS = 4;

    private Processes() {
    }

    /**
     * Descendants of this JVM, retried with a short backoff: on macOS the enumeration fails with "Cannot allocate
     * memory" when the process table grows between its size probe and its read, which happens under process churn.
     *
     * @throws RuntimeException the last failure, if every attempt failed
     */
    public static List<ProcessHandle> descendants() {
        RuntimeException last = null;
        for (int attempt = 0; attempt < ATTEMPTS; attempt++) {
            try {
                return ProcessHandle.current().descendants().toList();
            } catch (RuntimeException e) {
                last = e;
                LockSupport.parkNanos(5_000_000L << attempt);
            }
        }
        throw last;
    }
}
