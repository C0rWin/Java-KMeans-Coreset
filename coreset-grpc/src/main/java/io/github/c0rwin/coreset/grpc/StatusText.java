package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.node.Messages;
import io.grpc.Status;

/** One-line form of a gRPC failure for reports: code and description, without the cause's stack trace. */
public final class StatusText {

    private StatusText() {
    }

    public static String of(Throwable t) {
        Status s = Status.fromThrowable(t);
        return Messages.bounded(s.getDescription() == null ? s.getCode().toString()
                : s.getCode() + " " + s.getDescription());
    }
}
