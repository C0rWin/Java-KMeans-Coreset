package io.github.c0rwin.coreset.node;

/** Bounds caller-influenced text before it is echoed in results, statuses or thread names. */
public final class Messages {

    /** Longest text echoed back to a caller. */
    public static final int MAX_ECHO = 256;

    private Messages() {
    }

    /** {@code s} unchanged if short enough, otherwise its prefix and the original length. */
    public static String bounded(String s) {
        if (s == null || s.length() <= MAX_ECHO) {
            return s;
        }
        return s.substring(0, MAX_ECHO) + "...(" + s.length() + " chars)";
    }

    /** {@link #bounded} of {@code t.toString()}. */
    public static String bounded(Throwable t) {
        return bounded(String.valueOf(t));
    }
}
