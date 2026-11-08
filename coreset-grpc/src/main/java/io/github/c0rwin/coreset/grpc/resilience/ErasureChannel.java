package io.github.c0rwin.coreset.grpc.resilience;

/**
 * One stream's delivery indicators, drawn packet by packet from a {@link ChannelModel}. Not thread-safe: one
 * instance per stream, whose packets arrive serially.
 */
@FunctionalInterface
public interface ErasureChannel {

    /** Whether the next packet is delivered. */
    boolean deliver();
}
