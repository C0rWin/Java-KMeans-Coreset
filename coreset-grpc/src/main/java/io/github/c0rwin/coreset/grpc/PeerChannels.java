package io.github.c0rwin.coreset.grpc;

import io.grpc.Channel;
import io.grpc.ChannelCredentials;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import java.io.IOException;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/** Channels a node opens to its peers to store replicas of its message ({@code Replicate}). */
public interface PeerChannels extends AutoCloseable {

    /** @throws IllegalStateException if this node cannot reach peers */
    Channel channel(String target);

    @Override
    void close();

    /** No peer transport: {@code Replicate} fails with FAILED_PRECONDITION. */
    static PeerChannels none() {
        return of(target -> {
            throw new IllegalStateException("this node has no peer transport configured");
        });
    }

    /** Channels owned by the caller (in-process tests); closing does nothing. */
    static PeerChannels of(Function<String, ? extends Channel> channels) {
        Objects.requireNonNull(channels, "channels");
        return new PeerChannels() {
            @Override
            public Channel channel(String target) {
                return channels.apply(target);
            }

            @Override
            public void close() {
            }
        };
    }

    /** Netty channels, one per target, opened on first use with {@code security} and closed with this object. */
    static PeerChannels netty(ClientSecurity security, int maxMessageBytes) throws IOException {
        ChannelCredentials credentials = security.channelCredentials();
        ConcurrentHashMap<String, ManagedChannel> open = new ConcurrentHashMap<>();
        return new PeerChannels() {
            @Override
            public Channel channel(String target) {
                return open.computeIfAbsent(target, t -> {
                    NettyChannelBuilder b = NettyChannelBuilder.forTarget(t, credentials)
                            .maxInboundMessageSize(maxMessageBytes);
                    if (security.authToken() != null) {
                        b.intercept(TokenAuth.client(security.authToken()));
                    }
                    return b.build();
                });
            }

            @Override
            public void close() {
                for (ManagedChannel ch : open.values()) {
                    ch.shutdown();
                }
                for (ManagedChannel ch : open.values()) {
                    try {
                        if (!ch.awaitTermination(5, TimeUnit.SECONDS)) {
                            ch.shutdownNow();
                        }
                    } catch (InterruptedException e) {
                        ch.shutdownNow();
                        Thread.currentThread().interrupt();
                    }
                }
            }
        };
    }
}
