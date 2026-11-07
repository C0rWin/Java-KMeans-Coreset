package io.github.c0rwin.coreset.grpc;

import java.time.Duration;
import java.util.Objects;

/**
 * Transport settings of a {@link CoresetNodeServer}.
 *
 * @param bindHost                        address to listen on; loopback by default, {@code 0.0.0.0} exposes the node
 * @param port                            0 picks an ephemeral port
 * @param maxInboundMessageBytes          largest request message accepted
 * @param security                        TLS and authentication
 * @param handlerThreads                  gRPC callback threads (ingest and quick unary calls)
 * @param blockingThreads                 threads for calls that wait or scan (snapshot, cost, create/close session)
 * @param blockingQueue                   blocking calls queued beyond {@code blockingThreads}; more get
 *                                        RESOURCE_EXHAUSTED
 * @param maxConcurrentCallsPerConnection concurrent calls (streams) per client connection
 * @param maxConnectionIdle               idle connections are closed after this
 * @param maxConnectionAge                connections are recycled after this (clients reconnect)
 * @param maxConnectionAgeGrace           time in-flight calls get after a connection reached its age
 */
public record NodeServerOptions(String bindHost, int port, int maxInboundMessageBytes, ServerSecurity security,
                                int handlerThreads, int blockingThreads, int blockingQueue,
                                int maxConcurrentCallsPerConnection, Duration maxConnectionIdle,
                                Duration maxConnectionAge, Duration maxConnectionAgeGrace) {

    /** Server keepalive ping interval and ack timeout; clients may ping no more often than {@link #PERMIT_KEEPALIVE}. */
    public static final Duration KEEPALIVE_TIME = Duration.ofSeconds(60);
    public static final Duration KEEPALIVE_TIMEOUT = Duration.ofSeconds(20);
    public static final Duration PERMIT_KEEPALIVE = Duration.ofSeconds(20);

    public NodeServerOptions {
        Objects.requireNonNull(bindHost, "bindHost");
        Objects.requireNonNull(security, "security");
        Objects.requireNonNull(maxConnectionIdle, "maxConnectionIdle");
        Objects.requireNonNull(maxConnectionAge, "maxConnectionAge");
        Objects.requireNonNull(maxConnectionAgeGrace, "maxConnectionAgeGrace");
        if (port < 0 || port > 65535) {
            throw new IllegalArgumentException("port out of range: " + port);
        }
        if (maxInboundMessageBytes < 1 || handlerThreads < 1 || blockingThreads < 1 || blockingQueue < 1
                || maxConcurrentCallsPerConnection < 1) {
            throw new IllegalArgumentException("message size, thread, queue and call limits must be >= 1");
        }
        if (maxConnectionIdle.isNegative() || maxConnectionIdle.isZero() || maxConnectionAge.isNegative()
                || maxConnectionAge.isZero() || maxConnectionAgeGrace.isNegative()) {
            throw new IllegalArgumentException("connection idle and age must be > 0, grace >= 0");
        }
    }

    public static NodeServerOptions defaults() {
        return builder().build();
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Starts from the defaults: loopback, port 7070, plaintext, 16 MiB requests. */
    public static final class Builder {
        private String bindHost = "127.0.0.1";
        private int port = 7070;
        private int maxInboundMessageBytes = CoresetNodeServer.DEFAULT_MAX_INBOUND_MESSAGE_BYTES;
        private ServerSecurity security = ServerSecurity.none();
        private int handlerThreads = Math.max(4, 2 * Runtime.getRuntime().availableProcessors());
        private int blockingThreads = Math.max(4, Runtime.getRuntime().availableProcessors());
        private int blockingQueue = 256;
        private int maxConcurrentCallsPerConnection = 64;
        private Duration maxConnectionIdle = Duration.ofMinutes(30);
        private Duration maxConnectionAge = Duration.ofHours(2);
        // Longer than the coordinator's default 10-minute stream deadline, so a recycled connection drains.
        private Duration maxConnectionAgeGrace = Duration.ofMinutes(15);

        private Builder() {
        }

        public Builder bindHost(String v) {
            bindHost = v;
            return this;
        }

        public Builder port(int v) {
            port = v;
            return this;
        }

        public Builder maxInboundMessageBytes(int v) {
            maxInboundMessageBytes = v;
            return this;
        }

        public Builder security(ServerSecurity v) {
            security = v;
            return this;
        }

        public Builder handlerThreads(int v) {
            handlerThreads = v;
            return this;
        }

        public Builder blockingThreads(int v) {
            blockingThreads = v;
            return this;
        }

        public Builder blockingQueue(int v) {
            blockingQueue = v;
            return this;
        }

        public Builder maxConcurrentCallsPerConnection(int v) {
            maxConcurrentCallsPerConnection = v;
            return this;
        }

        public Builder maxConnectionIdle(Duration v) {
            maxConnectionIdle = v;
            return this;
        }

        public Builder maxConnectionAge(Duration v) {
            maxConnectionAge = v;
            return this;
        }

        public Builder maxConnectionAgeGrace(Duration v) {
            maxConnectionAgeGrace = v;
            return this;
        }

        public NodeServerOptions build() {
            return new NodeServerOptions(bindHost, port, maxInboundMessageBytes, security, handlerThreads,
                    blockingThreads, blockingQueue, maxConcurrentCallsPerConnection, maxConnectionIdle,
                    maxConnectionAge, maxConnectionAgeGrace);
        }
    }
}
