package io.github.c0rwin.coreset.grpc.resilience;

import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.SnapshotChunk;
import io.github.c0rwin.coreset.grpc.proto.SnapshotStreamRequest;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * Client interceptor that erases packets of {@code SnapshotStream} responses per a {@link ChannelModel}; headers
 * always pass, as the message model assumes them reliable. Each stream draws its indicators from a seed derived from
 * this interceptor's seed and the request's (owner, replica, stream) plus a per-key attempt counter, so losses are
 * reproducible per node and independent of the sender's sampling seed and of the payloads (def:oblivious). Every
 * dropped packet is replaced by a {@code request(1)}, so stub flow control never stalls.
 */
public final class LossyTransport implements ClientInterceptor {

    private static final String METHOD = CoresetNodeServiceGrpc.getSnapshotStreamMethod().getFullMethodName();

    private final ChannelModel model;
    private final long seed;
    private final ConcurrentHashMap<Long, AtomicLong> attempts = new ConcurrentHashMap<>();
    private final LongAdder delivered = new LongAdder();
    private final LongAdder dropped = new LongAdder();

    public LossyTransport(ChannelModel model, long seed) {
        this.model = Objects.requireNonNull(model, "model");
        this.seed = seed;
    }

    public ChannelModel model() {
        return model;
    }

    /** Packets delivered so far over all streams. */
    public long delivered() {
        return delivered.sum();
    }

    /** Packets erased so far over all streams. */
    public long dropped() {
        return dropped.sum();
    }

    /** Fraction of packets erased so far, or 0 before any packet. */
    public double observedLossRate() {
        long d = dropped.sum();
        long total = d + delivered.sum();
        return total == 0 ? 0.0 : (double) d / total;
    }

    @Override
    public <ReqT, RespT> ClientCall<ReqT, RespT> interceptCall(MethodDescriptor<ReqT, RespT> method,
                                                               CallOptions options, Channel next) {
        ClientCall<ReqT, RespT> call = next.newCall(method, options);
        if (!METHOD.equals(method.getFullMethodName())) {
            return call;
        }
        return new ForwardingClientCall.SimpleForwardingClientCall<>(call) {
            private volatile ErasureChannel channel; // set by the single request, before any response

            @Override
            public void start(Listener<RespT> listener, Metadata headers) {
                ClientCall<ReqT, RespT> self = this;
                super.start(new ForwardingClientCallListener.SimpleForwardingClientCallListener<>(listener) {
                    @Override
                    public void onMessage(RespT message) {
                        ErasureChannel c = channel;
                        if (c != null && ((SnapshotChunk) message).hasPacket()) {
                            if (!c.deliver()) {
                                dropped.increment();
                                self.request(1);
                                return;
                            }
                            delivered.increment();
                        }
                        super.onMessage(message);
                    }
                }, headers);
            }

            @Override
            public void sendMessage(ReqT message) {
                channel = model.open(streamSeed((SnapshotStreamRequest) message));
                super.sendMessage(message);
            }
        };
    }

    private long streamSeed(SnapshotStreamRequest r) {
        long key = mix64(r.getOwner() * 0x9E3779B97F4A7C15L ^ mix64(r.getStream() + 1) ^ (r.getReplica() ? 7 : 0));
        long attempt = attempts.computeIfAbsent(key, k -> new AtomicLong()).getAndIncrement();
        return mix64(seed ^ key ^ mix64(attempt + 0x632BE59BD9B4E019L));
    }

    private static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }
}
