package io.github.c0rwin.coreset.experiments.lossy;

import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingServerCall;
import io.grpc.ForwardingServerCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;

/**
 * Message-loss slots for gRPC: a client interceptor drops outgoing request messages and a server interceptor drops
 * incoming requests and outgoing responses of the selected methods, as decided by a {@link ChannelModel}. A dropped
 * message simply never reaches the other side (the call stays open), which is what a datagram-like lossy link does to
 * a protocol that relies on per-message acknowledgements. Counters report what was dropped.
 */
public final class LossyInterceptors {

    private final ChannelModel model;
    private final Set<String> methods;
    private final LongAdder seen = new LongAdder();
    private final LongAdder dropped = new LongAdder();

    /** @param methods bare method names (e.g. {@code Ingest}); empty means every method */
    public LossyInterceptors(ChannelModel model, Set<String> methods) {
        this.model = model;
        this.methods = Set.copyOf(methods);
    }

    public long seen() {
        return seen.sum();
    }

    public long dropped() {
        return dropped.sum();
    }

    private boolean applies(MethodDescriptor<?, ?> m) {
        return methods.isEmpty() || methods.contains(m.getBareMethodName());
    }

    private boolean lose() {
        seen.increment();
        if (model.drop()) {
            dropped.increment();
            return true;
        }
        return false;
    }

    /** Drops request messages leaving the client. */
    public ClientInterceptor client() {
        return new ClientInterceptor() {
            @Override
            public <Q, R> ClientCall<Q, R> interceptCall(MethodDescriptor<Q, R> method, CallOptions options,
                                                         Channel next) {
                ClientCall<Q, R> call = next.newCall(method, options);
                if (!applies(method)) {
                    return call;
                }
                return new ForwardingClientCall.SimpleForwardingClientCall<>(call) {
                    @Override
                    public void sendMessage(Q message) {
                        if (!lose()) {
                            super.sendMessage(message);
                        }
                    }
                };
            }
        };
    }

    /** Drops requests arriving at and responses leaving a server (for in-process or embedded nodes). */
    public ServerInterceptor server() {
        return new ServerInterceptor() {
            @Override
            public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
                                                               ServerCallHandler<Q, R> next) {
                if (!applies(call.getMethodDescriptor())) {
                    return next.startCall(call, headers);
                }
                ServerCall<Q, R> lossyOut = new ForwardingServerCall.SimpleForwardingServerCall<>(call) {
                    @Override
                    public void sendMessage(R message) {
                        if (!lose()) {
                            super.sendMessage(message);
                        }
                    }
                };
                return new ForwardingServerCallListener.SimpleForwardingServerCallListener<>(
                        next.startCall(lossyOut, headers)) {
                    @Override
                    public void onMessage(Q message) {
                        if (lose()) {
                            call.request(1); // keep flow control moving past the lost message
                        } else {
                            super.onMessage(message);
                        }
                    }
                };
            }
        };
    }
}
