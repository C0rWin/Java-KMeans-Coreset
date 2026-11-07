package io.github.c0rwin.coreset.grpc;

import com.google.protobuf.MessageLite;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.ForwardingClientCallListener;
import io.grpc.ForwardingServerCall;
import io.grpc.ForwardingServerCallListener;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;

/**
 * Interceptors recording per-method call latency (whole stream for streaming calls), serialized message bytes and
 * message counts in each direction, and final status codes. Names: {@code grpc.<side>.<Method>.latency_ns},
 * {@code .bytes_in}, {@code .bytes_out}, {@code .messages_in}, {@code .messages_out}, {@code .status.<CODE>}.
 * Protobuf memoizes the serialized size, so measuring it costs nothing extra when the message is then marshalled.
 */
public final class GrpcMetrics {

    private GrpcMetrics() {
    }

    public static ServerInterceptor server(MetricsSink sink) {
        return new ServerInterceptor() {
            @Override
            public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
                                                               ServerCallHandler<Q, R> next) {
                Recorder rec = new Recorder(sink, "grpc.server." + call.getMethodDescriptor().getBareMethodName());
                ServerCall<Q, R> counted = new ForwardingServerCall.SimpleForwardingServerCall<>(call) {
                    @Override
                    public void sendMessage(R message) {
                        rec.out(message);
                        super.sendMessage(message);
                    }

                    @Override
                    public void close(Status status, Metadata trailers) {
                        rec.done(status);
                        super.close(status, trailers);
                    }
                };
                return new ForwardingServerCallListener.SimpleForwardingServerCallListener<>(
                        next.startCall(counted, headers)) {
                    @Override
                    public void onMessage(Q message) {
                        rec.in(message);
                        super.onMessage(message);
                    }
                };
            }
        };
    }

    public static ClientInterceptor client(MetricsSink sink) {
        return new ClientInterceptor() {
            @Override
            public <Q, R> ClientCall<Q, R> interceptCall(MethodDescriptor<Q, R> method, CallOptions options,
                                                         Channel next) {
                Recorder rec = new Recorder(sink, "grpc.client." + method.getBareMethodName());
                return new ForwardingClientCall.SimpleForwardingClientCall<>(next.newCall(method, options)) {
                    @Override
                    public void start(Listener<R> listener, Metadata headers) {
                        super.start(new ForwardingClientCallListener.SimpleForwardingClientCallListener<>(listener) {
                            @Override
                            public void onMessage(R message) {
                                rec.in(message);
                                super.onMessage(message);
                            }

                            @Override
                            public void onClose(Status status, Metadata trailers) {
                                rec.done(status);
                                super.onClose(status, trailers);
                            }
                        }, headers);
                    }

                    @Override
                    public void sendMessage(Q message) {
                        rec.out(message);
                        super.sendMessage(message);
                    }
                };
            }
        };
    }

    private static final class Recorder {
        private final MetricsSink sink;
        private final String prefix;
        private final long start = System.nanoTime();

        Recorder(MetricsSink sink, String prefix) {
            this.sink = sink;
            this.prefix = prefix;
        }

        void in(Object message) {
            sink.count(prefix + ".messages_in", 1);
            sink.count(prefix + ".bytes_in", size(message));
        }

        void out(Object message) {
            sink.count(prefix + ".messages_out", 1);
            sink.count(prefix + ".bytes_out", size(message));
        }

        void done(Status status) {
            sink.timeNanos(prefix + ".latency_ns", System.nanoTime() - start);
            sink.count(prefix + ".status." + status.getCode(), 1);
        }

        private static long size(Object message) {
            return message instanceof MessageLite m ? m.getSerializedSize() : 0L;
        }
    }
}
