package io.github.c0rwin.coreset.grpc;

import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptor;
import io.grpc.ForwardingClientCall;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;

/** Shared-token authentication: {@code authorization: Bearer <token>} on every call, compared in constant time. */
public final class TokenAuth {

    static final Metadata.Key<String> AUTHORIZATION =
            Metadata.Key.of("authorization", Metadata.ASCII_STRING_MARSHALLER);
    private static final String BEARER = "Bearer ";

    private TokenAuth() {
    }

    /** Rejects calls without the token with UNAUTHENTICATED before any handler runs. */
    public static ServerInterceptor server(String token) {
        byte[] expected = (BEARER + token).getBytes(StandardCharsets.UTF_8);
        return new ServerInterceptor() {
            @Override
            public <Q, R> ServerCall.Listener<Q> interceptCall(ServerCall<Q, R> call, Metadata headers,
                                                               ServerCallHandler<Q, R> next) {
                String got = headers.get(AUTHORIZATION);
                if (got == null || !MessageDigest.isEqual(expected, got.getBytes(StandardCharsets.UTF_8))) {
                    call.close(Status.UNAUTHENTICATED.withDescription("missing or invalid auth token"), new Metadata());
                    return new ServerCall.Listener<>() {
                    };
                }
                return next.startCall(call, headers);
            }
        };
    }

    /** Adds the token to every outgoing call. */
    public static ClientInterceptor client(String token) {
        String value = BEARER + token;
        return new ClientInterceptor() {
            @Override
            public <Q, R> ClientCall<Q, R> interceptCall(MethodDescriptor<Q, R> method, CallOptions options,
                                                         Channel next) {
                return new ForwardingClientCall.SimpleForwardingClientCall<>(next.newCall(method, options)) {
                    @Override
                    public void start(Listener<R> listener, Metadata headers) {
                        headers.put(AUTHORIZATION, value);
                        super.start(listener, headers);
                    }
                };
            }
        };
    }
}
