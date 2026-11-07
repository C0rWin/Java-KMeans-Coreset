package io.github.c0rwin.coreset.grpc;

import java.nio.file.Path;

/**
 * Opt-in transport security of a node.
 *
 * @param certChain  PEM certificate chain served to clients; with {@code privateKey} enables TLS
 * @param privateKey PEM PKCS#8 private key of {@code certChain}
 * @param clientCa   PEM trust anchors for client certificates; when set, clients must present one (mutual TLS)
 * @param authToken  shared secret every call must carry as {@code authorization: Bearer <token>}; send it only over
 *                   TLS or a trusted network
 */
public record ServerSecurity(Path certChain, Path privateKey, Path clientCa, String authToken) {

    public ServerSecurity {
        if ((certChain == null) != (privateKey == null)) {
            throw new IllegalArgumentException("TLS needs both a certificate chain and a private key");
        }
        if (clientCa != null && certChain == null) {
            throw new IllegalArgumentException("client certificate authentication needs TLS");
        }
        if (authToken != null && authToken.isEmpty()) {
            throw new IllegalArgumentException("auth token must not be empty");
        }
    }

    /** Plaintext, unauthenticated: only for loopback or a trusted network. */
    public static ServerSecurity none() {
        return new ServerSecurity(null, null, null, null);
    }

    public boolean tls() {
        return certChain != null;
    }

    /** True when every caller that reaches a handler has proven its identity (token or client certificate). */
    public boolean authenticatesCallers() {
        return authToken != null || clientCa != null;
    }
}
