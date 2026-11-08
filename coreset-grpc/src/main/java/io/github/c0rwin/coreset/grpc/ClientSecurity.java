package io.github.c0rwin.coreset.grpc;

import io.grpc.ChannelCredentials;
import io.grpc.InsecureChannelCredentials;
import io.grpc.TlsChannelCredentials;
import java.io.IOException;
import java.nio.file.Path;

/**
 * Transport security of client channels (the coordinator's, and a node's channels to its peers).
 *
 * @param trustCa    PEM trust anchors for node certificates; enables TLS (otherwise plaintext)
 * @param certChain  PEM client certificate chain for mutual TLS, with {@code privateKey}
 * @param privateKey PEM PKCS#8 private key of {@code certChain}
 * @param authToken  shared secret sent as {@code authorization: Bearer <token>} on every call
 */
public record ClientSecurity(Path trustCa, Path certChain, Path privateKey, String authToken) {

    public ClientSecurity {
        if ((certChain == null) != (privateKey == null)) {
            throw new IllegalArgumentException("a client certificate needs both a chain and a private key");
        }
        if (certChain != null && trustCa == null) {
            throw new IllegalArgumentException("a client certificate needs TLS (trustCa)");
        }
        if (authToken != null && authToken.isEmpty()) {
            throw new IllegalArgumentException("auth token must not be empty");
        }
    }

    public static ClientSecurity plaintext() {
        return new ClientSecurity(null, null, null, null);
    }

    public boolean tls() {
        return trustCa != null;
    }

    /** Plaintext, or TLS trusting {@code trustCa} with the client certificate if one is set. */
    public ChannelCredentials channelCredentials() throws IOException {
        if (!tls()) {
            return InsecureChannelCredentials.create();
        }
        TlsChannelCredentials.Builder tls = TlsChannelCredentials.newBuilder().trustManager(trustCa.toFile());
        if (certChain != null) {
            tls.keyManager(certChain.toFile(), privateKey.toFile());
        }
        return tls.build();
    }
}
