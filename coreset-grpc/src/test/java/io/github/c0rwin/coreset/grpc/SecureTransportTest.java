package io.github.c0rwin.coreset.grpc;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.grpc.cli.NodeCommand;
import io.github.c0rwin.coreset.grpc.proto.StatsResponse;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.testing.TestData;
import io.github.c0rwin.coreset.node.testing.TestProvider;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

/** Bind defaults, TLS, mutual TLS and the shared token, over real Netty transports (review H6, M8). */
class SecureTransportTest {

    private static final Duration WAIT = Duration.ofSeconds(30);

    private static NodeCommand.Running node(String... args) throws Exception {
        NodeCommand cmd = new NodeCommand();
        List<String> all = new ArrayList<>(List.of("--port", "0", "--metrics-port", "0", "--shutdown-grace-ms", "200"));
        all.addAll(List.of(args));
        new CommandLine(cmd).parseArgs(all.toArray(String[]::new));
        return cmd.start(TestProvider.registry());
    }

    private static CoordinatorOptions fast() {
        return new CoordinatorOptions(4, Duration.ofSeconds(10), Duration.ofMinutes(1), Duration.ofMillis(1),
                Duration.ofMillis(10), 2, CoordinatorOptions.DEFAULT_MAX_MESSAGE_BYTES, WAIT, WAIT, 1 << 24, 2);
    }

    /** Creates a session, streams two batches and gathers them; returns the node's Stats. */
    private static StatsResponse roundTrip(int port, ClientSecurity security) throws Exception {
        try (CoresetCoordinator c = CoresetCoordinator.connect(List.of("localhost:" + port), fast(), security,
                MetricsSink.NOOP)) {
            c.createSession("secure", new SessionSpec("identity", Map.of(), 1L, 1, 4));
            try (IngestStream s = c.openIngest("secure")) {
                s.submit(TestData.batch(new SplittableRandom(1), 3, 4));
                s.submit(TestData.batch(new SplittableRandom(2), 3, 4));
                s.flush(WAIT);
            }
            assertEquals(2, c.gather("secure", WAIT).appliedBatches());
            StatsResponse stats = c.stats().get(0);
            c.closeSession("secure");
            return stats;
        }
    }

    private static Status.Code failure(int port, ClientSecurity security) throws Exception {
        try (CoresetCoordinator c = CoresetCoordinator.connect(List.of("localhost:" + port), fast(), security,
                MetricsSink.NOOP)) {
            return assertThrows(StatusRuntimeException.class,
                    () -> c.createSession("denied", new SessionSpec("identity", Map.of(), 1L, 1, 2)))
                    .getStatus().getCode();
        }
    }

    @Test
    void listensOnLoopbackUnlessToldOtherwise() throws Exception {
        try (NodeCommand.Running local = node(); NodeCommand.Running exposed = node("--bind", "0.0.0.0")) {
            for (SocketAddress a : local.listenAddresses()) {
                assertTrue(((InetSocketAddress) a).getAddress().isLoopbackAddress(), a.toString());
            }
            assertTrue(local.metricsAddress().getAddress().isLoopbackAddress(), local.metricsAddress().toString());
            for (SocketAddress a : exposed.listenAddresses()) {
                assertTrue(((InetSocketAddress) a).getAddress().isAnyLocalAddress(), a.toString());
            }
            // Plaintext without a token: the node works, but Stats does not list sessions to anonymous callers.
            assertEquals(List.of(), roundTrip(local.port(), ClientSecurity.plaintext()).getSessionIdsList());
        }
    }

    @Test
    void tlsWithTokenAuthenticatesEveryCall(@TempDir Path dir) throws Exception {
        TestCerts.Pem server = TestCerts.generate(dir, "node");
        Path tokenFile = Files.writeString(dir.resolve("token"), "s3cret\n");
        try (NodeCommand.Running n = node("--tls-cert", server.cert().toString(), "--tls-key", server.key().toString(),
                "--auth-token-file", tokenFile.toString())) {
            StatsResponse stats = roundTrip(n.port(), new ClientSecurity(server.cert(), null, null, "s3cret"));
            assertEquals(List.of("secure"), stats.getSessionIdsList()); // authenticated callers may list sessions
            assertEquals(Status.Code.UNAUTHENTICATED, failure(n.port(), new ClientSecurity(server.cert(), null, null,
                    null)));
            assertEquals(Status.Code.UNAUTHENTICATED, failure(n.port(), new ClientSecurity(server.cert(), null, null,
                    "wrong")));
            assertEquals(Status.Code.UNAVAILABLE, failure(n.port(), ClientSecurity.plaintext()));
        }
    }

    @Test
    void mutualTlsRequiresATrustedClientCertificate(@TempDir Path dir) throws Exception {
        TestCerts.Pem server = TestCerts.generate(dir, "node");
        TestCerts.Pem client = TestCerts.generate(dir, "coordinator");
        TestCerts.Pem stranger = TestCerts.generate(dir, "stranger");
        try (NodeCommand.Running n = node("--tls-cert", server.cert().toString(), "--tls-key", server.key().toString(),
                "--tls-client-ca", client.cert().toString())) {
            StatsResponse stats = roundTrip(n.port(), new ClientSecurity(server.cert(), client.cert(), client.key(),
                    null));
            assertFalse(stats.getMetricsJson().isEmpty());
            assertEquals(List.of("secure"), stats.getSessionIdsList());
            assertEquals(Status.Code.UNAVAILABLE, failure(n.port(), new ClientSecurity(server.cert(), null, null,
                    null)));
            assertEquals(Status.Code.UNAVAILABLE, failure(n.port(), new ClientSecurity(server.cert(),
                    stranger.cert(), stranger.key(), null)));
        }
    }
}
