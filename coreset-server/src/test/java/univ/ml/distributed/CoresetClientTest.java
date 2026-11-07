package univ.ml.distributed;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;

import org.apache.thrift.server.TNonblockingServer;
import org.apache.thrift.server.TServer;
import org.apache.thrift.transport.TNonblockingServerSocket;
import org.apache.thrift.transport.TTransportException;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import univ.ml.distributed.coreset.ClientConfig;
import univ.ml.distributed.coreset.CoresetAlgorithm;
import univ.ml.distributed.coreset.CoresetClient;
import univ.ml.distributed.coreset.CoresetService;
import univ.ml.distributed.coreset.CoresetServiceHandler;
import univ.ml.distributed.coreset.RandomPointsProvider;

class CoresetClientTest {

    private static final Duration HANG = Duration.ofSeconds(10);

    private TNonblockingServerSocket serverSocket;

    private TServer server;

    private Thread serverThread;

    @BeforeEach
    void startServer() throws TTransportException {
        // Port 0: an ephemeral port, so the test never collides with another server.
        serverSocket = new TNonblockingServerSocket(0);
        server = new TNonblockingServer(new TNonblockingServer.Args(serverSocket)
                .processor(new CoresetService.Processor<>(new CoresetServiceHandler())));
        serverThread = new Thread(server::serve, "coreset-test-server");
        serverThread.setDaemon(true);
        serverThread.start();
    }

    @AfterEach
    void stopServer() throws InterruptedException {
        server.stop();
        serverThread.join(HANG.toMillis());
        serverSocket.close();
    }

    @Test
    void distributeAllPointsFailsWhenABatchCannotBeSent() {
        final CoresetClient client = new CoresetClient(config(), new RandomPointsProvider(30, 3));
        assertTrue(client.initialize());
        // A closed transport makes every send fail, like a server that went away.
        client.closeAll();
        assertThrows(IllegalStateException.class, client::distributeAllPoints);
    }

    @Test
    void distributeAllPointsFailsWithoutAConnectedServer() {
        // No successful initialize(), which is where main() ends up when it returns false.
        final CoresetClient client = new CoresetClient(config(), new RandomPointsProvider(30, 3));
        assertTimeoutPreemptively(HANG, () -> assertThrows(IllegalStateException.class, client::distributeAllPoints));
    }

    private ClientConfig config() {
        return ClientConfig.init()
                .k(2)
                .coresetAlgName(CoresetAlgorithm.UNIFORM)
                .sampleSize(100)
                .batchSize(10)
                .hosts("localhost:" + serverSocket.getPort())
                .create();
    }
}
