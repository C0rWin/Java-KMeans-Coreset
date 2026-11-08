package io.github.c0rwin.coreset.bench.perf;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.LongAdder;

/**
 * Loopback TCP relay counting the bytes that cross it in each direction, so wire numbers include gRPC's HTTP/2
 * framing, headers and acks. Thread per direction per connection.
 */
final class CountingProxy implements AutoCloseable {

    private final ServerSocket server;
    private final int targetPort;
    private final LongAdder toServer = new LongAdder();
    private final LongAdder toClient = new LongAdder();
    private final List<Socket> sockets = new CopyOnWriteArrayList<>();
    private final Thread acceptor;

    CountingProxy(int targetPort) throws IOException {
        this.targetPort = targetPort;
        this.server = new ServerSocket(0, 50, InetAddress.getLoopbackAddress());
        this.acceptor = new Thread(this::acceptLoop, "proxy-accept");
        acceptor.setDaemon(true);
        acceptor.start();
    }

    int port() {
        return server.getLocalPort();
    }

    long bytesToServer() {
        return toServer.sum();
    }

    long bytesToClient() {
        return toClient.sum();
    }

    private void acceptLoop() {
        while (!server.isClosed()) {
            try {
                Socket client = server.accept();
                Socket upstream = new Socket(InetAddress.getLoopbackAddress(), targetPort);
                client.setTcpNoDelay(true);
                upstream.setTcpNoDelay(true);
                sockets.add(client);
                sockets.add(upstream);
                pump(client, upstream, toServer);
                pump(upstream, client, toClient);
            } catch (IOException e) {
                return; // closed
            }
        }
    }

    private static void pump(Socket from, Socket to, LongAdder counter) {
        Thread t = new Thread(() -> {
            byte[] buf = new byte[64 * 1024];
            try (InputStream in = from.getInputStream(); OutputStream out = to.getOutputStream()) {
                int n;
                while ((n = in.read(buf)) >= 0) {
                    out.write(buf, 0, n);
                    counter.add(n);
                }
            } catch (IOException e) {
                // peer closed
            } finally {
                closeQuietly(to);
                closeQuietly(from);
            }
        }, "proxy-pump");
        t.setDaemon(true);
        t.start();
    }

    private static void closeQuietly(Socket s) {
        try {
            s.close();
        } catch (IOException e) {
            // already closed
        }
    }

    @Override
    public void close() throws IOException {
        server.close();
        for (Socket s : sockets) {
            closeQuietly(s);
        }
    }
}
