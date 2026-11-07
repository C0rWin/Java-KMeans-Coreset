package io.github.c0rwin.coreset.grpc;

import com.sun.net.httpserver.HttpServer;
import io.micrometer.prometheusmetrics.PrometheusMeterRegistry;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Serves {@code GET /metrics} in Prometheus text format from the JDK's built-in HTTP server. */
public final class MetricsHttpServer implements AutoCloseable {

    private final HttpServer server;
    private final ExecutorService executor;

    private MetricsHttpServer(HttpServer server, ExecutorService executor) {
        this.server = server;
        this.executor = executor;
    }

    public static MetricsHttpServer start(InetSocketAddress address, PrometheusMeterRegistry registry)
            throws IOException {
        HttpServer server = HttpServer.create(address, 0);
        server.createContext("/metrics", exchange -> {
            try (exchange) {
                if (!"GET".equals(exchange.getRequestMethod())) {
                    exchange.sendResponseHeaders(405, -1);
                    return;
                }
                byte[] body = registry.scrape().getBytes(StandardCharsets.UTF_8);
                exchange.getResponseHeaders().set("Content-Type", "text/plain; version=0.0.4; charset=utf-8");
                exchange.sendResponseHeaders(200, body.length);
                try (OutputStream out = exchange.getResponseBody()) {
                    out.write(body);
                }
            }
        });
        ExecutorService executor = Executors.newSingleThreadExecutor(r -> {
            Thread t = new Thread(r, "coreset-metrics-http");
            t.setDaemon(true);
            return t;
        });
        server.setExecutor(executor);
        server.start();
        return new MetricsHttpServer(server, executor);
    }

    public int port() {
        return server.getAddress().getPort();
    }

    @Override
    public void close() {
        server.stop(0);
        executor.shutdownNow();
    }
}
