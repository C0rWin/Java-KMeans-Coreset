package io.github.c0rwin.coreset.grpc.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.CloseSessionRequest;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.node.ServiceLoaderAlgorithmResolver;
import io.grpc.ManagedChannel;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringReader;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

class CliTest {

    @Test
    void readsDenseAndSparseLines() throws IOException {
        PointsReader dense = new PointsReader(new BufferedReader(new StringReader("# header\n1,2 3\n\n4 5 6\n")), 0);
        List<WeightedPoint> d = dense.next(10);
        assertEquals(List.of(SparseVector.dense(new double[] {1, 2, 3}), SparseVector.dense(new double[] {4, 5, 6})),
                d.stream().map(WeightedPoint::vector).toList());
        assertTrue(dense.next(10).isEmpty());

        PointsReader sparse = new PointsReader(new BufferedReader(new StringReader("7 0:1.5 4:2\n3:1\n")), 5);
        List<WeightedPoint> s = sparse.next(1);
        assertEquals(SparseVector.of(5, new int[] {0, 4}, new double[] {1.5, 2}), s.get(0).vector());
        assertEquals(1.0, s.get(0).weight());
        assertEquals(SparseVector.of(5, new int[] {3}, new double[] {1}), sparse.next(1).get(0).vector());

        PointsReader bad = new PointsReader(new BufferedReader(new StringReader("1,2\n1,2,3\n")), 0);
        IOException e = assertThrows(IOException.class, () -> bad.next(10));
        assertTrue(e.getMessage().startsWith("line 2:"), e.getMessage());
    }

    @Test
    void nodeAndCoordinatorCommandsRunEndToEnd(@TempDir Path dir) throws Exception {
        // Node algorithms and the coordinator's final compression both come from ServiceLoader (test provider).
        List<NodeCommand.Running> nodes = new ArrayList<>();
        try {
            for (int i = 0; i < 2; i++) {
                NodeCommand cmd = new NodeCommand();
                new CommandLine(cmd).parseArgs("--port", "0", "--metrics-port", "-1");
                nodes.add(cmd.start(ServiceLoaderAlgorithmResolver.load()));
            }
            assertEquals(-1, nodes.get(0).metricsPort());
            Path input = dir.resolve("points.csv");
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < 600; i++) {
                int c = i % 2 == 0 ? 0 : 50;
                sb.append(c + i % 3).append(',').append(c).append('\n');
            }
            Files.writeString(input, sb);
            StringWriter out = new StringWriter();
            StringWriter err = new StringWriter();
            CommandLine cli = new CommandLine(new CoordinatorCommand());
            cli.setOut(new PrintWriter(out));
            cli.setErr(new PrintWriter(err));
            int code = cli.execute("--nodes", "localhost:" + nodes.get(0).port() + ",localhost:" + nodes.get(1).port(),
                    "--input", input.toString(), "--algorithm", "merge-equal", "--batch-size", "50", "--k", "2",
                    "--workers", "2", "--queue-capacity", "8", "--final-algorithm", "identity", "--session", "cli");
            assertEquals(0, code, err.toString());
            List<String> lines = Arrays.asList(out.toString().strip().split("\\R"));
            assertEquals(3, lines.size(), out.toString());
            assertTrue(lines.get(0).startsWith("# session=cli points=600 batches=12 "), lines.get(0));
            // Centers are the means of the two groups: x in {0,1,2} or {50,51,52}, y in {0, 50}.
            List<String> centers = lines.subList(1, 3).stream().sorted().toList();
            assertEquals(List.of("1.0,0.0", "51.0,50.0"), centers);
        } finally {
            for (NodeCommand.Running r : nodes) {
                r.close();
            }
        }
    }

    @Test
    void coordinatorClosesItsSessionWhenTheRunFails(@TempDir Path dir) throws Exception {
        NodeCommand cmd = new NodeCommand();
        new CommandLine(cmd).parseArgs("--port", "0", "--metrics-port", "-1");
        try (NodeCommand.Running node = cmd.start(ServiceLoaderAlgorithmResolver.load())) {
            Path input = dir.resolve("bad.csv");
            Files.writeString(input, "1,2\n3,4\n5,6,7\n"); // the third line has the wrong width
            StringWriter err = new StringWriter();
            CommandLine cli = new CommandLine(new CoordinatorCommand());
            cli.setOut(new PrintWriter(new StringWriter()));
            cli.setErr(new PrintWriter(err));
            int code = cli.execute("--nodes", "localhost:" + node.port(), "--input", input.toString(),
                    "--algorithm", "identity", "--batch-size", "1", "--k", "1", "--session", "leak");
            assertTrue(code != 0, err.toString());
            ManagedChannel ch = NettyChannelBuilder.forTarget("localhost:" + node.port()).usePlaintext().build();
            try {
                boolean stillOpen = CoresetNodeServiceGrpc.newBlockingStub(ch).withDeadlineAfter(10, TimeUnit.SECONDS)
                        .closeSession(CloseSessionRequest.newBuilder().setSessionId("leak").build()).getClosed();
                assertFalse(stillOpen, "the failed run left its session open on the node");
            } finally {
                ch.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            }
        }
    }

    @Test
    void coordinatorRejectsMissingRequiredOptions() {
        StringWriter err = new StringWriter();
        CommandLine cli = new CommandLine(new CoordinatorCommand());
        cli.setErr(new PrintWriter(err));
        assertEquals(2, cli.execute("--nodes", "localhost:1"));
        assertTrue(err.toString().contains("Missing required options"), err.toString());
    }
}
