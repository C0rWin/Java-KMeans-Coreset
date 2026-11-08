package io.github.c0rwin.coreset.bench.jmh;

import io.github.c0rwin.coreset.core.metrics.RecordingMetrics;
import io.github.c0rwin.coreset.grpc.GrpcMetrics;
import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.AckStatus;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.ClientInterceptors;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Per-message cost of the {@link GrpcMetrics} client interceptor on a long-lived ingest stream: one outbound batch
 * and one inbound ack recorded into {@link RecordingMetrics}, over a transport stub that does nothing, so only the
 * instrumentation is measured.
 */
@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 2, jvmArgsAppend = {"-Xms2g", "-Xmx2g"})
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class GrpcMetricsBenchmark {

    private ClientCall<Batch, Ack> call;
    private ClientCall.Listener<Ack> listener;
    private Batch batch;
    private Ack ack;

    @Setup
    public void setup() {
        Channel inert = new Channel() {
            @Override
            @SuppressWarnings("unchecked")
            public <Q, R> ClientCall<Q, R> newCall(MethodDescriptor<Q, R> method, CallOptions options) {
                return new ClientCall<>() {
                    @Override
                    public void start(Listener<R> l, Metadata headers) {
                        listener = (Listener<Ack>) l;
                    }

                    @Override
                    public void request(int numMessages) {
                    }

                    @Override
                    public void cancel(String message, Throwable cause) {
                    }

                    @Override
                    public void halfClose() {
                    }

                    @Override
                    public void sendMessage(Q message) {
                    }
                };
            }

            @Override
            public String authority() {
                return "bench";
            }
        };
        Channel ch = ClientInterceptors.intercept(inert, GrpcMetrics.client(new RecordingMetrics()));
        call = ch.newCall(CoresetNodeServiceGrpc.getIngestMethod(), CallOptions.DEFAULT);
        call.start(new ClientCall.Listener<>() {
        }, new Metadata());
        batch = Batch.newBuilder().setSessionId("bench").setSeq(1).build();
        ack = Ack.newBuilder().setSeq(1).setStatus(AckStatus.ACK_STATUS_ACCEPTED).build();
    }

    @Benchmark
    public void sendAndAck() {
        call.sendMessage(batch);
        listener.onMessage(ack);
    }
}
