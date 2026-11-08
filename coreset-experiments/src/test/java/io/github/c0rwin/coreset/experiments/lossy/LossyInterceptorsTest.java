package io.github.c0rwin.coreset.experiments.lossy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.LossyConfig;
import io.grpc.CallOptions;
import io.grpc.Channel;
import io.grpc.ClientCall;
import io.grpc.Metadata;
import io.grpc.MethodDescriptor;
import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class LossyInterceptorsTest {

    @Test
    void modelsLoseAtTheirRates() {
        ChannelModel none = ChannelModel.of(LossyConfig.OFF);
        ChannelModel bernoulli = ChannelModel.of(new LossyConfig("bernoulli", 0.3, null, null, null, null, null, 5L));
        ChannelModel ge = ChannelModel.of(new LossyConfig("gilbert-elliott", null, 0.01, 0.25, 0.0, 1.0, null, 5L));
        int n = 200_000;
        int lostNone = 0;
        int lostB = 0;
        int lostGe = 0;
        int bursts = 0;
        boolean prev = false;
        for (int i = 0; i < n; i++) {
            lostNone += none.drop() ? 1 : 0;
            lostB += bernoulli.drop() ? 1 : 0;
            boolean d = ge.drop();
            lostGe += d ? 1 : 0;
            bursts += d && !prev ? 1 : 0;
            prev = d;
        }
        assertEquals(0, lostNone);
        assertEquals(0.3, lostB / (double) n, 0.01);
        // stationary bad-state share pGB / (pGB + pBG) = 0.0385; mean burst 1 / pBG = 4
        assertEquals(0.01 / 0.26, lostGe / (double) n, 0.006);
        assertEquals(4.0, lostGe / (double) bursts, 0.4);
    }

    @Test
    void clientInterceptorDropsOnlySelectedMethods() {
        LossyInterceptors lossy = new LossyInterceptors(() -> true, Set.of("Ingest"));
        List<Object> sent = new ArrayList<>();
        Channel recording = new Channel() {
            @Override
            public <Q, R> ClientCall<Q, R> newCall(MethodDescriptor<Q, R> method, CallOptions options) {
                return new ClientCall<>() {
                    @Override
                    public void start(Listener<R> listener, Metadata headers) {
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
                        sent.add(message);
                    }
                };
            }

            @Override
            public String authority() {
                return "test";
            }
        };
        Channel ch = io.grpc.ClientInterceptors.intercept(recording, lossy.client());
        ch.newCall(method("Ingest"), CallOptions.DEFAULT).sendMessage("batch");
        ch.newCall(method("Snapshot"), CallOptions.DEFAULT).sendMessage("snapshot");
        assertEquals(List.of("snapshot"), sent);
        assertEquals(1, lossy.seen());
        assertEquals(1, lossy.dropped());
        assertTrue(lossy.dropped() <= lossy.seen());
    }

    private static MethodDescriptor<String, String> method(String name) {
        MethodDescriptor.Marshaller<String> m = new MethodDescriptor.Marshaller<>() {
            @Override
            public InputStream stream(String value) {
                return new ByteArrayInputStream(value.getBytes());
            }

            @Override
            public String parse(InputStream stream) {
                return "";
            }
        };
        return MethodDescriptor.<String, String>newBuilder().setType(MethodDescriptor.MethodType.BIDI_STREAMING)
                .setFullMethodName("svc/" + name).setRequestMarshaller(m).setResponseMarshaller(m).build();
    }
}
