package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.grpc.proto.MessageHeader;
import io.github.c0rwin.coreset.grpc.proto.Packet;
import io.github.c0rwin.coreset.grpc.proto.SnapshotChunk;
import io.github.c0rwin.coreset.grpc.proto.SnapshotStreamRequest;
import io.github.c0rwin.coreset.grpc.resilience.Census;
import io.github.c0rwin.coreset.grpc.resilience.GroupLayout;
import io.github.c0rwin.coreset.grpc.resilience.MessageDecoder;
import io.github.c0rwin.coreset.grpc.resilience.ReceivedMessage;
import io.github.c0rwin.coreset.grpc.resilience.ResilienceOptions;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot.GroupOutcome;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot.SourceOutcome;
import io.github.c0rwin.coreset.grpc.resilience.SurvivorReweighting;
import io.grpc.Status;
import io.grpc.stub.ClientCallStreamObserver;
import io.grpc.stub.ClientResponseObserver;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReferenceArray;

/**
 * One resilient gather: fetches every node's message (failing over to its replica holders), waits for all of them,
 * the deadline, or a quorum per group, then decodes per group and rescales for lost groups.
 */
final class ResilientGather {

    /** Outcome of fetching one node's message; immutable once published. */
    private record Fetch(ReceivedMessage message, SourceOutcome outcome) {
    }

    private final List<CoresetNodeServiceGrpc.CoresetNodeServiceStub> stubs;
    private final String sessionId;
    private final ResilienceOptions options;
    private final int maxDim;
    private final AtomicReferenceArray<Fetch> fetched;
    private final List<ClientCallStreamObserver<?>> calls = new ArrayList<>();
    private final LinkedBlockingQueue<Integer> resolved = new LinkedBlockingQueue<>();
    private volatile boolean finished; // set under the calls monitor

    ResilientGather(List<CoresetNodeServiceGrpc.CoresetNodeServiceStub> stubs, String sessionId,
                    ResilienceOptions options, int maxDim) {
        if (options.layout().nodes() != stubs.size()) {
            throw new IllegalArgumentException("layout has " + options.layout().nodes() + " nodes, coordinator "
                    + stubs.size());
        }
        this.stubs = stubs;
        this.sessionId = sessionId;
        this.options = options;
        this.maxDim = maxDim;
        this.fetched = new AtomicReferenceArray<>(stubs.size());
    }

    ResilientSnapshot run() throws InterruptedException {
        int n = stubs.size();
        for (int i = 0; i < n; i++) {
            fetch(i, 0, "");
        }
        awaitResolution();
        synchronized (calls) {
            finished = true;
            for (ClientCallStreamObserver<?> c : calls) {
                c.cancel("gather finished", null); // no-op for completed calls
            }
        }
        return decode();
    }

    /** Attempt 0 asks the owner; attempt j >= 1 asks replica holder owner + j. */
    private void fetch(int owner, int attempt, String lastError) {
        if (finished) {
            return;
        }
        int n = stubs.size();
        int server = (owner + attempt) % n;
        boolean replica = attempt > 0;
        SnapshotStreamRequest req = SnapshotStreamRequest.newBuilder()
                .setSessionId(sessionId)
                .setOwner(owner)
                .setReplica(replica)
                .setStream(options.streamOf(owner))
                .setAwaitAppliedMs(replica ? 0 : options.awaitApplied().toMillis())
                .build();
        long deadlineNanos = options.callDeadline().plus(replica ? Duration.ZERO : options.awaitApplied()).toNanos();
        stubs.get(server).withDeadlineAfter(deadlineNanos, TimeUnit.NANOSECONDS)
                .snapshotStream(req, new ClientResponseObserver<SnapshotStreamRequest, SnapshotChunk>() {
                    private MessageHeader header; // callbacks of one call are serialised
                    private final List<Packet> packets = new ArrayList<>();

                    @Override
                    public void beforeStart(ClientCallStreamObserver<SnapshotStreamRequest> call) {
                        synchronized (calls) {
                            calls.add(call);
                        }
                    }

                    @Override
                    public void onNext(SnapshotChunk chunk) {
                        if (chunk.hasHeader() && header == null) {
                            header = chunk.getHeader();
                        } else if (chunk.hasPacket() && header != null) {
                            packets.add(chunk.getPacket());
                        }
                    }

                    @Override
                    public void onError(Throwable t) {
                        String error = "node " + server + ": " + StatusText.of(t);
                        if (attempt < options.replication()) {
                            fetch(owner, attempt + 1, error); // a fresh full copy beats a truncated one
                        } else if (header != null) {
                            // Truncation is content-oblivious under random packing, so what arrived is decodable.
                            publish(owner, new Fetch(new ReceivedMessage(server, header, packets),
                                    new SourceOutcome(owner, server, false, error, header.getPackets(),
                                            packets.size())));
                        } else {
                            publish(owner, new Fetch(null, new SourceOutcome(owner, -1, false, error, 0, 0)));
                        }
                    }

                    @Override
                    public void onCompleted() {
                        if (header == null) {
                            onError(Status.DATA_LOSS.withDescription("stream ended without a header")
                                    .asRuntimeException());
                            return;
                        }
                        publish(owner, new Fetch(new ReceivedMessage(server, header, packets),
                                new SourceOutcome(owner, server, true, lastError, header.getPackets(),
                                        packets.size())));
                    }
                });
    }

    private void publish(int owner, Fetch f) {
        if (fetched.compareAndSet(owner, null, f)) {
            resolved.add(owner);
        }
    }

    /** Until every node resolved, every group has its quorum of completed messages, or the overall deadline. */
    private void awaitResolution() throws InterruptedException {
        int n = stubs.size();
        long budget = options.callDeadline().plus(options.awaitApplied()).toNanos()
                + options.callDeadline().toNanos() * options.replication();
        long deadline = System.nanoTime() + budget + TimeUnit.SECONDS.toNanos(1);
        int done = 0;
        while (done < n) {
            if (options.minResponders() > 0 && quorum()) {
                return;
            }
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) {
                return;
            }
            if (resolved.poll(remaining, TimeUnit.NANOSECONDS) != null) {
                done++;
            }
        }
    }

    private boolean quorum() {
        GroupLayout layout = options.layout();
        for (int g = 0; g < layout.groupCount(); g++) {
            int completed = 0;
            for (int node : layout.members(g)) {
                Fetch f = fetched.get(node);
                completed += f != null && f.outcome().completed() ? 1 : 0;
            }
            if (completed < Math.min(options.minResponders(), layout.members(g).length)) {
                return false;
            }
        }
        return true;
    }

    private ResilientSnapshot decode() {
        GroupLayout layout = options.layout();
        int groups = layout.groupCount();
        List<List<ReceivedMessage>> pooled = new ArrayList<>(groups);
        List<List<SourceOutcome>> outcomes = new ArrayList<>(groups);
        double deliveredWeight = 0.0;
        int delivered = 0;
        for (int g = 0; g < groups; g++) {
            List<ReceivedMessage> messages = new ArrayList<>();
            List<SourceOutcome> sources = new ArrayList<>();
            for (int node : layout.members(g)) {
                Fetch f = fetched.get(node);
                if (f == null) {
                    sources.add(new SourceOutcome(node, -1, false, "straggler: no reply in time", 0, 0));
                    continue;
                }
                sources.add(f.outcome());
                if (f.message() != null) {
                    messages.add(f.message());
                }
            }
            if (!messages.isEmpty() && layout.mode() == GroupLayout.Mode.FRACTIONAL_REPETITION) {
                messages = List.of(MessageDecoder.unionOfIdentical(messages));
            } else if (!messages.isEmpty() && layout.mode() == GroupLayout.Mode.INDEPENDENT) {
                messages = List.of(messages.get(0));
            }
            if (!decodable(messages)) {
                messages = List.of(); // headers alone carry no mass: the group counts as lost
            }
            pooled.add(messages);
            outcomes.add(sources);
            if (!messages.isEmpty()) {
                delivered++;
                deliveredWeight += messages.get(0).header().getNodeWeight();
            }
        }
        double survivorScale = survivorScale(groups, delivered, deliveredWeight);
        List<Coreset> parts = new ArrayList<>(delivered);
        List<GroupOutcome> report = new ArrayList<>(groups);
        for (int g = 0; g < groups; g++) {
            List<ReceivedMessage> messages = pooled.get(g);
            if (messages.isEmpty()) {
                report.add(new GroupOutcome(g, false, Double.NaN, 0.0, 0, 0, outcomes.get(g)));
                continue;
            }
            MessageDecoder.Decoded d = MessageDecoder.decode(messages, options.decoder(), options.knownLossRate(),
                    survivorScale, maxDim);
            parts.add(d.coreset());
            long sent = 0;
            for (ReceivedMessage m : messages) {
                sent += Integer.toUnsignedLong(m.header().getRecords());
            }
            report.add(new GroupOutcome(g, true, messages.get(0).header().getNodeWeight(), d.decoderScale(), sent,
                    d.recordsReceived(), outcomes.get(g)));
        }
        return new ResilientSnapshot(Coreset.union(parts), survivorScale, report);
    }

    /** Some record arrived, or there was nothing to send (an empty node is delivered). */
    private static boolean decodable(List<ReceivedMessage> messages) {
        boolean empty = !messages.isEmpty();
        for (ReceivedMessage m : messages) {
            if (m.recordsReceived() > 0) {
                return true;
            }
            empty &= m.header().getRecords() == 0;
        }
        return empty;
    }

    private double survivorScale(int groups, int delivered, double deliveredWeight) {
        if (delivered == 0 || delivered == groups) {
            return 1.0;
        }
        SurvivorReweighting r = options.reweighting();
        if (r == SurvivorReweighting.NODE_FRACTION) {
            return (double) groups / delivered;
        }
        if (r == SurvivorReweighting.REALIZED_WEIGHT) {
            Census census = options.census();
            double total = census.totalWeight(options.layout());
            if (!(total >= 0.0) || Double.isInfinite(total)) {
                throw new IllegalStateException("census lacks the weight of some group");
            }
            return deliveredWeight > 0.0 ? total / deliveredWeight : 1.0;
        }
        return 1.0;
    }
}
