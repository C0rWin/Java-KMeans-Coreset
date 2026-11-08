package io.github.c0rwin.coreset.grpc;

import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.grpc.proto.Ack;
import io.github.c0rwin.coreset.grpc.proto.AckStatus;
import io.github.c0rwin.coreset.grpc.proto.Batch;
import io.github.c0rwin.coreset.grpc.proto.CoresetNodeServiceGrpc;
import io.github.c0rwin.coreset.node.Messages;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ConcurrentSkipListMap;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.LockSupport;

/**
 * Distributes batches of one session round-robin over the coordinator's node groups, one bidirectional stream per
 * node; a batch goes to every member of its group (one node per group unless a replica layout says otherwise).
 *
 * <p>Every batch gets a per-group dense seq (0, 1, 2, ... in each group, so every member's dedupe state stays
 * compact and the members see the same seq for the same batch) under this stream's {@link #producerId()} (random by
 * default), so a later stream into the same session numbers its batches independently instead of colliding with
 * this one. A batch stays in that node's outstanding map until
 * acknowledged ACCEPTED or DUPLICATE. BUSY acks resend the same seq after a backoff; a broken or expired stream is
 * reopened and all outstanding batches are resent with their original seqs, so nothing is lost and nothing is double
 * counted. A DUPLICATE ack for a batch that was sent only once cannot be a retry and fails the stream. At most
 * {@link CoordinatorOptions#maxInFlightPerNode()} batches are outstanding per node: {@link #submit} blocks beyond.
 *
 * <p>{@link #submit} may be called from several threads. Sends on a stream are serialised by a monitor because
 * gRPC's {@link StreamObserver} is not thread-safe; that lock is client-side only and never held while waiting.
 */
public final class IngestStream implements AutoCloseable {

    private final String sessionId;
    private final String producerId;
    private final CoordinatorOptions options;
    private final ScheduledExecutorService scheduler;
    private final NodeStream[] nodes;
    private final int[][] groups;
    private final AtomicLong[] groupSeqs;
    private final AtomicLong roundRobin = new AtomicLong();
    private final AtomicReference<IngestFailedException> failure = new AtomicReference<>();
    private final ConcurrentLinkedQueue<Thread> flushers = new ConcurrentLinkedQueue<>();
    private final LongAdder accepted = new LongAdder();
    private final LongAdder duplicates = new LongAdder();
    private final LongAdder busyRetries = new LongAdder();
    private final LongAdder reconnects = new LongAdder();
    private volatile boolean closed;

    IngestStream(String sessionId, String producerId, List<CoresetNodeServiceGrpc.CoresetNodeServiceStub> stubs,
                 int[][] groups, CoordinatorOptions options, ScheduledExecutorService scheduler) {
        this.sessionId = sessionId;
        this.producerId = producerId;
        this.options = options;
        this.scheduler = scheduler;
        this.nodes = new NodeStream[stubs.size()];
        for (int i = 0; i < nodes.length; i++) {
            nodes[i] = new NodeStream(i, stubs.get(i));
        }
        this.groups = groups;
        this.groupSeqs = new AtomicLong[groups.length];
        for (int g = 0; g < groups.length; g++) {
            groupSeqs[g] = new AtomicLong();
        }
    }

    /** Id naming this stream's seq namespace on every node (random unless the caller chose one). */
    public String producerId() {
        return producerId;
    }

    /**
     * Queues one batch for every member of the next group in round-robin order, blocking while a member has the
     * maximum number of unacknowledged batches. The points are encoded once, straight into the outgoing message.
     *
     * @throws IngestFailedException if the stream already failed, or no capacity freed up within
     *                               {@link CoordinatorOptions#submitTimeout()} (the stream is then failed)
     */
    public void submit(List<WeightedPoint> batch) throws InterruptedException, IngestFailedException {
        throwIfFailed();
        if (closed) {
            throw new IllegalStateException("ingest stream closed");
        }
        Batch.Builder message = Batch.newBuilder().setSessionId(sessionId).setProducerId(producerId);
        for (int i = 0, n = batch.size(); i < n; i++) {
            message.addPoints(ProtoCodec.toProto(batch.get(i)));
        }
        int g = (int) Math.floorMod(roundRobin.getAndIncrement(), (long) groups.length);
        long seq = groupSeqs[g].getAndIncrement();
        for (int member : groups[g]) {
            nodes[member].submit(message, seq);
        }
    }

    /**
     * Waits until every submitted batch is acknowledged by its node (accepted or duplicate).
     *
     * @throws TimeoutException      if batches are still unacknowledged after {@code timeout}
     * @throws IngestFailedException if a batch was rejected permanently or a node stayed unreachable
     */
    public void flush(Duration timeout) throws InterruptedException, TimeoutException, IngestFailedException {
        long deadline = System.nanoTime() + timeout.toNanos();
        Thread me = Thread.currentThread();
        flushers.add(me);
        try {
            for (;;) {
                throwIfFailed();
                long outstanding = outstanding();
                if (outstanding == 0) {
                    return;
                }
                long remaining = deadline - System.nanoTime();
                if (remaining <= 0) {
                    throw new TimeoutException(outstanding + " batches unacknowledged after " + timeout);
                }
                LockSupport.parkNanos(this, remaining);
                if (Thread.interrupted()) {
                    throw new InterruptedException();
                }
            }
        } finally {
            flushers.remove(me);
        }
    }

    /** Batches submitted but not yet acknowledged. */
    public long outstanding() {
        long n = 0;
        for (NodeStream node : nodes) {
            n += node.outstanding.size();
        }
        return n;
    }

    public long acceptedBatches() {
        return accepted.sum();
    }

    public long duplicateAcks() {
        return duplicates.sum();
    }

    public long busyRetries() {
        return busyRetries.sum();
    }

    public long reconnects() {
        return reconnects.sum();
    }

    /** Half-closes every stream; call {@link #flush} first, unacknowledged batches are abandoned. */
    @Override
    public void close() {
        closed = true;
        for (NodeStream node : nodes) {
            node.halfClose();
        }
        signalFlushers();
    }

    private void throwIfFailed() throws IngestFailedException {
        IngestFailedException f = failure.get();
        if (f != null) {
            throw new IngestFailedException(f.getMessage(), f.getCause());
        }
    }

    private void fail(String message, Throwable cause) {
        failure.compareAndSet(null, new IngestFailedException(message, cause));
        for (NodeStream node : nodes) {
            node.permits.release(options.maxInFlightPerNode()); // unblock producers so they observe the failure
        }
        signalFlushers();
    }

    private void signalFlushers() {
        for (Thread t : flushers) {
            LockSupport.unpark(t);
        }
    }

    private static final class Pending {
        final Batch batch;
        int busyAttempts; // acks of one stream are delivered serially
        long firstBusyNanos;
        /** Times handed to a stream; written under the NodeStream monitor before the send. */
        volatile int sends;
        /** A BUSY resend is scheduled: a reconnect must not send it a second time meanwhile. */
        final AtomicBoolean resendScheduled = new AtomicBoolean();
        /** Counted as acknowledged; a batch acknowledged twice (resent) is counted once. */
        final AtomicBoolean acknowledged = new AtomicBoolean();

        Pending(Batch batch) {
            this.batch = batch;
        }
    }

    private final class NodeStream {
        final int index;
        final CoresetNodeServiceGrpc.CoresetNodeServiceStub stub;
        final Semaphore permits = new Semaphore(options.maxInFlightPerNode());
        final ConcurrentSkipListMap<Long, Pending> outstanding = new ConcurrentSkipListMap<>();
        private StreamObserver<Batch> requests; // guarded by this
        private int generation; // guarded by this
        private int brokenGeneration; // guarded by this
        private int consecutiveFailures; // guarded by this

        NodeStream(int index, CoresetNodeServiceGrpc.CoresetNodeServiceStub stub) {
            this.index = index;
            this.stub = stub;
        }

        void submit(Batch.Builder message, long seq) throws InterruptedException, IngestFailedException {
            if (!permits.tryAcquire(options.submitTimeout().toNanos(), TimeUnit.NANOSECONDS)) {
                fail("node " + index + " acknowledged nothing for " + options.submitTimeout() + " with "
                        + options.maxInFlightPerNode() + " batches in flight", null);
            }
            throwIfFailed();
            Pending p = new Pending(message.setSeq(seq).build());
            outstanding.put(seq, p);
            send(p);
        }

        synchronized void send(Pending p) {
            if (closed || failure.get() != null) {
                return;
            }
            StreamObserver<Batch> r = requests;
            if (r == null) {
                r = open(); // a synchronous start failure may already have reset `requests`
            }
            p.sends++; // before the send, so an ack racing back never sees a resend counted as the first send
            r.onNext(p.batch);
        }

        /** Opens a new stream generation; acks or errors from older generations are ignored. */
        private StreamObserver<Batch> open() {
            final int gen = ++generation;
            StreamObserver<Batch> r = stub.withDeadlineAfter(options.streamDeadline().toNanos(), TimeUnit.NANOSECONDS)
                    .ingest(new StreamObserver<>() {
                        @Override
                        public void onNext(Ack ack) {
                            onAck(gen, ack);
                        }

                        @Override
                        public void onError(Throwable t) {
                            onBroken(gen, t);
                        }

                        @Override
                        public void onCompleted() {
                            onBroken(gen, Status.UNAVAILABLE.withDescription("server ended the stream")
                                    .asRuntimeException());
                        }
                    });
            if (gen != brokenGeneration) {
                requests = r;
            }
            return r;
        }

        private void onAck(int gen, Ack ack) {
            long seq = ack.getSeq();
            switch (ack.getStatus()) {
                case ACK_STATUS_ACCEPTED, ACK_STATUS_DUPLICATE -> {
                    synchronized (this) {
                        if (gen == generation) {
                            consecutiveFailures = 0; // only progress counts: a node answering BUSY is not healthy
                        }
                    }
                    Pending p = outstanding.get(seq);
                    if (p != null && ack.getStatus() == AckStatus.ACK_STATUS_DUPLICATE && p.sends == 1) {
                        fail("node " + index + " acknowledged batch " + seq + " of producer " + producerId
                                + " as DUPLICATE although it was sent once; refusing to count it as ingested", null);
                        return;
                    }
                    // A batch resent after a reconnect or BUSY can be acknowledged twice; count it once, and before
                    // it leaves outstanding, so a flush that sees nothing outstanding also sees the count.
                    if (p != null && p.acknowledged.compareAndSet(false, true)) {
                        (ack.getStatus() == AckStatus.ACK_STATUS_ACCEPTED ? accepted : duplicates).increment();
                        outstanding.remove(seq);
                        permits.release();
                        signalFlushers();
                    }
                }
                case ACK_STATUS_BUSY -> {
                    Pending p = outstanding.get(seq);
                    if (p != null) {
                        busyRetries.increment();
                        long now = System.nanoTime();
                        if (p.busyAttempts++ == 0) {
                            p.firstBusyNanos = now;
                        } else if (now - p.firstBusyNanos > options.maxBusy().toNanos()) {
                            fail("node " + index + " answered BUSY to batch " + seq + " for longer than "
                                    + options.maxBusy(), null);
                            return;
                        }
                        if (p.resendScheduled.compareAndSet(false, true)) {
                            scheduler.schedule(() -> {
                                p.resendScheduled.set(false);
                                if (outstanding.containsKey(seq)) {
                                    send(p);
                                }
                            }, options.backoffNanos(p.busyAttempts), TimeUnit.NANOSECONDS);
                        }
                    }
                }
                default -> fail("node " + index + " rejected batch " + seq + ": " + Messages.bounded(ack.getMessage()),
                        null);
            }
        }

        private void onBroken(int gen, Throwable t) {
            int attempt;
            synchronized (this) {
                if (gen != generation || closed) {
                    return;
                }
                brokenGeneration = gen;
                requests = null;
                attempt = ++consecutiveFailures;
            }
            if (Status.fromThrowable(t).getCode() == Status.Code.INVALID_ARGUMENT) {
                fail("node " + index + " rejected the stream: "
                        + Messages.bounded(Status.fromThrowable(t).getDescription()), t);
                return; // resending the same bytes cannot succeed
            }
            if (attempt > options.maxTransportFailures()) {
                fail("node " + index + " unreachable after " + (attempt - 1) + " reconnects", t);
                return;
            }
            if (outstanding.isEmpty()) {
                return; // reopened lazily by the next submit
            }
            reconnects.increment();
            scheduler.schedule(this::resendAll, options.backoffNanos(attempt), TimeUnit.NANOSECONDS);
        }

        private void resendAll() {
            for (Map.Entry<Long, Pending> e : outstanding.entrySet()) {
                if (!e.getValue().resendScheduled.get()) {
                    send(e.getValue());
                }
            }
        }

        synchronized void halfClose() {
            if (requests != null) {
                requests.onCompleted();
                requests = null;
                generation++;
            }
        }
    }
}
