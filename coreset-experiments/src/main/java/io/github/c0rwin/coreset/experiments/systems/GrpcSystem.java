package io.github.c0rwin.coreset.experiments.systems;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.experiments.config.CellSpec;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.FaultConfig;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.SystemConfig;
import io.github.c0rwin.coreset.experiments.data.ChunkReader;
import io.github.c0rwin.coreset.experiments.data.PointSource;
import io.github.c0rwin.coreset.experiments.lossy.ChannelModel;
import io.github.c0rwin.coreset.experiments.lossy.LossyInterceptors;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestFailedException;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.node.NodeSnapshot;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.grpc.ManagedChannel;
import io.grpc.StatusRuntimeException;
import io.grpc.netty.shaded.io.grpc.netty.NettyChannelBuilder;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.concurrent.atomic.DoubleAdder;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * The repository's own system: N {@code coreset-node} processes and the coordinator in this JVM. Shard {@code i}
 * (the same contiguous row range a Spark partition or Flink subtask reads) is read and streamed to node {@code i} by
 * one coordinator thread through its own {@link IngestStream}, with the stock retry and dedupe protocol.
 *
 * <p>The stock coordinator fails a whole ingest or gather when one node is gone. To measure behaviour under node
 * loss the harness keeps one single-node coordinator per node (sessions are still created through one coordinator so
 * node {@code i} keeps partition {@code i}): a node whose stream fails is declared dead, its remaining shard is dropped
 * or rerouted to survivors, and the final coreset is gathered from live nodes only, plus the last snapshot of each dead
 * node when one was taken. Lost weight is the input weight not reflected in that union; optionally the coreset is
 * reweighted by input weight / realized weight.
 */
final class GrpcSystem implements EngineSystem {

    private static final Pattern BYTES = Pattern.compile(
            "^grpc_server_(\\w+?)_bytes_(in|out)_total(?:\\{[^}]*\\})?\\s+([0-9.eE+-]+)", Pattern.MULTILINE);

    private final CellSpec cell;
    private final PointSource src;
    private final SystemConfig sys;
    private final int n;
    private final boolean external;
    private final LossyInterceptors lossy;
    private final HttpClient http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    private final Duration flushTimeout;
    private final CoordinatorOptions options;
    private final List<NodeProcess> nodes = new ArrayList<>();
    private final List<String> targets = new ArrayList<>();
    private final List<String> metricsUrls = new ArrayList<>();
    private final List<ManagedChannel> channels = new ArrayList<>();
    private final List<CoresetCoordinator> perNode = new ArrayList<>();
    private CoresetCoordinator all;

    GrpcSystem(CellSpec cell, PointSource src) {
        this.cell = cell;
        this.src = src;
        this.sys = cell.system();
        this.n = cell.parallelism();
        this.external = sys.mode().equals("external");
        this.lossy = cell.lossy().enabled()
                ? new LossyInterceptors(ChannelModel.of(cell.lossy()), new HashSet<>(cell.lossy().methods())) : null;
        this.flushTimeout = Duration.ofSeconds(sys.longOption("flushTimeoutSeconds", 600));
        this.options = CoordinatorOptions.defaults().withMaxInFlightPerNode(sys.intOption("maxInFlight", 16));
        if (!external && !sys.mode().equals("spawn")) {
            throw new IllegalArgumentException("grpc mode must be spawn or external: " + sys.mode());
        }
    }

    @Override
    public String selfRole() {
        return "coordinator";
    }

    @Override
    public void start() throws Exception {
        if (external) {
            List<String> t = cell.cluster().grpcTargets();
            if (t.size() < n) {
                throw new IllegalArgumentException("external grpc needs " + n + " targets, config has " + t.size());
            }
            targets.addAll(t.subList(0, n));
            if (cell.cluster().grpcMetrics().size() >= n) {
                metricsUrls.addAll(cell.cluster().grpcMetrics().subList(0, n));
            }
        } else {
            spawnAll();
        }
        connect();
    }

    private int nodeHeapMB() {
        return Math.max(256, cell.memory().workersMB() / n);
    }

    private void spawnAll() throws Exception {
        ExecutorService ex = Executors.newFixedThreadPool(n);
        try {
            List<Future<NodeProcess>> f = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                final int idx = i;
                f.add(ex.submit(() -> NodeProcess.spawn(idx, cell.jvmFlags(), nodeHeapMB(),
                        Path.of(cell.outDir(), "logs"), Duration.ofSeconds(120))));
            }
            for (Future<NodeProcess> x : f) {
                NodeProcess p = x.get();
                nodes.add(p);
                targets.add(p.target());
                metricsUrls.add(p.metricsUrl());
            }
        } finally {
            ex.shutdown();
        }
    }

    private void connect() {
        for (String t : targets) {
            NettyChannelBuilder b = NettyChannelBuilder.forTarget(t).usePlaintext()
                    .maxInboundMessageSize(options.maxMessageBytes())
                    .keepAliveTime(CoresetCoordinator.KEEPALIVE_TIME.toNanos(), TimeUnit.NANOSECONDS)
                    .keepAliveTimeout(CoresetCoordinator.KEEPALIVE_TIMEOUT.toNanos(), TimeUnit.NANOSECONDS);
            if (lossy != null) {
                b.intercept(lossy.client());
            }
            ManagedChannel ch = b.build();
            channels.add(ch);
            perNode.add(new CoresetCoordinator(List.of(ch), options));
        }
        all = new CoresetCoordinator(channels, options);
    }

    private void disconnect() {
        perNode.forEach(CoresetCoordinator::close);
        perNode.clear();
        if (all != null) {
            all.close();
            all = null;
        }
        channels.forEach(ManagedChannel::shutdownNow);
        channels.clear();
    }

    /** Respawns nodes killed by an earlier fault repetition; returns the seconds spent, 0 if none was dead. */
    private double ensureNodes() throws Exception {
        if (external || nodes.stream().allMatch(NodeProcess::alive)) {
            return 0.0;
        }
        long t0 = System.nanoTime();
        disconnect();
        for (int i = 0; i < nodes.size(); i++) {
            if (!nodes.get(i).alive()) {
                NodeProcess p = NodeProcess.spawn(i, cell.jvmFlags(), nodeHeapMB(), Path.of(cell.outDir(), "logs"),
                        Duration.ofSeconds(120));
                nodes.set(i, p);
                targets.set(i, p.target());
                metricsUrls.set(i, p.metricsUrl());
            }
        }
        connect();
        return (System.nanoTime() - t0) / 1e9;
    }

    @Override
    public Map<Long, String> roles() {
        Map<Long, String> r = new LinkedHashMap<>();
        for (NodeProcess p : nodes) {
            r.put(p.pid(), "node-" + p.index);
        }
        return r;
    }

    @Override
    public RunOutput run(int rep, long seed, FaultConfig fault) throws Exception {
        double respawn = ensureNodes();
        if (fault != null && !fault.kind().equals("kill-nodes")) {
            throw new IllegalArgumentException("grpc supports fault kind kill-nodes, not " + fault.kind());
        }
        if (fault != null && fault.count() >= n) {
            throw new IllegalArgumentException("kill-nodes needs a survivor: count " + fault.count() + " >= N " + n);
        }
        String session = cell.id() + "-r" + rep + "-" + UUID.randomUUID();
        SessionSpec spec = new SessionSpec(cell.algorithm().id(), cell.algorithm().params(), seed,
                sys.intOption("workersPerNode", 1), sys.intOption("queueCapacity", 64), TreeMode.SINGLE_WRITER);
        List<Map<String, Double>> before = scrape();
        all.createSession(session, spec);

        Ingest ingest = new Ingest(session, fault);
        long t0 = System.nanoTime();
        ingest.runShards();
        double ingestSeconds = (System.nanoTime() - t0) / 1e9;

        long g0 = System.nanoTime();
        List<Coreset> parts = new ArrayList<>();
        double realized = 0.0;
        boolean complete = true;
        for (int i = 0; i < n; i++) {
            if (ingest.dead.get(i) == 0) {
                try {
                    GatheredSnapshot s = perNode.get(i).gather(session, flushTimeout);
                    NodeSnapshot ns = s.nodes().get(0);
                    parts.add(ns.coreset());
                    realized += ns.appliedWeight();
                    complete &= s.complete();
                    continue;
                } catch (StatusRuntimeException e) {
                    ingest.markDead(i);
                }
            }
            NodeSnapshot saved = ingest.saved[i];
            if (saved != null) { // a dead node's last snapshot still summarises what it held then
                parts.add(saved.coreset());
                realized += saved.appliedWeight();
            }
        }
        Coreset union = Coreset.union(parts);
        double gatherSeconds = (System.nanoTime() - g0) / 1e9;
        double inputWeight = ingest.submittedWeight.sum() + ingest.droppedWeight.sum();
        boolean reweight = fault != null && fault.reweight() && realized > 0.0 && realized < inputWeight;
        Coreset out = reweight ? scale(union, inputWeight / realized) : union;

        for (int i = 0; i < n; i++) {
            if (ingest.dead.get(i) == 0) {
                try {
                    perNode.get(i).closeSession(session);
                } catch (StatusRuntimeException ignored) {
                    // node died after the gather
                }
            }
        }
        List<Map<String, Double>> after = scrape();

        Map<String, Object> m = EngineSystem.map();
        m.put("nodes", n);
        m.put("workers_per_node", spec.workers());
        m.put("batch_points", cell.batchSize());
        m.put("snapshot_complete", complete);
        m.put("respawn_s", respawn);
        putBytes(m, before, after);
        if (lossy != null) {
            m.put("lossy_seen", lossy.seen());
            m.put("lossy_dropped", lossy.dropped());
        }
        Map<String, Object> f = EngineSystem.map();
        if (fault != null) {
            f.putAll(ingest.faultInfo);
            f.put("input_weight", inputWeight);
            f.put("realized_weight", realized);
            f.put("lost_weight", inputWeight - realized);
            f.put("lost_fraction", (inputWeight - realized) / inputWeight);
            f.put("reweighted", reweight);
            f.put("survivors", n - ingest.deadCount());
        }
        return new RunOutput(out, null, ingestSeconds, gatherSeconds, Double.NaN, m, f);
    }

    private static Coreset scale(Coreset c, double s) {
        Coreset.Builder b = Coreset.builder(c.size());
        for (int i = 0; i < c.size(); i++) {
            b.add(c.pointSet().vector(i), c.pointSet().weight(i) * s);
        }
        return b.build(c.offset() * s);
    }

    /** One repetition's ingest: shard threads, fault trigger, dead-node bookkeeping. */
    private final class Ingest {
        final String session;
        final FaultConfig fault;
        final IngestStream[] streams = new IngestStream[n];
        final AtomicLongArray dead = new AtomicLongArray(n); // 0 = alive, else nanoTime declared dead
        final NodeSnapshot[] saved = new NodeSnapshot[n];
        final DoubleAdder submittedWeight = new DoubleAdder();
        final DoubleAdder droppedWeight = new DoubleAdder();
        final AtomicLong submittedPoints = new AtomicLong();
        final AtomicLong droppedRows = new AtomicLong();
        final AtomicLong reroutedBatches = new AtomicLong();
        final AtomicInteger rr = new AtomicInteger();
        final AtomicBoolean snapshotFired = new AtomicBoolean();
        final AtomicBoolean killFired = new AtomicBoolean();
        final Map<String, Object> faultInfo = java.util.Collections.synchronizedMap(new LinkedHashMap<>());
        final ExecutorService faultExec = Executors.newSingleThreadExecutor();
        final long killAt;
        final long snapAt;
        final Set<Integer> victims = new HashSet<>();
        volatile long killNanos;
        long start;

        Ingest(String session, FaultConfig fault) {
            this.session = session;
            this.fault = fault;
            for (int i = 0; i < n; i++) {
                // fixed producer ids: batch coresets depend on (seed, partition, producer, seq), so equal seeds replay
                streams[i] = perNode.get(i).openIngest(session, "shard-" + i);
            }
            long total = src.size();
            killAt = fault == null ? Long.MAX_VALUE : (long) (fault.at() * total);
            snapAt = fault == null || fault.snapshotAt() == null ? Long.MAX_VALUE : (long) (fault.snapshotAt() * total);
            if (fault != null) {
                for (int v = n - fault.count(); v < n; v++) {
                    victims.add(v);
                }
            }
        }

        int deadCount() {
            int c = 0;
            for (int i = 0; i < n; i++) {
                c += dead.get(i) != 0 ? 1 : 0;
            }
            return c;
        }

        void markDead(int i) {
            if (dead.compareAndSet(i, 0, System.nanoTime()) && killNanos != 0 && victims.contains(i)) {
                faultInfo.put("detect_s_node" + i, (dead.get(i) - killNanos) / 1e9);
            }
        }

        void runShards() throws Exception {
            start = System.nanoTime();
            ExecutorService ex = Executors.newFixedThreadPool(n, r -> new Thread(r, "grpc-shard"));
            try {
                List<Future<?>> f = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    final int shard = i;
                    f.add(ex.submit(() -> {
                        shard(shard);
                        return null;
                    }));
                }
                for (Future<?> x : f) {
                    x.get();
                }
                for (int i = 0; i < n; i++) {
                    if (dead.get(i) == 0) {
                        try {
                            streams[i].flush(flushTimeout);
                        } catch (IngestFailedException e) {
                            markDead(i);
                        } catch (TimeoutException e) {
                            throw new IllegalStateException("node " + i + ": " + e.getMessage()
                                    + (lossy != null ? " (lossy transport dropped " + lossy.dropped() + " messages)"
                                    : ""), e);
                        }
                    }
                    streams[i].close();
                }
                faultExec.shutdown();
                faultExec.awaitTermination(1, TimeUnit.MINUTES);
            } finally {
                ex.shutdownNow();
                faultExec.shutdownNow();
            }
            faultInfo.put("dropped_rows", droppedRows.get());
            faultInfo.put("rerouted_batches", reroutedBatches.get());
        }

        private void shard(int i) throws IOException, InterruptedException {
            long from = PointSource.shardStart(src.size(), i, n);
            long to = PointSource.shardStart(src.size(), i + 1, n);
            int target = i;
            boolean dropping = false;
            try (ChunkReader r = src.open(from, to)) {
                for (PointBlock b = r.next(cell.batchSize()); b != null; b = r.next(cell.batchSize())) {
                    if (dropping) { // the shard's owner is gone: account its remaining rows as lost input
                        droppedRows.addAndGet(b.size());
                        droppedWeight.add(b.totalWeight());
                        continue;
                    }
                    for (;;) {
                        if (dead.get(target) != 0) {
                            if (fault == null || fault.afterKill().equals("drop")) {
                                dropping = true;
                                droppedRows.addAndGet(b.size());
                                droppedWeight.add(b.totalWeight());
                                break;
                            }
                            target = survivor();
                            reroutedBatches.incrementAndGet();
                        }
                        try {
                            streams[target].submit(ChunkReader.asList(b));
                            submittedWeight.add(b.totalWeight());
                            break;
                        } catch (IngestFailedException e) {
                            markDead(target);
                        }
                    }
                    long done = submittedPoints.addAndGet(b.size());
                    if (done >= snapAt && snapshotFired.compareAndSet(false, true)) {
                        faultExec.submit(this::snapshotAll);
                    }
                    if (done >= killAt && killFired.compareAndSet(false, true)) {
                        faultExec.submit(this::killVictims);
                    }
                }
            }
        }

        private int survivor() {
            for (int tries = 0; tries < n; tries++) {
                int c = Math.floorMod(rr.getAndIncrement(), n);
                if (dead.get(c) == 0) {
                    return c;
                }
            }
            throw new IllegalStateException("no live node left to reroute to");
        }

        private void snapshotAll() {
            long s0 = System.nanoTime();
            for (int i = 0; i < n; i++) {
                try {
                    saved[i] = perNode.get(i).gather(session, Duration.ZERO).nodes().get(0);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                } catch (StatusRuntimeException e) {
                    markDead(i);
                }
            }
            faultInfo.put("snapshot_at_s", (s0 - start) / 1e9);
            faultInfo.put("snapshot_s", (System.nanoTime() - s0) / 1e9);
        }

        private void killVictims() {
            killNanos = System.nanoTime();
            faultInfo.put("kill_at_s", (killNanos - start) / 1e9);
            faultInfo.put("kill_at_points", submittedPoints.get());
            faultInfo.put("victims", new ArrayList<>(victims));
            for (int v : victims) {
                if (external) {
                    killExternal(targets.get(v));
                } else {
                    nodes.get(v).kill();
                }
            }
        }
    }

    private void killExternal(String target) {
        String template = cell.cluster().grpcKillCommand();
        if (template == null) {
            throw new IllegalStateException("kill-nodes on external targets needs cluster.grpcKillCommand");
        }
        int colon = target.lastIndexOf(':');
        String cmd = template.replace("{host}", target.substring(0, colon)).replace("{port}",
                target.substring(colon + 1));
        try {
            new ProcessBuilder("sh", "-c", cmd).inheritIO().start().waitFor(30, TimeUnit.SECONDS);
        } catch (IOException e) {
            throw new IllegalStateException("kill command failed: " + cmd, e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Per node (null if unreachable), serialized-message byte counters by method and direction. */
    private List<Map<String, Double>> scrape() {
        List<Map<String, Double>> out = new ArrayList<>();
        for (String url : metricsUrls) {
            try {
                String body = http.send(HttpRequest.newBuilder(URI.create(url)).timeout(Duration.ofSeconds(5)).build(),
                        HttpResponse.BodyHandlers.ofString()).body();
                Map<String, Double> node = new LinkedHashMap<>();
                Matcher m = BYTES.matcher(body);
                while (m.find()) {
                    node.merge(m.group(1) + "_" + m.group(2), Double.parseDouble(m.group(3)), Double::sum);
                }
                out.add(node);
            } catch (IOException e) {
                out.add(null); // a killed node has no counters any more
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                out.add(null);
            }
        }
        return out;
    }

    private void putBytes(Map<String, Object> m, List<Map<String, Double>> before, List<Map<String, Double>> after) {
        if (metricsUrls.isEmpty()) {
            m.put("bytes_available", false);
            return;
        }
        double ingest = 0.0;
        double snapshot = 0.0;
        double total = 0.0;
        boolean partial = false;
        for (int i = 0; i < before.size(); i++) {
            Map<String, Double> b = before.get(i);
            Map<String, Double> a = after.get(i);
            if (b == null || a == null) {
                partial = true; // counters of a node killed during the run are gone with it
                continue;
            }
            ingest += delta(b, a, "Ingest_in");
            snapshot += delta(b, a, "Snapshot_out");
            for (String k : a.keySet()) {
                total += delta(b, a, k);
            }
        }
        m.put("bytes_ingest", ingest);
        m.put("bytes_snapshot", snapshot);
        m.put("bytes_summary", snapshot);
        m.put("bytes_total", total);
        m.put("bytes_partial", partial);
    }

    private static double delta(Map<String, Double> before, Map<String, Double> after, String k) {
        return Math.max(0.0, after.getOrDefault(k, 0.0) - before.getOrDefault(k, 0.0));
    }

    @Override
    public void close() {
        disconnect();
        for (NodeProcess p : nodes) {
            try {
                p.stop();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
