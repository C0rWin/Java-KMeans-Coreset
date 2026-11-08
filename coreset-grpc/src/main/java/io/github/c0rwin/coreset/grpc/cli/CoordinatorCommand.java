package io.github.c0rwin.coreset.grpc.cli;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithms;
import io.github.c0rwin.coreset.core.Cost;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.kmeans.KMeansResult;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.core.validate.Candidate;
import io.github.c0rwin.coreset.core.validate.CandidateSolutions;
import io.github.c0rwin.coreset.core.validate.Distortion;
import io.github.c0rwin.coreset.grpc.ClientSecurity;
import io.github.c0rwin.coreset.grpc.CoordinatorOptions;
import io.github.c0rwin.coreset.grpc.CoresetCoordinator;
import io.github.c0rwin.coreset.grpc.GatheredSnapshot;
import io.github.c0rwin.coreset.grpc.IngestStream;
import io.github.c0rwin.coreset.grpc.StatusText;
import io.github.c0rwin.coreset.grpc.proto.ReplicateResponse;
import io.github.c0rwin.coreset.grpc.resilience.Census;
import io.github.c0rwin.coreset.grpc.resilience.ChannelModel;
import io.github.c0rwin.coreset.grpc.resilience.Decoder;
import io.github.c0rwin.coreset.grpc.resilience.GroupLayout;
import io.github.c0rwin.coreset.grpc.resilience.LossBounds;
import io.github.c0rwin.coreset.grpc.resilience.LossyTransport;
import io.github.c0rwin.coreset.grpc.resilience.ResilienceOptions;
import io.github.c0rwin.coreset.grpc.resilience.ResilientSnapshot;
import io.github.c0rwin.coreset.grpc.resilience.SurvivorReweighting;
import io.github.c0rwin.coreset.node.MessageSpec;
import io.github.c0rwin.coreset.node.Packetization;
import io.github.c0rwin.coreset.node.SessionSpec;
import io.github.c0rwin.coreset.node.TreeMode;
import io.grpc.StatusRuntimeException;
import java.io.BufferedReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.UUID;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Spec;

/**
 * {@code coreset-coordinator}: streams a points file to the nodes, gathers their coresets, clusters the union with
 * k-means and reports the centers and the distributed coreset cost.
 */
@Command(name = "coreset-coordinator", mixinStandardHelpOptions = true,
        description = "Distributes a points file over coreset nodes and clusters the gathered coreset.")
public final class CoordinatorCommand implements Callable<Integer> {

    @Spec
    CommandLine.Model.CommandSpec spec;

    @Option(names = "--nodes", required = true, split = ",", description = "Node targets host:port, comma separated.")
    List<String> nodes;

    @Option(names = "--input", required = true, description = "Points file (dense CSV/whitespace, or sparse).")
    Path input;

    @Option(names = "--sparse-dim", defaultValue = "0",
            description = "Read index:value lines of this dimension; 0 = dense lines. Default: ${DEFAULT-VALUE}")
    int sparseDim;

    @Option(names = "--algorithm", required = true, description = "Node-side coreset algorithm id.")
    String algorithm;

    @Option(names = "--param", description = "Algorithm parameter key=value (repeatable).")
    Map<String, String> params = new LinkedHashMap<>();

    @Option(names = "--session", description = "Session id. Default: random UUID.")
    String session;

    @Option(names = "--seed", defaultValue = "1", description = "Session and clustering seed. Default: ${DEFAULT-VALUE}")
    long seed;

    @Option(names = "--workers", defaultValue = "4", description = "Workers per node. Default: ${DEFAULT-VALUE}")
    int workers;

    @Option(names = "--queue-capacity", defaultValue = "64",
            description = "Per-worker queue capacity (power of two). Default: ${DEFAULT-VALUE}")
    int queueCapacity;

    @Option(names = "--tree-mode", defaultValue = "single-writer",
            description = "single-writer or cas. Default: ${DEFAULT-VALUE}")
    String treeMode;

    @Option(names = "--batch-size", defaultValue = "1000", description = "Points per batch. Default: ${DEFAULT-VALUE}")
    int batchSize;

    @Option(names = "--max-in-flight", defaultValue = "16",
            description = "Unacknowledged batches per node. Default: ${DEFAULT-VALUE}")
    int maxInFlight;

    @Option(names = "--k", required = true, description = "Number of centers.")
    int k;

    @Option(names = "--iterations", defaultValue = "50", description = "Lloyd iterations. Default: ${DEFAULT-VALUE}")
    int iterations;

    @Option(names = "--restarts", defaultValue = "10",
            description = "k-means++ and Lloyd restarts; the cheapest is kept. Default: ${DEFAULT-VALUE}")
    int restarts;

    @Option(names = "--final-algorithm", description = "Compress the gathered union once more with this algorithm.")
    String finalAlgorithm;

    @Option(names = "--final-param", description = "Final algorithm parameter key=value (repeatable).")
    Map<String, String> finalParams = new LinkedHashMap<>();

    @Option(names = "--timeout-ms", defaultValue = "600000",
            description = "Bound on flushing ingest and on awaiting applied batches. Default: ${DEFAULT-VALUE}")
    long timeoutMs;

    @Option(names = "--keep-session", description = "Leave the session open on the nodes.")
    boolean keepSession;

    @Option(names = "--producer-id",
            description = "Producer id of this run's batches; fixed ids make runs reproducible. Default: random.")
    String producerId;

    @Option(names = "--tls-ca", description = "PEM CAs trusted for node certificates; enables TLS.")
    Path tlsCa;

    @Option(names = "--tls-cert", description = "PEM client certificate chain for mutual TLS (with --tls-key).")
    Path tlsCert;

    @Option(names = "--tls-key", description = "PEM PKCS#8 private key of --tls-cert.")
    Path tlsKey;

    @Option(names = "--auth-token-file", description = "File holding the nodes' shared auth token.")
    Path authTokenFile;

    @Option(names = "--gather", defaultValue = "classic",
            description = "classic (unary snapshots, any node failure fails the run) or resilient (packet streams, "
                    + "decoders, replicas, survivor reweighting). Default: ${DEFAULT-VALUE}")
    String gatherMode;

    @Option(names = "--packetization", defaultValue = "summary",
            description = "summary, merged, per-draw, mean-global or mean-cf. Default: ${DEFAULT-VALUE}")
    String packetization;

    @Option(names = "--message-size", defaultValue = "0",
            description = "Draws (merged, per-draw) or cells (mean-*) per node message. Default: ${DEFAULT-VALUE}")
    int messageSize;

    @Option(names = "--message-k", defaultValue = "0",
            description = "Seeding centers of sampled messages (0 = --k). Default: ${DEFAULT-VALUE}")
    int messageK;

    @Option(names = "--records-per-packet", defaultValue = "16", description = "B. Default: ${DEFAULT-VALUE}")
    int recordsPerPacket;

    @Option(names = "--message-lloyd-iterations", defaultValue = "10",
            description = "Lloyd steps of the message seeding or cell partition. Default: ${DEFAULT-VALUE}")
    int messageLloydIterations;

    @Option(names = "--layout", defaultValue = "independent",
            description = "independent, fr (fractional repetition) or rateless. Default: ${DEFAULT-VALUE}")
    String layoutMode;

    @Option(names = "--group-size", defaultValue = "2",
            description = "Members per group: fr tolerates group-size - 1 stragglers. Default: ${DEFAULT-VALUE}")
    int groupSize;

    @Option(names = "--decoder", defaultValue = "realized",
            description = "as-sent, ht, realized or realized-weight. Default: ${DEFAULT-VALUE}")
    String decoder;

    @Option(names = "--known-loss", defaultValue = "0", description = "Loss rate assumed by --decoder ht.")
    double knownLoss;

    @Option(names = "--reweight", defaultValue = "weight",
            description = "Survivor reweighting: none, nodes or weight (census). Default: ${DEFAULT-VALUE}")
    String reweight;

    @Option(names = "--replicate", defaultValue = "0",
            description = "Store each node's frozen message on this many peers before gathering. Default: "
                    + "${DEFAULT-VALUE}")
    int replicate;

    @Option(names = "--peer-targets", split = ",",
            description = "Node addresses as the nodes reach each other (default: --nodes).")
    List<String> peerTargets;

    @Option(names = "--loss", defaultValue = "none",
            description = "Packet erasures on snapshot streams: none, bernoulli:RHO, gilbert:RHO,BURST or "
                    + "ge:G2B,B2G,LOSS_GOOD,LOSS_BAD. Default: ${DEFAULT-VALUE}")
    String loss;

    @Option(names = "--loss-seed", defaultValue = "7",
            description = "Seed of the erasure channel, independent of --seed. Default: ${DEFAULT-VALUE}")
    long lossSeed;

    @Option(names = "--gather-deadline-ms", defaultValue = "30000",
            description = "Deadline of one message fetch; slower nodes count as stragglers. Default: ${DEFAULT-VALUE}")
    long gatherDeadlineMs;

    @Option(names = "--min-responders", defaultValue = "0",
            description = "Proceed once every group has this many complete messages (0 = wait). Default: "
                    + "${DEFAULT-VALUE}")
    int minResponders;

    @Option(names = "--repeat", defaultValue = "1",
            description = "Resilient gathers to run, each on fresh sampling streams. Default: ${DEFAULT-VALUE}")
    int repeat;

    @Option(names = "--distortion-queries", defaultValue = "0",
            description = "Measure each gathered coreset's distortion on this many k-means++ center sets of the "
                    + "input (keeps the input in memory). Default: ${DEFAULT-VALUE}")
    int distortionQueries;

    @Option(names = "--eta", defaultValue = "0.05", description = "eta of the cor:loss-eps band. Default: "
            + "${DEFAULT-VALUE}")
    double eta;

    @Option(names = "--fault-barrier",
            description = "After ingest, census, replication and one pre-fault gather print 'FAULT_BARRIER' and wait "
                    + "for this file (inject faults meanwhile).")
    Path faultBarrier;

    @Override
    public Integer call() throws Exception {
        PrintWriter out = spec.commandLine().getOut();
        CoresetAlgorithm finalCompress = finalAlgorithm == null ? null
                : CoresetAlgorithms.load().create(finalAlgorithm, finalParams);
        String sessionId = session != null ? session : UUID.randomUUID().toString();
        boolean resilient = switch (gatherMode) {
            case "classic" -> false;
            case "resilient" -> true;
            default -> throw new CommandLine.ParameterException(spec.commandLine(),
                    "--gather must be classic or resilient: " + gatherMode);
        };
        Packetization form = Packetization.fromId(packetization);
        MessageSpec message = form == Packetization.SUMMARY
                ? new MessageSpec(form, 0, 0, recordsPerPacket, 0)
                : new MessageSpec(form, messageSize, form.sampled() ? (messageK > 0 ? messageK : k) : 0,
                        recordsPerPacket, messageLloydIterations);
        SessionSpec sessionSpec = new SessionSpec(algorithm, params, seed, workers, queueCapacity,
                TreeMode.fromId(treeMode), message);
        GroupLayout layout = switch (GroupLayout.Mode.fromId(layoutMode)) {
            case INDEPENDENT -> GroupLayout.independent(nodes.size());
            case FRACTIONAL_REPETITION -> GroupLayout.fractionalRepetition(nodes.size(), groupSize - 1);
            case RATELESS -> GroupLayout.rateless(nodes.size(), groupSize);
        };
        ChannelModel channel = ChannelModel.parse(loss);
        LossyTransport lossy = channel instanceof ChannelModel.Perfect ? null : new LossyTransport(channel, lossSeed);
        if (!resilient && (lossy != null || replicate > 0 || layout.mode() != GroupLayout.Mode.INDEPENDENT)) {
            throw new CommandLine.ParameterException(spec.commandLine(),
                    "--loss, --replicate and --layout need --gather resilient");
        }
        Duration timeout = Duration.ofMillis(timeoutMs);
        CoordinatorOptions options = CoordinatorOptions.defaults().withMaxInFlightPerNode(maxInFlight);
        String token = authTokenFile == null ? null : Files.readString(authTokenFile, StandardCharsets.UTF_8).strip();
        ClientSecurity security = new ClientSecurity(tlsCa, tlsCert, tlsKey, token);
        try (CoresetCoordinator coordinator = CoresetCoordinator.connect(nodes, options, security, MetricsSink.NOOP,
                lossy == null ? List.of() : List.of(lossy));
             SessionLease ignored = SessionLease.create(coordinator, sessionId, sessionSpec, layout, keepSession,
                     resilient ? spec.commandLine().getErr() : null)) {
            List<WeightedPoint> kept = resilient && distortionQueries > 0 ? new ArrayList<>() : null;
            long points = 0;
            long batches;
            try (IngestStream stream = coordinator.openIngest(sessionId,
                    producerId == null ? UUID.randomUUID().toString() : producerId, layout);
                 BufferedReader reader = Files.newBufferedReader(input, StandardCharsets.UTF_8)) {
                PointsReader pr = new PointsReader(reader, sparseDim);
                for (List<WeightedPoint> batch = pr.next(batchSize); !batch.isEmpty(); batch = pr.next(batchSize)) {
                    stream.submit(batch);
                    points += batch.size();
                    if (kept != null) {
                        kept.addAll(batch);
                    }
                }
                stream.flush(timeout);
                batches = stream.acceptedBatches();
            }
            if (resilient) {
                return resilient(out, coordinator, sessionId, layout, form, channel, lossy, kept, points, batches,
                        timeout, finalCompress);
            }
            GatheredSnapshot snapshot = coordinator.gather(sessionId, timeout);
            if (!snapshot.complete()) {
                spec.commandLine().getErr().printf("warning: snapshot incomplete (%d of %d batches applied)%n",
                        snapshot.appliedBatches(), snapshot.acceptedBatches());
            }
            KMeansResult result = CoresetCoordinator.cluster(snapshot.coreset(), k, iterations, restarts, seed,
                    finalCompress);
            double cost = coordinator.cost(sessionId, result.centers());
            out.printf("# session=%s points=%d batches=%d coreset_points=%d cost=%s%n", sessionId, points,
                    snapshot.acceptedBatches(), snapshot.coreset().size(), cost);
            printCenters(out, result.centers());
        }
        return 0;
    }

    /** Census, optional replication and fault barrier, then {@code --repeat} resilient gathers with reports. */
    private int resilient(PrintWriter out, CoresetCoordinator coordinator, String sessionId, GroupLayout layout,
                          Packetization form, ChannelModel channel, LossyTransport lossy, List<WeightedPoint> kept,
                          long points, long batches, Duration timeout, CoresetAlgorithm finalCompress)
            throws Exception {
        Census census = coordinator.census(sessionId, layout, timeout);
        out.printf("# session=%s points=%d batches=%d layout=%s packetization=%s census_weight=%s%n", sessionId,
                points, batches, layout, form.id(), Arrays.toString(census.nodeWeight()));
        if (replicate > 0) {
            List<ReplicateResponse> rs = coordinator.replicate(sessionId, replicate,
                    peerTargets == null ? nodes : peerTargets, layout, timeout);
            for (int i = 0; i < rs.size(); i++) {
                out.printf("# replicate node=%d records=%d stored=%d failed=%s%n", i, rs.get(i).getRecords(),
                        rs.get(i).getStored(), rs.get(i).getFailedPeersList());
            }
        }
        ResilienceOptions ro = new ResilienceOptions(layout, Decoder.fromId(decoder), knownLoss,
                SurvivorReweighting.fromId(reweight), census, replicate, Duration.ZERO,
                Duration.ofMillis(gatherDeadlineMs), minResponders, 0);
        Coreset preFault = null;
        if (faultBarrier != null) {
            preFault = coordinator.gatherResilient(sessionId, ro).coreset();
            out.printf("# pre-fault gather coreset_points=%d weight=%.3f%n", preFault.size(), preFault.totalWeight());
            out.printf("FAULT_BARRIER waiting for %s%n", faultBarrier);
            out.flush();
            long deadline = System.nanoTime() + timeout.toNanos();
            while (!Files.exists(faultBarrier)) {
                if (System.nanoTime() > deadline) {
                    throw new IllegalStateException("fault barrier " + faultBarrier + " not released in " + timeout);
                }
                Thread.sleep(50);
            }
        }
        printBand(out, form, channel);
        Coreset full = kept == null ? null : Coreset.of(kept);
        List<Candidate> candidates = full == null ? null
                : CandidateSolutions.kMeansPlusPlusOnFull(full, k, distortionQueries, new SplittableRandom(seed + 1));
        double[] distortions = new double[repeat];
        ResilientSnapshot last = null;
        for (int rep = 0; rep < repeat; rep++) {
            long dropped = lossy == null ? 0 : lossy.dropped();
            long delivered = lossy == null ? 0 : lossy.delivered();
            long start = System.nanoTime();
            last = coordinator.gatherResilient(sessionId, ro.withStreamOffset((long) rep * layout.nodes()));
            long micros = (System.nanoTime() - start) / 1000;
            long d = lossy == null ? 0 : lossy.dropped() - dropped;
            long a = lossy == null ? 0 : lossy.delivered() - delivered;
            String distortion = "-";
            String ratios = "-";
            if (candidates != null) {
                Distortion.Report report = Distortion.evaluate(full, last.coreset(), candidates);
                distortions[rep] = report.max() - 1.0;
                distortion = Double.toString(distortions[rep]);
                double lo = Double.POSITIVE_INFINITY;
                double hi = 0.0;
                for (Distortion.Entry e : report.entries()) {
                    lo = Math.min(lo, e.ratio());
                    hi = Math.max(hi, e.ratio());
                }
                ratios = String.format("[%.4f,%.4f]", lo, hi); // cost(C, Q) / cost(P, Q) over the queries
            }
            out.printf("# gather=%d lost_groups=%d survivor_scale=%.6f packets=%d/%d dropped=%d observed_loss=%.4f "
                            + "coreset_points=%d weight=%.3f offset=%.6g distortion=%s cost_ratio=%s micros=%d%n",
                    rep, last.lostGroups(), last.survivorScale(), last.packetsReceived(), last.packetsSent(), d,
                    a + d == 0 ? 0.0 : (double) d / (a + d), last.coreset().size(), last.coreset().totalWeight(),
                    last.coreset().offset(), distortion, ratios, micros);
            if (rep == 0) {
                if (preFault != null) {
                    out.printf("#   identical_to_pre_fault=%s%n", preFault.equals(last.coreset()));
                }
                for (ResilientSnapshot.GroupOutcome g : last.groups()) {
                    out.printf("#   group=%d delivered=%s weight=%s decoder_scale=%.6f records=%d/%d sources=%s%n",
                            g.group(), g.delivered(), g.nodeWeight(), g.decoderScale(), g.recordsReceived(),
                            g.recordsSent(), g.sources());
                }
            }
        }
        if (candidates != null && repeat > 1) {
            double[] sorted = distortions.clone();
            Arrays.sort(sorted);
            double mean = 0.0;
            for (double x : sorted) {
                mean += x / sorted.length;
            }
            out.printf("# distortion over %d gathers: mean=%.6f median=%.6f p90=%.6f max=%.6f%n", repeat, mean,
                    sorted[sorted.length / 2], sorted[(int) Math.ceil(0.9 * sorted.length) - 1],
                    sorted[sorted.length - 1]);
        }
        KMeansResult result = CoresetCoordinator.cluster(last.coreset(), k, iterations, restarts, seed, finalCompress);
        out.printf("# coreset_cost=%s full_cost=%s%n", Cost.cost(last.coreset(), result.centers()),
                full == null ? "-" : Double.toString(Cost.cost(full, result.centers())));
        printCenters(out, result.centers());
        return 0;
    }

    /** cor:loss-eps prediction for per-draw messages over a memoryless or Gilbert channel. */
    private void printBand(PrintWriter out, Packetization form, ChannelModel channel) {
        if (form != Packetization.PER_DRAW || messageSize < 1 || channel instanceof ChannelModel.Perfect) {
            return;
        }
        long mEta;
        if (channel instanceof ChannelModel.Bernoulli b && messageSize % recordsPerPacket == 0) {
            mEta = LossBounds.memorylessDraws(messageSize, b.loss(), eta, recordsPerPacket);
        } else if (channel instanceof ChannelModel.GilbertElliott ge && ge.lossGood() == 0.0 && ge.lossBad() == 1.0
                && recordsPerPacket == 1) {
            mEta = LossBounds.gilbertDraws(messageSize, ge.lossRate(), ge.lambda(), eta);
        } else {
            return;
        }
        out.printf("# cor:loss-eps m=%d rho=%.4f B=%d eta=%s m_eta=%d inflation=%.6f%n", messageSize,
                channel.lossRate(), recordsPerPacket, eta, mEta, LossBounds.inflation(messageSize, mEta));
    }

    private static void printCenters(PrintWriter out, List<SparseVector> centers) {
        for (SparseVector c : centers) {
            double[] d = c.toDense();
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < d.length; i++) {
                if (i > 0) {
                    sb.append(',');
                }
                sb.append(d[i]);
            }
            out.println(sb);
        }
        out.flush();
    }

    public static void main(String[] args) {
        System.exit(new CommandLine(new CoordinatorCommand()).execute(args));
    }

    /**
     * Creates the session and closes it again on every exit path, unless it is to be kept. With {@code warn} set, a
     * failed close (a crashed node) is reported there instead of failing the run.
     */
    private record SessionLease(CoresetCoordinator coordinator, String sessionId, boolean keep, PrintWriter warn)
            implements AutoCloseable {

        static SessionLease create(CoresetCoordinator coordinator, String sessionId, SessionSpec spec,
                                   GroupLayout layout, boolean keep, PrintWriter warn) throws InterruptedException {
            try {
                coordinator.createSession(sessionId, spec, layout);
            } catch (RuntimeException e) {
                if (!keep) { // some nodes may have created it before another refused
                    try {
                        coordinator.closeSession(sessionId);
                    } catch (RuntimeException suppressed) {
                        e.addSuppressed(suppressed);
                    }
                }
                throw e;
            }
            return new SessionLease(coordinator, sessionId, keep, warn);
        }

        @Override
        public void close() throws InterruptedException {
            if (keep) {
                return;
            }
            try {
                coordinator.closeSession(sessionId);
            } catch (StatusRuntimeException e) {
                if (warn == null) {
                    throw e;
                }
                warn.printf("warning: closing the session failed on some node: %s%n", StatusText.of(e));
                warn.flush();
            }
        }
    }
}
