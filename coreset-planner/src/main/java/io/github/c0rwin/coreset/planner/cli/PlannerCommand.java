package io.github.c0rwin.coreset.planner.cli;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.planner.Configuration;
import io.github.c0rwin.coreset.planner.EnvironmentProfile;
import io.github.c0rwin.coreset.planner.EnvironmentProfile.Partition;
import io.github.c0rwin.coreset.planner.Plan;
import io.github.c0rwin.coreset.planner.Planner;
import io.github.c0rwin.coreset.planner.Requirement;
import io.github.c0rwin.coreset.planner.Requirement.Guarantee;
import io.github.c0rwin.coreset.planner.Requirement.Mode;
import io.github.c0rwin.coreset.planner.Requirement.Objective;
import io.github.c0rwin.coreset.planner.WorkloadEstimator;
import io.github.c0rwin.coreset.planner.WorkloadProfile;
import io.github.c0rwin.coreset.planner.model.Calibration;
import io.github.c0rwin.coreset.planner.model.PaperResilienceModel;
import io.github.c0rwin.coreset.planner.validate.PlanSimulator;
import io.github.c0rwin.coreset.testbed.data.Conversion;
import io.github.c0rwin.coreset.testbed.data.PointFile;
import io.github.c0rwin.coreset.testbed.data.PublicDataset;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Callable;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Spec;

/** {@code coreset-planner plan|validate}: choose a resilient coreset configuration and check it in-process. */
@Command(name = "coreset-planner", mixinStandardHelpOptions = true,
        subcommands = {PlannerCommand.PlanCommand.class, PlannerCommand.ValidateCommand.class},
        description = "Plans coreset family, size, packetization, transport, redundancy and topology for a workload, "
                + "an environment and an accuracy requirement.")
public final class PlannerCommand implements Runnable {

    @Spec
    CommandSpec spec;

    public static void main(String[] args) {
        System.exit(new CommandLine(new PlannerCommand()).execute(args));
    }

    @Override
    public void run() {
        spec.commandLine().usage(spec.commandLine().getOut());
    }

    /** Options shared by both subcommands. */
    static final class Common {
        @Option(names = "--dataset", required = true, description = "public dataset id or a .cbin/.csr point file")
        String dataset;
        @Option(names = "--variant", defaultValue = "zscore", description = "raw, zscore or tfidf (public ids)")
        String variant;
        @Option(names = "--cache-dir", description = "dataset cache (default ~/.cache/coreset-testbed)")
        Path cacheDir;
        @Option(names = "--k", defaultValue = "10")
        int k;
        @Option(names = "--pilot", defaultValue = "" + WorkloadEstimator.DEFAULT_PILOT)
        int pilot;
        @Option(names = "--rho", defaultValue = "0", description = "packet erasure rate")
        double rho;
        @Option(names = "--bursty", defaultValue = "1", description = "mean erasure burst length in packets")
        double bursty;
        @Option(names = "--nodes", defaultValue = "16")
        int nodes;
        @Option(names = "--fail", defaultValue = "0", description = "probability that a node is down at upload")
        double fail;
        @Option(names = "--fail-phase", defaultValue = "post", description = "pre, post or both")
        String failPhase;
        @Option(names = "--partition", defaultValue = "random", description = "random or adversarial")
        String partition;
        @Option(names = "--stragglers", defaultValue = "0")
        int stragglers;
        @Option(names = "--straggler-delay", defaultValue = "1.0", description = "seconds")
        double stragglerDelay;
        @Option(names = "--rtt", defaultValue = "0.0005", description = "seconds")
        double rtt;
        @Option(names = "--bandwidth", defaultValue = "125e6", description = "bytes per second")
        double bandwidth;
        @Option(names = "--mtu", defaultValue = "1400", description = "largest datagram, headers included")
        int mtu;
        @Option(names = "--pre-replicated", description = "input already stored on r+1 nodes")
        boolean preReplicated;
        @Option(names = "--eps", defaultValue = "0.05")
        double eps;
        @Option(names = "--delta", defaultValue = "0.05")
        double delta;
        @Option(names = "--objective", defaultValue = "bytes", description = "bytes, time or weighted")
        String objective;
        @Option(names = "--mode", defaultValue = "certified", description = "certified or calibrated")
        String mode;
        @Option(names = "--guarantee", defaultValue = "instance", description = "instance or uniform")
        String guarantee;
        @Option(names = "--calibration", description = "calibration JSON (default: bundled)")
        Path calibration;
        @Option(names = "--families", split = ",", defaultValue = "uniform,sensitivity,lightweight,mean",
                description = "coreset families to consider")
        java.util.List<String> families;
        @Option(names = "--json", description = "also write the result as JSON to this file")
        Path json;
        @Option(names = "--seed", defaultValue = "42")
        long seed;

        Path file() throws IOException, InterruptedException {
            Path p = Path.of(dataset);
            if (Files.isRegularFile(p)) {
                return p;
            }
            Path cache = cacheDir != null ? cacheDir : PublicDataset.defaultCacheDir();
            return Conversion.ensure(PublicDataset.named(dataset), cache, Conversion.Variant.of(variant), false);
        }

        EnvironmentProfile env() {
            double pre = 0.0;
            double post = 0.0;
            switch (failPhase.toLowerCase(Locale.ROOT)) {
                case "pre" -> pre = fail;
                case "post" -> post = fail;
                case "both" -> {
                    pre = 1.0 - Math.sqrt(1.0 - fail);
                    post = pre;
                }
                default -> throw new IllegalArgumentException("--fail-phase must be pre, post or both");
            }
            return new EnvironmentProfile(rho, bursty, nodes, pre, post,
                    Partition.valueOf(partition.toUpperCase(Locale.ROOT)), stragglers, stragglerDelay, rtt, bandwidth,
                    mtu, preReplicated);
        }

        Requirement requirement() {
            Objective o = switch (objective.toLowerCase(Locale.ROOT)) {
                case "bytes" -> Objective.MIN_BYTES;
                case "time" -> Objective.MIN_TIME;
                case "weighted" -> Objective.WEIGHTED;
                default -> throw new IllegalArgumentException("--objective must be bytes, time or weighted");
            };
            return new Requirement(eps, delta, k, Guarantee.valueOf(guarantee.toUpperCase(Locale.ROOT)),
                    Mode.valueOf(mode.toUpperCase(Locale.ROOT)), o, 1.0, 1.0);
        }

        Planner planner() throws IOException {
            Calibration cal = calibration == null ? Calibration.bundled() : Calibration.load(calibration);
            java.util.Set<Configuration.Family> fams = java.util.EnumSet.noneOf(Configuration.Family.class);
            for (String f : families) {
                fams.add(Configuration.Family.valueOf(f.trim().toUpperCase(Locale.ROOT)));
            }
            return new Planner(new PaperResilienceModel(cal), fams);
        }

        WorkloadProfile profile(Path file) throws IOException {
            return new WorkloadEstimator(pilot, k, seed, true).estimate(dataset, file);
        }
    }

    static ObjectMapper mapper() {
        return new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
    }

    static void print(PrintWriter out, Plan plan, WorkloadProfile w) {
        out.printf(Locale.ROOT, "workload %s: n=%d d=%d density=%.3f pilot=%d k=%d balancedness=%.3f%n",
                w.datasetId(), w.n(), w.d(), w.density(), w.pilotSize(), w.k(), w.balancedness());
        out.printf(Locale.ROOT, "plan (%s): %s%n", plan.feasible() ? "feasible" : "INFEASIBLE",
                plan.config().describe());
        var p = plan.prediction();
        out.printf(Locale.ROOT, "  eps point %.4f, bound %.4f (worst: %s); satisfies %s%n", p.epsPoint(), p.epsBound(),
                p.worstCandidate(), plan.satisfies());
        out.printf(Locale.ROOT, "  bytes %.4g, time %.4g s, records %d, received draws >= %d, erasure %.3g, "
                + "lost fraction <= %.3g%n", p.bytes(), p.seconds(), p.recordsSent(), p.receivedDraws(),
                p.effectiveErasure(), p.lostFraction());
        for (String b : plan.binding()) {
            out.println("  - " + b);
        }
        for (String n : p.notes()) {
            out.println("  note: " + n);
        }
        out.println("  alternatives (cheapest per family/packetization/transport):");
        plan.alternatives().stream().limit(12).forEach(a -> out.printf(Locale.ROOT,
                "    %-90s eps %.4f/%.4f bytes %.4g time %.4g%n", a.config(), a.epsPoint(), a.epsBound(), a.bytes(),
                a.seconds()));
        out.println("  model " + plan.model() + ", " + plan.evaluated() + " skeletons");
    }

    @Command(name = "plan", mixinStandardHelpOptions = true, description = "Print the cheapest feasible plan")
    static final class PlanCommand implements Callable<Integer> {
        @Mixin
        Common o = new Common();
        @Spec
        CommandSpec spec;

        @Override
        public Integer call() throws IOException, InterruptedException {
            PrintWriter out = spec.commandLine().getOut();
            Path file = o.file();
            WorkloadProfile w = o.profile(file);
            Plan plan = o.planner().plan(w, o.env(), o.requirement());
            print(out, plan, w);
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("workload", w);
            doc.put("environment", o.env());
            doc.put("requirement", o.requirement());
            doc.put("plan", plan);
            String json = mapper().writeValueAsString(doc);
            out.println(json);
            if (o.json != null) {
                Files.writeString(o.json, json);
            }
            out.flush();
            return plan.feasible() ? 0 : 2;
        }
    }

    @Command(name = "validate", mixinStandardHelpOptions = true,
            description = "Plan, then run the plan in-process with injected faults; report predicted vs measured eps")
    static final class ValidateCommand implements Callable<Integer> {
        @Mixin
        Common o = new Common();
        @Option(names = "--trials", defaultValue = "20")
        int trials;
        @Spec
        CommandSpec spec;

        @Override
        public Integer call() throws IOException, InterruptedException {
            PrintWriter out = spec.commandLine().getOut();
            Path file = o.file();
            WorkloadProfile w = o.profile(file);
            Plan plan = o.planner().plan(w, o.env(), o.requirement());
            print(out, plan, w);
            PointSet data = PointFile.readAll(file);
            PlanSimulator sim = new PlanSimulator(data, w.tables().centers());
            long t0 = System.nanoTime();
            PlanSimulator.Result res = sim.run(plan.config(), plan.prediction(), o.env(), o.requirement(), trials,
                    o.seed + 1);
            double secs = (System.nanoTime() - t0) / 1e9;
            out.printf(Locale.ROOT, "validate: %d trials in %.1f s%n", trials, secs);
            out.printf(Locale.ROOT, "  predicted eps point %.4f, bound %.4f%n", plan.prediction().epsPoint(),
                    plan.prediction().epsBound());
            out.printf(Locale.ROOT, "  measured  (1-delta) quantile %.4f, mean %.4f, max %.4f%n", res.quantile(),
                    res.mean(), res.max());
            out.printf(Locale.ROOT, "  measured lost fraction %.4f, packet erasure %.4f%n", res.meanLostFraction(),
                    res.meanErasure());
            out.printf(Locale.ROOT, "  measured/point %.3f, measured <= bound: %s%n",
                    res.quantile() / plan.prediction().epsPoint(), res.quantile() <= plan.prediction().epsBound());
            Map<String, Object> doc = new LinkedHashMap<>();
            doc.put("plan", plan);
            doc.put("measured", res);
            String json = mapper().writeValueAsString(doc);
            out.println(json);
            if (o.json != null) {
                Files.writeString(o.json, json);
            }
            out.flush();
            return 0;
        }
    }
}
