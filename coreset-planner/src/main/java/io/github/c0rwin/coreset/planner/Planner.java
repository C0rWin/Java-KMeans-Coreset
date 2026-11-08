package io.github.c0rwin.coreset.planner;

import io.github.c0rwin.coreset.planner.Configuration.Family;
import io.github.c0rwin.coreset.planner.Configuration.Packetization;
import io.github.c0rwin.coreset.planner.Configuration.Redundancy;
import io.github.c0rwin.coreset.planner.Configuration.StragglerPolicy;
import io.github.c0rwin.coreset.planner.Configuration.Transport;
import io.github.c0rwin.coreset.planner.model.ResilienceModel;
import io.github.c0rwin.coreset.planner.model.ResilienceModel.Prediction;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Exhaustive search over configuration skeletons; for each, the least m whose predicted error (in the requirement's
 * mode) meets eps, found by bisection on m (the model is monotone in m). The cheapest skeleton wins; ties go to
 * fewer bytes, then less time, then enumeration order (per-draw before merged, no redundancy before redundancy).
 */
public final class Planner {

    /** Largest m searched: per-draw sampling may draw a point more than once, so m may exceed n. */
    public static final int MAX_M = 1 << 23;
    static final int[] SPLIT_BUDGETS = {4};
    static final int[][] FEC = {{16, 2}, {16, 4}, {8, 4}};
    static final int[] POST_R = {1, 2};
    static final int[] FR_R = {1, 2};
    static final int[] RATELESS_MU = {2, 4};
    static final int[] ARITIES = {2, 4, 8};

    private final ResilienceModel model;
    private final java.util.Set<Family> families;

    public Planner(ResilienceModel model) {
        this(model, java.util.EnumSet.allOf(Family.class));
    }

    /** A planner that only considers {@code families} (e.g. when the mean coreset is not deployable). */
    public Planner(ResilienceModel model, java.util.Set<Family> families) {
        if (families.isEmpty()) {
            throw new IllegalArgumentException("no coreset family to plan with");
        }
        this.model = model;
        this.families = java.util.EnumSet.copyOf(families);
    }

    public ResilienceModel model() {
        return model;
    }

    /** All skeletons (m = 1) the planner considers for this environment. */
    public List<Configuration> skeletons(EnvironmentProfile env) {
        List<Configuration> out = new ArrayList<>();
        int spare = Math.max(1, env.stragglers());
        for (Family f : families) {
            List<Packetization> packs = f.sampling()
                    ? List.of(Packetization.PER_DRAW, Packetization.MERGED, Packetization.SPLIT)
                    : List.of(Packetization.CELLS, Packetization.CELLS_CF);
            for (Packetization p : packs) {
                int[] budgets = p == Packetization.SPLIT ? SPLIT_BUDGETS : new int[] {0};
                for (int beta : budgets) {
                    for (Transport t : Transport.values()) {
                        if (!f.sampling() && t == Transport.DATAGRAM_OVERSAMPLE) {
                            continue; // cells are not draws: nothing to oversample
                        }
                        int[][] fecs = t == Transport.DATAGRAM_FEC ? FEC : new int[][] {{0, 0}};
                        for (int[] fec : fecs) {
                            for (int[] red : redundancies(spare)) {
                                for (int a : arities(f)) {
                                    for (boolean ck : a == 0 ? new boolean[] {false} : new boolean[] {false, true}) {
                                        for (StragglerPolicy sp : env.stragglers() > 0 && red[0] <= 1
                                                ? StragglerPolicy.values() : new StragglerPolicy[] {StragglerPolicy.WAIT}) {
                                            out.add(new Configuration(f, p, beta, t, fec[0], fec[1],
                                                    Redundancy.values()[red[0]], red[1], red[2], a, ck, sp, 1));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return out;
    }

    private static List<int[]> redundancies(int spare) {
        List<int[]> out = new ArrayList<>();
        out.add(new int[] {Redundancy.NONE.ordinal(), 0, 0});
        for (int r : POST_R) {
            out.add(new int[] {Redundancy.POST_REPLICATION.ordinal(), r, 0});
        }
        for (int r : FR_R) {
            out.add(new int[] {Redundancy.FRACTIONAL_REPETITION.ordinal(), r, 0});
        }
        for (int mu : RATELESS_MU) {
            out.add(new int[] {Redundancy.RATELESS.ordinal(), mu, spare});
        }
        return out;
    }

    private static int[] arities(Family f) {
        if (!f.sampling()) {
            return new int[] {0};
        }
        int[] out = new int[ARITIES.length + 1];
        System.arraycopy(ARITIES, 0, out, 1, ARITIES.length);
        return out;
    }

    /** Least feasible m for one skeleton, with its prediction; null prediction's reason explains exclusion. */
    public Evaluated evaluate(WorkloadProfile w, EnvironmentProfile e, Requirement r, Configuration skeleton) {
        int lo = Math.max(1, r.k());
        if (!skeleton.family().sampling()) {
            lo = Math.max(lo, w.mean().cells()[0]);
        }
        int hi = skeleton.family().sampling() ? MAX_M : (int) Math.min(MAX_M, Math.max(w.n(), lo));
        Prediction top = model.predict(w, e, r, skeleton.withM(hi));
        if (!top.admissible()) {
            return new Evaluated(skeleton.withM(hi), top, false, top.reason());
        }
        if (top.eps(r.mode()) > r.eps()) {
            return new Evaluated(skeleton.withM(hi), top, false, String.format(
                    "eps %s %.4g > %.4g even at m=%d (dominant term: %s)", r.mode().name().toLowerCase(),
                    top.eps(r.mode()), r.eps(), hi, dominant(top)));
        }
        while (lo < hi) {
            int mid = (int) (((long) lo + hi) >>> 1);
            Prediction p = model.predict(w, e, r, skeleton.withM(mid));
            if (p.admissible() && p.eps(r.mode()) <= r.eps()) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Configuration c = skeleton.withM(lo);
        return new Evaluated(c, model.predict(w, e, r, c), true, null);
    }

    /** A skeleton at its least feasible m (or at the largest m when infeasible). */
    public record Evaluated(Configuration config, Prediction prediction, boolean feasible, String reason) {
    }

    public Plan plan(WorkloadProfile w, EnvironmentProfile e, Requirement r) {
        List<Evaluated> all = new ArrayList<>();
        for (Configuration s : skeletons(e)) {
            all.add(evaluate(w, e, r, s));
        }
        Comparator<Evaluated> cheapest = Comparator
                .comparingDouble((Evaluated x) -> r.cost(x.prediction().bytes(), x.prediction().seconds()))
                .thenComparingDouble(x -> x.prediction().bytes())
                .thenComparingDouble(x -> x.prediction().seconds());
        List<Evaluated> feasible = all.stream().filter(Evaluated::feasible).sorted(cheapest).toList();

        Map<String, Evaluated> perGroup = new LinkedHashMap<>();
        for (Evaluated x : feasible) {
            perGroup.putIfAbsent(x.config().family() + "/" + x.config().packetization() + "/"
                    + x.config().transport(), x);
        }
        List<Plan.Alternative> alts = new ArrayList<>();
        for (Evaluated x : perGroup.values()) {
            alts.add(alternative(r, x, "feasible"));
        }

        if (feasible.isEmpty()) {
            Evaluated best = all.stream().filter(x -> x.prediction().admissible())
                    .min(Comparator.comparingDouble(x -> x.prediction().eps(r.mode()))).orElse(all.get(0));
            List<String> why = new ArrayList<>();
            why.add("no configuration meets eps=" + r.eps() + " in " + r.mode().name().toLowerCase()
                    + " mode; closest shown");
            why.addAll(exclusionSummary(all));
            return new Plan(false, best.config(), best.prediction(), r.cost(best.prediction().bytes(),
                    best.prediction().seconds()), satisfied(r, best.prediction()), why, alts, model.id(), all.size());
        }
        Evaluated win = feasible.get(0);
        List<String> binding = new ArrayList<>();
        Prediction p = win.prediction();
        binding.add(String.format("accuracy binds: m=%d is the least m with eps_%s=%.4g <= %.4g; worst candidate %s, "
                        + "dominant term %s", win.config().m(), r.mode().name().toLowerCase(), p.eps(r.mode()),
                r.eps(), p.worstCandidate(), dominant(p)));
        if (feasible.size() > 1) {
            Evaluated second = feasible.get(1);
            binding.add(String.format("runner-up %s costs %.4g vs %.4g", second.config().describe(),
                    r.cost(second.prediction().bytes(), second.prediction().seconds()),
                    r.cost(p.bytes(), p.seconds())));
        }
        binding.addAll(exclusionSummary(all));
        return new Plan(true, win.config(), p, r.cost(p.bytes(), p.seconds()), satisfied(r, p), binding, alts,
                model.id(), all.size());
    }

    /** Modes whose prediction meets eps (a calibrated plan may or may not also be certified). */
    static List<Requirement.Mode> satisfied(Requirement r, Prediction p) {
        List<Requirement.Mode> out = new ArrayList<>();
        for (Requirement.Mode m : Requirement.Mode.values()) {
            if (p.admissible() && p.eps(m) <= r.eps()) {
                out.add(m);
            }
        }
        return out;
    }

    private static Plan.Alternative alternative(Requirement r, Evaluated x, String status) {
        Prediction p = x.prediction();
        return new Plan.Alternative(x.config().describe(), p.epsPoint(), p.epsBound(), p.bytes(), p.seconds(),
                r.cost(p.bytes(), p.seconds()), status);
    }

    static String dominant(Prediction p) {
        return p.terms().entrySet().stream().max(Map.Entry.comparingByValue())
                .map(en -> String.format("%s (%.3g)", en.getKey(), en.getValue())).orElse("none");
    }

    /** Counts of excluded or unreachable skeletons, grouped by the rule that excluded them. */
    static List<String> exclusionSummary(List<Evaluated> all) {
        Map<String, Integer> counts = new TreeMap<>();
        for (Evaluated x : all) {
            if (!x.feasible()) {
                String reason = x.reason();
                String key = reason.startsWith("eps ") ? "accuracy unreachable within m<=" + MAX_M
                        : reason.replaceAll("[0-9.eE+-]{3,}", "#");
                counts.merge(key, 1, Integer::sum);
            }
        }
        List<String> out = new ArrayList<>();
        counts.forEach((k, v) -> out.add("excluded " + v + "x: " + k));
        return out;
    }
}
