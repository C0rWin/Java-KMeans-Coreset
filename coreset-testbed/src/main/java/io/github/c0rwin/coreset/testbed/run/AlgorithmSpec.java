package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.core.algo.Algorithms;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * One algorithm of the grid, resolved through {@link io.github.c0rwin.coreset.core.CoresetAlgorithms}, so any
 * registered id works, built-in or from a provider jar on the class path.
 *
 * @param label    name in reports and tables; defaults to the id plus its fixed parameters
 * @param id       registered algorithm id, e.g. {@code uniform}, {@code sensitivity}, {@code mean}
 * @param params   fixed parameters
 * @param inject   grid parameters added per run: {@code k} and/or {@code m}; defaults to what the built-in ids take
 * @param oneSided whether the algorithm never underestimates cost (Barger-Feldman mean coreset), so
 *                 {@code integrity.oneSidedViolations} applies; defaults to {@code id == mean}
 */
public record AlgorithmSpec(String label, String id, Map<String, String> params, List<String> inject,
                            Boolean oneSided) {

    public AlgorithmSpec {
        Objects.requireNonNull(id, "algorithm id");
        params = params == null ? Map.of() : Map.copyOf(new TreeMap<>(params));
        if (inject == null) {
            inject = switch (id) {
                case Algorithms.SENSITIVITY -> List.of("k", "m");
                case Algorithms.UNIFORM, Algorithms.MEAN -> List.of("m");
                default -> throw new IllegalArgumentException("algorithm " + id
                        + ": say which grid parameters it takes with inject: [k, m]");
            };
        }
        inject = List.copyOf(inject);
        for (String p : inject) {
            if (!p.equals("k") && !p.equals("m")) {
                throw new IllegalArgumentException("algorithm " + id + ": inject may only list k and m, got " + p);
            }
            if (params.containsKey(p)) {
                throw new IllegalArgumentException("algorithm " + id + ": " + p + " is injected, do not fix it");
            }
        }
        if (label == null || label.isBlank()) {
            label = params.isEmpty() ? id : id + params.values().stream().reduce("", (a, b) -> a + "-" + b);
        }
        if (oneSided == null) {
            oneSided = Algorithms.MEAN.equals(id);
        }
    }

    public AlgorithmSpec(String id) {
        this(null, id, null, null, null);
    }

    public boolean takesK() {
        return inject.contains("k");
    }

    /** Fixed parameters plus the injected grid values, in a stable order. */
    public Map<String, String> paramsFor(int k, int m) {
        Map<String, String> out = new LinkedHashMap<>(new TreeMap<>(params));
        if (inject.contains("k")) {
            out.put("k", Integer.toString(k));
        }
        if (inject.contains("m")) {
            out.put("m", Integer.toString(m));
        }
        return out;
    }
}
