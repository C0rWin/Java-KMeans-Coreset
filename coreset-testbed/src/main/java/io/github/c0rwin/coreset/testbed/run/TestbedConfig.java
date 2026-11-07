package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.problem.Problem;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * A testbed configuration: the grid {@code datasets x ks x sizes x algorithms x modes x replicates}, minus
 * {@code exclude} rules, evaluated as described by {@code evaluation}.
 *
 * @param seed root of every random choice; replicate {@code r} of every configuration uses the same derived seed, so
 *             algorithms are compared on paired randomness
 */
public record TestbedConfig(String name, String problem, List<DatasetSpec> datasets, List<Integer> ks,
                            List<Integer> sizes, List<AlgorithmSpec> algorithms, List<ModeSpec> modes,
                            Integer replicates, Long seed, EvaluationSpec evaluation, List<Exclude> exclude) {

    public TestbedConfig {
        name = name == null || name.isBlank() ? "testbed" : name;
        problem = Problem.of(problem == null ? Problem.KMEANS : problem).id();
        datasets = nonEmpty("datasets", datasets);
        ks = nonEmpty("ks", ks);
        sizes = nonEmpty("sizes", sizes);
        algorithms = nonEmpty("algorithms", algorithms);
        modes = modes == null || modes.isEmpty() ? List.of(ModeSpec.offline()) : List.copyOf(modes);
        replicates = replicates == null ? 5 : replicates;
        seed = seed == null ? 42L : seed;
        evaluation = evaluation == null ? EvaluationSpec.defaults() : evaluation;
        exclude = exclude == null ? List.of() : List.copyOf(exclude);
        if (replicates < 1) {
            throw new IllegalArgumentException("replicates must be >= 1: " + replicates);
        }
        for (int k : ks) {
            if (k < 1) {
                throw new IllegalArgumentException("k must be >= 1: " + k);
            }
        }
        for (int m : sizes) {
            if (m < 1) {
                throw new IllegalArgumentException("size must be >= 1: " + m);
            }
        }
        unique("dataset id", datasets.stream().map(DatasetSpec::id).toList());
        unique("algorithm label", algorithms.stream().map(AlgorithmSpec::label).toList());
        unique("mode label", modes.stream().map(ModeSpec::label).toList());
    }

    /**
     * Removes matching grid points; a null field matches anything.
     *
     * @param minN matches datasets with at least this many points (lets a rule say "mean only on small inputs")
     */
    public record Exclude(String dataset, String algorithm, String mode, Integer k, Integer size, Integer minN) {

        boolean matches(String datasetId, int n, String algorithmLabel, String modeLabel, int kk, int m) {
            return (dataset == null || dataset.equals(datasetId))
                    && (algorithm == null || algorithm.equals(algorithmLabel))
                    && (mode == null || mode.equals(modeLabel))
                    && (k == null || k == kk)
                    && (size == null || size == m)
                    && (minN == null || n >= minN);
        }
    }

    public boolean excluded(String datasetId, int n, String algorithmLabel, String modeLabel, int k, int m) {
        for (Exclude e : exclude) {
            if (e.matches(datasetId, n, algorithmLabel, modeLabel, k, m)) {
                return true;
            }
        }
        return false;
    }

    private static <T> List<T> nonEmpty(String name, List<T> list) {
        if (list == null || list.isEmpty()) {
            throw new IllegalArgumentException(name + " must not be empty");
        }
        List<T> out = new ArrayList<>(list.size());
        for (T t : list) {
            out.add(Objects.requireNonNull(t, name + " entry"));
        }
        return List.copyOf(out);
    }

    private static void unique(String what, List<String> values) {
        Set<String> seen = new HashSet<>();
        for (String v : values) {
            if (!seen.add(v)) {
                throw new IllegalArgumentException("duplicate " + what + ": " + v);
            }
        }
    }
}
