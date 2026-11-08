package io.github.c0rwin.coreset.examples.lightweight;

import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.CoresetAlgorithmProvider;
import io.github.c0rwin.coreset.core.algo.Algorithms;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * Registers {@link LightweightCoreset} as algorithm {@code lightweight} with parameter {@code m} (required, at most
 * {@link Algorithms#DEFAULT_MAX_M}). The only wiring is
 * {@code META-INF/services/io.github.c0rwin.coreset.core.CoresetAlgorithmProvider}: with this jar on the class path,
 * the node, the gRPC CLIs, Spark, Flink and the testbed all accept {@code lightweight}.
 */
public final class LightweightCoresetProvider implements CoresetAlgorithmProvider {

    public static final String ID = "lightweight";

    @Override
    public Set<String> ids() {
        return Set.of(ID);
    }

    @Override
    public CoresetAlgorithm create(String id, Map<String, String> params) {
        if (!params.keySet().equals(Set.of("m"))) {
            throw new IllegalArgumentException(id + ": expected exactly parameter m, got " + new TreeSet<>(params.keySet()));
        }
        String raw = params.get("m");
        int m;
        try {
            m = Integer.parseInt(raw.trim());
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(id + ": parameter m must be an integer, got '" + raw + "'", e);
        }
        if (m > Algorithms.DEFAULT_MAX_M) {
            throw new IllegalArgumentException(id + ": m must be <= " + Algorithms.DEFAULT_MAX_M + ", got " + m);
        }
        return new LightweightCoreset(m);
    }
}
