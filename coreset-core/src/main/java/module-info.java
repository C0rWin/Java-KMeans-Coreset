/**
 * k-means coresets for sparse weighted data: points and coresets ({@code core}), coreset algorithms and the algorithm
 * SPI ({@code core}, {@code algo}), weighted k-means ({@code kmeans}), merge-reduce streaming ({@code stream}),
 * sampling, metrics and validation. See {@code docs/api.md} for the entry points per use case.
 *
 * <p>{@code io.github.c0rwin.coreset.core.internal} holds kernels shared between the packages and is not exported.
 * Allocation probes in {@link io.github.c0rwin.coreset.core.metrics.Span} use {@code jdk.management} when it is
 * resolved (on the module path: {@code --add-modules jdk.management}); they are skipped otherwise.
 */
module io.github.c0rwin.coreset.core {
    requires java.management;
    requires static jdk.management;

    exports io.github.c0rwin.coreset.core;
    exports io.github.c0rwin.coreset.core.algo;
    exports io.github.c0rwin.coreset.core.kmeans;
    exports io.github.c0rwin.coreset.core.metrics;
    exports io.github.c0rwin.coreset.core.sampling;
    exports io.github.c0rwin.coreset.core.stream;
    exports io.github.c0rwin.coreset.core.validate;

    uses io.github.c0rwin.coreset.core.CoresetAlgorithmProvider;

    provides io.github.c0rwin.coreset.core.CoresetAlgorithmProvider
            with io.github.c0rwin.coreset.core.algo.BuiltInAlgorithms;
}
