package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * An in-memory input point set with the provenance a validity report needs.
 *
 * @param moments       one weighted point at the input mean plus an offset, with exactly the input's total weight,
 *                      first moment and second moment: integrity checks against it need no pass over the input
 * @param sha256        hex digest of the source file, empty for generated data
 * @param source        URL, file path or generator description
 * @param preprocessing steps applied after reading, e.g. {@code drop:caseid}
 * @param benchmark     the ESA'22 instance when this dataset is one (enables the delta-sweep lower bound)
 */
public record Dataset(String id, Coreset coreset, Coreset moments, int dim, long nnz, double weightSum,
                      String sha256, String license, String source, List<String> preprocessing,
                      Optional<Esa22Benchmark> benchmark) {

    public Dataset {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(coreset, "coreset");
        Objects.requireNonNull(moments, "moments");
        Objects.requireNonNull(sha256, "sha256");
        Objects.requireNonNull(license, "license");
        Objects.requireNonNull(source, "source");
        preprocessing = List.copyOf(preprocessing);
        Objects.requireNonNull(benchmark, "benchmark");
    }

    /**
     * Wraps non-empty points; one pass computes nnz, weight and moments. The point list is copied once (by
     * {@link Coreset}), points are shared.
     */
    public static Dataset of(String id, List<WeightedPoint> points, String sha256, String license, String source,
                             List<String> preprocessing, Optional<Esa22Benchmark> benchmark) {
        if (points.isEmpty()) {
            throw new IllegalArgumentException("dataset " + id + " has no points");
        }
        int dim = points.get(0).vector().dim();
        double[] first = new double[dim];
        long nnz = 0;
        double weight = 0.0;
        double second = 0.0;
        for (int i = 0, n = points.size(); i < n; i++) {
            WeightedPoint p = points.get(i);
            double w = p.weight();
            nnz += p.vector().nnz();
            weight += w;
            second += w * p.vector().squaredNorm();
            p.vector().addTo(first, w);
        }
        if (!(weight > 0.0)) {
            throw new IllegalArgumentException("dataset " + id + " has no positive weight");
        }
        double meanNorm = 0.0;
        for (int t = 0; t < dim; t++) {
            first[t] /= weight;
            meanNorm += first[t] * first[t];
        }
        // sum w ||p||^2 = W ||mean||^2 + sum w ||p - mean||^2, so the offset is the spread around the mean.
        Coreset moments = new Coreset(List.of(new WeightedPoint(SparseVector.dense(first), weight)),
                Math.max(0.0, second - weight * meanNorm));
        return new Dataset(id, Coreset.of(points), moments, dim, nnz, weight, sha256, license, source,
                preprocessing, benchmark);
    }

    public List<WeightedPoint> points() {
        return coreset.points();
    }

    public int n() {
        return coreset.size();
    }
}
