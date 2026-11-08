package io.github.c0rwin.coreset.experiments.data;

import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.DatasetConfig;
import io.github.c0rwin.coreset.testbed.data.Dataset;
import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.data.Datasets;
import io.github.c0rwin.coreset.testbed.data.Esa22Benchmark;
import io.github.c0rwin.coreset.testbed.data.SyntheticData;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Resolves dataset configs to {@link PointSource}s. Generated and testbed-loaded datasets are written once to a dense
 * file in the cache ({@code <cacheDir>/datasets/<id>-<key>.f64}, keyed by type and parameters) so every system and
 * every repetition reads identical rows from storage instead of regenerating them.
 *
 * <p>Types: {@code gaussian-mixture} and {@code esa22} (testbed generators, same parameters as the testbed);
 * {@code testbed} (any testbed dataset spec: {@code testbedType} plus its parameters, materialised dense);
 * {@code dense-file} ({@code path} of an existing dense-f64 file). Other storage formats plug in as further
 * {@link PointSource} implementations selected here.
 */
public final class PointSources {

    private PointSources() {
    }

    /** Opens the source, materialising it into the cache first if needed. */
    public static PointSource open(DatasetConfig spec, Path cacheDir) throws IOException {
        if (spec.type().equals("dense-file")) {
            Path file = Path.of(required(spec, "path"));
            return DenseFileSource.open(spec.id(), file, provenance(spec, ""));
        }
        Path file = cacheDir.resolve("datasets").resolve(spec.id() + "-" + key(spec) + ".f64");
        Path digest = file.resolveSibling(file.getFileName() + ".sha256");
        if (!Files.exists(file) || !Files.exists(digest)) {
            materialize(spec, file);
            Files.writeString(digest, sha256(file), StandardCharsets.UTF_8);
        }
        return DenseFileSource.open(spec.id(), file, provenance(spec, Files.readString(digest).strip()));
    }

    private static Map<String, Object> provenance(DatasetConfig spec, String sha256) {
        Map<String, Object> p = new LinkedHashMap<>();
        p.put("type", spec.type());
        p.put("params", new TreeMap<>(spec.params()));
        p.put("sha256", sha256);
        return p;
    }

    private static void materialize(DatasetConfig spec, Path file) throws IOException {
        List<WeightedPoint> points = generate(spec);
        int d = points.get(0).vector().dim();
        boolean weighted = points.stream().anyMatch(p -> p.weight() != 1.0);
        Files.createDirectories(file.getParent());
        Path tmp = Files.createTempFile(file.getParent(), file.getFileName().toString(), ".tmp");
        try (DenseFileSource.Writer w = new DenseFileSource.Writer(tmp, d, weighted)) {
            double[] row = new double[d];
            for (WeightedPoint p : points) {
                SparseVector v = p.vector();
                java.util.Arrays.fill(row, 0.0);
                for (int e = 0, nnz = v.nnz(); e < nnz; e++) {
                    row[v.indexAt(e)] = v.valueAt(e);
                }
                w.add(row, p.weight());
            }
        }
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }

    private static List<WeightedPoint> generate(DatasetConfig spec) {
        Map<String, String> p = spec.params();
        switch (spec.type()) {
            case "gaussian-mixture":
                return SyntheticData.gaussianMixture(Integer.parseInt(required(spec, "n")),
                        Integer.parseInt(required(spec, "d")), Integer.parseInt(required(spec, "clusters")),
                        Double.parseDouble(required(spec, "sigma")), Double.parseDouble(p.getOrDefault("imbalance", "0")),
                        Double.parseDouble(p.getOrDefault("box", "100")), Long.parseLong(p.getOrDefault("seed", "1")));
            case "esa22":
                return Esa22Benchmark.of(Integer.parseInt(required(spec, "k")),
                        Integer.parseInt(required(spec, "alpha"))).points();
            case "testbed": {
                Map<String, String> rest = new TreeMap<>(p);
                String type = rest.remove("testbedType");
                if (type == null) {
                    throw new IllegalArgumentException("dataset " + spec.id() + ": testbed type needs testbedType");
                }
                Path cache = Path.of(System.getProperty("user.home"), ".cache", "coreset-testbed");
                Dataset ds = new Datasets(cache, false).load(new DatasetSpec(spec.id(), type, rest));
                return ds.points();
            }
            default:
                throw new IllegalArgumentException("dataset " + spec.id() + ": unknown type '" + spec.type()
                        + "', expected gaussian-mixture, esa22, testbed or dense-file");
        }
    }

    private static String required(DatasetConfig spec, String key) {
        String v = spec.params().get(key);
        if (v == null) {
            throw new IllegalArgumentException("dataset " + spec.id() + " (" + spec.type() + ") needs " + key);
        }
        return v;
    }

    private static String key(DatasetConfig spec) {
        String canonical = spec.type() + new TreeMap<>(spec.params());
        try {
            byte[] h = MessageDigest.getInstance("SHA-256").digest(canonical.getBytes(StandardCharsets.UTF_8));
            return HexFormat.of().formatHex(h, 0, 6);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    static String sha256(Path file) throws IOException {
        try (InputStream in = Files.newInputStream(file)) {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] buf = new byte[1 << 20];
            for (int r; (r = in.read(buf)) > 0; ) {
                md.update(buf, 0, r);
            }
            return HexFormat.of().formatHex(md.digest());
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }
}
