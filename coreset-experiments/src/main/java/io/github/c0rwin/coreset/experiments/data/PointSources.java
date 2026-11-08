package io.github.c0rwin.coreset.experiments.data;

import io.github.c0rwin.coreset.core.PointBlock;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.experiments.config.ExperimentConfig.DatasetConfig;
import io.github.c0rwin.coreset.testbed.data.Conversion;
import io.github.c0rwin.coreset.testbed.data.DatasetSpec;
import io.github.c0rwin.coreset.testbed.data.Datasets;
import io.github.c0rwin.coreset.testbed.data.Esa22Benchmark;
import io.github.c0rwin.coreset.testbed.data.PointFileWriter;
import io.github.c0rwin.coreset.testbed.data.PublicDataset;
import io.github.c0rwin.coreset.testbed.data.SyntheticData;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Resolves dataset configs to {@link PointSource}s over testbed {@code PointFile}s, so every system and repetition
 * reads identical rows from storage in the repository's one binary format.
 *
 * <p>Types:
 * <ul>
 *   <li>{@code public}: a testbed dataset ({@code name}, optional {@code variant} {@code raw}|{@code zscore}|
 *       {@code tfidf}, optional {@code cacheDir}, default {@code ~/.cache/coreset-testbed}), read from its converted
 *       binary; it is never downloaded or converted here ({@code coreset-testbed datasets convert <name>}).</li>
 *   <li>{@code gaussian-mixture}, {@code esa22}: testbed generators, written once to
 *       {@code <cacheDir>/datasets/<id>-<key>.cbin} (keyed by type and parameters).</li>
 *   <li>{@code testbed}: any testbed dataset spec ({@code testbedType} plus its parameters), written once as
 *       {@code .cbin}, or {@code .csr} when under half of the entries are non-zero.</li>
 *   <li>{@code file}: {@code path} of an existing {@code .cbin} or {@code .csr}.</li>
 * </ul>
 */
public final class PointSources {

    private PointSources() {
    }

    /** Opens the source, materialising generated data into the cache first if needed. */
    public static PointSource open(DatasetConfig spec, Path cacheDir) throws IOException {
        Map<String, String> p = spec.params();
        Map<String, Object> prov = new LinkedHashMap<>();
        prov.put("type", spec.type());
        prov.put("params", new TreeMap<>(p));
        switch (spec.type()) {
            case "public": {
                PublicDataset pd = PublicDataset.named(required(spec, "name"));
                Conversion.Variant v = Conversion.Variant.of(p.getOrDefault("variant", "raw"));
                Path dir = p.containsKey("cacheDir") ? Path.of(p.get("cacheDir"))
                        : Path.of(System.getProperty("user.home"), ".cache", "coreset-testbed");
                Path file = Conversion.binaryFile(pd, dir, v);
                if (!Files.exists(file)) {
                    throw new IllegalStateException("dataset " + spec.id() + ": " + file + " is missing; convert it"
                            + " with 'coreset-testbed datasets convert " + pd.id()
                            + (v == Conversion.Variant.RAW ? "" : " --variant " + v.id()) + "'");
                }
                prov.put("downloadSha256", pd.sha256());
                prov.put("license", pd.license());
                prov.put("source", pd.url());
                return PointFileSource.open(spec.id(), file, prov);
            }
            case "file":
                return PointFileSource.open(spec.id(), Path.of(required(spec, "path")), prov);
            case "gaussian-mixture":
            case "esa22":
            case "testbed": {
                Path dense = cacheDir.resolve("datasets").resolve(spec.id() + "-" + key(spec) + ".cbin");
                Path sparse = dense.resolveSibling(spec.id() + "-" + key(spec) + ".csr");
                Path file = Files.exists(dense) ? dense : Files.exists(sparse) ? sparse : materialize(spec, dense,
                        sparse);
                return PointFileSource.open(spec.id(), file, prov);
            }
            default:
                throw new IllegalArgumentException("dataset " + spec.id() + ": unknown type '" + spec.type()
                        + "', expected public, file, gaussian-mixture, esa22 or testbed");
        }
    }

    private static Path materialize(DatasetConfig spec, Path dense, Path sparse) throws IOException {
        PointSet rows = generate(spec);
        if (rows.isEmpty()) {
            throw new IllegalArgumentException("dataset " + spec.id() + " has no rows");
        }
        int d = rows.dim();
        boolean csr = rows.storedEntries() * 2 < (long) rows.size() * d;
        Path target = csr ? sparse : dense;
        // the writer builds <target>.part and moves it into place, so a crash leaves no half-written dataset
        try (PointFileWriter w = csr ? PointFileWriter.csr(target, rows.size(), d, rows.storedEntries(), 0)
                : PointFileWriter.dense(target, d, 0)) {
            double[] row = new double[d];
            for (int s = 0; s < rows.segmentCount(); s++) {
                PointBlock b = rows.segment(s);
                for (int i = 0; i < b.size(); i++) {
                    if (b.weight(i) != 1.0) {
                        throw new IllegalArgumentException("dataset " + spec.id() + ": point files hold unit weights");
                    }
                    if (csr) {
                        int from = b.rowStart(i);
                        int to = b.rowEnd(i);
                        int[] idx = new int[to - from];
                        double[] val = new double[to - from];
                        for (int e = from; e < to; e++) {
                            idx[e - from] = b.index(e);
                            val[e - from] = b.value(e);
                        }
                        w.writeRow(idx, val, 0, idx.length);
                    } else {
                        Arrays.fill(row, 0.0);
                        for (int e = b.rowStart(i); e < b.rowEnd(i); e++) {
                            row[b.index(e)] = b.value(e);
                        }
                        w.writeRow(row);
                    }
                }
            }
            w.finish();
        }
        return target;
    }

    private static PointSet generate(DatasetConfig spec) {
        Map<String, String> p = spec.params();
        switch (spec.type()) {
            case "gaussian-mixture":
                return points(SyntheticData.gaussianMixture(Integer.parseInt(required(spec, "n")),
                        Integer.parseInt(required(spec, "d")), Integer.parseInt(required(spec, "clusters")),
                        Double.parseDouble(required(spec, "sigma")), Double.parseDouble(p.getOrDefault("imbalance", "0")),
                        Double.parseDouble(p.getOrDefault("box", "100")), Long.parseLong(p.getOrDefault("seed", "1"))));
            case "esa22":
                return points(Esa22Benchmark.of(Integer.parseInt(required(spec, "k")),
                        Integer.parseInt(required(spec, "alpha"))).points());
            default: {
                Map<String, String> rest = new TreeMap<>(p);
                String type = rest.remove("testbedType");
                if (type == null) {
                    throw new IllegalArgumentException("dataset " + spec.id() + ": testbed type needs testbedType");
                }
                Path cache = Path.of(System.getProperty("user.home"), ".cache", "coreset-testbed");
                return new Datasets(cache, false).load(new DatasetSpec(spec.id(), type, rest)).coreset().pointSet();
            }
        }
    }

    private static PointSet points(List<WeightedPoint> list) {
        return PointSet.of(PointBlock.of(list));
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
}
