package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.core.WeightedPoint;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Resolves {@link DatasetSpec}s to in-memory {@link Dataset}s and memoises them, so a suite generates or reads each
 * dataset once. Public datasets are only downloaded when {@code allowDownload} is set.
 */
public final class Datasets {

    private final Path cacheDir;
    private final boolean allowDownload;
    private final Map<DatasetSpec, Dataset> loaded = new ConcurrentHashMap<>();

    public Datasets(Path cacheDir, boolean allowDownload) {
        this.cacheDir = cacheDir;
        this.allowDownload = allowDownload;
    }

    public Dataset load(DatasetSpec spec) {
        return loaded.computeIfAbsent(spec, s -> {
            try {
                return create(s);
            } catch (IOException e) {
                throw new UncheckedIOException("dataset " + s.id() + ": " + e.getMessage(), e);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted while loading " + s.id(), e);
            }
        });
    }

    /** Drops a memoised dataset so its memory can be reclaimed. */
    public void evict(DatasetSpec spec) {
        loaded.remove(spec);
    }

    private Dataset create(DatasetSpec spec) throws IOException, InterruptedException {
        Params p = new Params(spec);
        switch (spec.type()) {
            case "gaussian-mixture": {
                p.allow("n", "d", "clusters", "sigma", "imbalance", "box", "seed");
                List<WeightedPoint> pts = SyntheticData.gaussianMixture(p.integer("n"), p.integer("d"),
                        p.integer("clusters"), p.real("sigma"), p.real("imbalance", 0.0), p.real("box", 100.0),
                        p.longValue("seed", 1L));
                return Dataset.of(spec.id(), pts, "", "generated", "synthetic:gaussian-mixture" + spec.params(),
                        List.of(), Optional.empty());
            }
            case "esa22": {
                p.allow("k", "alpha");
                Esa22Benchmark b = Esa22Benchmark.of(p.integer("k"), p.integer("alpha"));
                return Dataset.of(spec.id(), b.points(), "", "generated", "synthetic:esa22" + spec.params(),
                        List.of(), Optional.of(b));
            }
            case "sparse-bow": {
                p.allow("n", "d", "topics", "topicVocabulary", "docLength", "topicMix", "imbalance", "seed");
                int d = p.integer("d");
                int topics = p.integer("topics");
                List<WeightedPoint> pts = SyntheticData.sparseBagOfWords(p.integer("n"), d, topics,
                        p.integer("topicVocabulary", Math.max(1, d / topics)), p.integer("docLength"),
                        p.real("topicMix", 0.8), p.real("imbalance", 0.0), p.longValue("seed", 1L));
                return Dataset.of(spec.id(), pts, "", "generated", "synthetic:sparse-bow" + spec.params(),
                        List.of(), Optional.empty());
            }
            case "file": {
                p.allow("path", "format", "header", "drop", "dim", "license");
                Path file = Path.of(p.string("path"));
                Loaders.Format format = Loaders.Format.of(p.string("format"));
                List<String> drop = p.list("drop");
                Loaders.Options options = new Loaders.Options(
                        Loaders.Header.valueOf(p.string("header", "auto").toUpperCase(Locale.ROOT)), drop,
                        p.integer("dim", 0));
                List<WeightedPoint> pts = Loaders.load(file, format, options);
                return Dataset.of(spec.id(), pts, PublicDataset.sha256(file), p.string("license", "unknown"),
                        file.toAbsolutePath().toString(), dropSteps(drop), Optional.empty());
            }
            case "public": {
                p.allow("name");
                String name = p.string("name");
                PublicDataset pd = PublicDataset.REGISTRY.get(name);
                if (pd == null) {
                    throw new IllegalArgumentException("unknown public dataset '" + name + "', known: "
                            + new TreeSet<>(PublicDataset.REGISTRY.keySet()));
                }
                Path file = pd.cached(cacheDir).orElse(null);
                if (file == null) {
                    if (!allowDownload) {
                        throw new IllegalStateException("public dataset " + name + " is not cached in " + cacheDir
                                + "; fetch it with 'coreset-testbed datasets fetch " + name + "' or pass "
                                + "--allow-download");
                    }
                    file = pd.fetch(cacheDir);
                }
                List<WeightedPoint> pts = Loaders.load(file, pd.format(),
                        new Loaders.Options(Loaders.Header.AUTO, pd.drop(), 0));
                if (pts.size() != pd.n() || pts.get(0).vector().dim() != pd.d()) {
                    throw new IllegalStateException(name + ": loaded " + pts.size() + " x "
                            + pts.get(0).vector().dim() + ", descriptor says " + pd.n() + " x " + pd.d());
                }
                return Dataset.of(spec.id(), pts, pd.sha256(), pd.license(), pd.url(), dropSteps(pd.drop()),
                        Optional.empty());
            }
            default:
                throw new IllegalArgumentException("unknown dataset type '" + spec.type()
                        + "', expected gaussian-mixture, esa22, sparse-bow, file or public");
        }
    }

    private static List<String> dropSteps(List<String> drop) {
        List<String> out = new ArrayList<>(drop.size());
        for (String d : drop) {
            out.add("drop:" + d);
        }
        return out;
    }

    /** Typed access to string parameters with unknown-key detection. */
    static final class Params {
        private final DatasetSpec spec;

        Params(DatasetSpec spec) {
            this.spec = spec;
        }

        void allow(String... names) {
            Set<String> unknown = new TreeSet<>(spec.params().keySet());
            unknown.removeAll(Arrays.asList(names));
            if (!unknown.isEmpty()) {
                throw new IllegalArgumentException(spec.id() + " (" + spec.type() + "): unknown parameter(s) "
                        + unknown + ", allowed " + new TreeSet<>(Arrays.asList(names)));
            }
        }

        String string(String name) {
            String v = spec.params().get(name);
            if (v == null) {
                throw new IllegalArgumentException(spec.id() + " (" + spec.type() + "): missing parameter " + name);
            }
            return v.trim();
        }

        String string(String name, String fallback) {
            String v = spec.params().get(name);
            return v == null ? fallback : v.trim();
        }

        List<String> list(String name) {
            String v = spec.params().get(name);
            if (v == null || v.isBlank()) {
                return List.of();
            }
            List<String> out = new ArrayList<>();
            for (String s : v.split(",")) {
                if (!s.isBlank()) {
                    out.add(s.trim());
                }
            }
            return out;
        }

        int integer(String name) {
            return toInt(name, parseLong(name, string(name)));
        }

        int integer(String name, int fallback) {
            String v = spec.params().get(name);
            return v == null ? fallback : toInt(name, parseLong(name, v));
        }

        private int toInt(String name, long v) {
            if (v < Integer.MIN_VALUE || v > Integer.MAX_VALUE) {
                throw new IllegalArgumentException(spec.id() + ": parameter " + name + " out of int range: " + v);
            }
            return (int) v;
        }

        long longValue(String name, long fallback) {
            String v = spec.params().get(name);
            return v == null ? fallback : parseLong(name, v);
        }

        double real(String name) {
            return parseDouble(name, string(name));
        }

        double real(String name, double fallback) {
            String v = spec.params().get(name);
            return v == null ? fallback : parseDouble(name, v);
        }

        private long parseLong(String name, String v) {
            String s = v.trim();
            try {
                return Long.parseLong(s);
            } catch (NumberFormatException notPlain) {
                try {
                    double d = Double.parseDouble(s); // accepts 1e6
                    if (d != Math.rint(d) || Math.abs(d) >= 0x1p53) {
                        throw new NumberFormatException();
                    }
                    return (long) d;
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException(spec.id() + ": parameter " + name
                            + " must be an integer, got '" + v + "'");
                }
            }
        }

        private double parseDouble(String name, String v) {
            try {
                return Double.parseDouble(v.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(spec.id() + ": parameter " + name + " must be a number, got '"
                        + v + "'");
            }
        }
    }
}
