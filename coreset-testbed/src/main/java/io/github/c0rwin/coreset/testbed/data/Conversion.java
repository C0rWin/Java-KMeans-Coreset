package io.github.c0rwin.coreset.testbed.data;

import io.github.c0rwin.coreset.testbed.data.PointFile.Header;
import io.github.c0rwin.coreset.testbed.data.PointFile.Layout;
import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

/**
 * Converts downloads into {@link PointFile}s once and derives the preprocessing variants from them. Deterministic:
 * values are parsed as float64 exactly as {@link Double#parseDouble} reads them, rows keep the source order (no
 * shuffling; experiments shuffle with their own seeds), and the variants are fixed functions of the raw file.
 * Binaries live next to the download: {@code <cache>/<id>/<id>[.<variant>].cbin|.csr}.
 */
public final class Conversion {

    private Conversion() {
    }

    /** Preprocessing applied on top of the raw conversion. */
    public enum Variant {
        /** The parsed source after dropping the descriptor's columns. */
        RAW,
        /** Dense only: per-feature z-score, {@code (x - mean) / std} with the population std; constant features 0. */
        ZSCORE,
        /** Sparse only: {@code count * ln(N / df)} per entry, then every non-empty row scaled to unit L2 norm. */
        TFIDF;

        public static Variant of(String id) {
            return valueOf(id.trim().toUpperCase(Locale.ROOT));
        }

        public String id() {
            return name().toLowerCase(Locale.ROOT);
        }

        Layout layout(Layout raw) {
            if ((this == ZSCORE && raw != Layout.DENSE) || (this == TFIDF && raw != Layout.CSR)) {
                throw new IllegalArgumentException("variant " + id() + " does not apply to " + raw + " data");
            }
            return raw;
        }
    }

    static Layout rawLayout(PublicDataset pd) {
        return pd.format() == Loaders.Format.BOW ? Layout.CSR : Layout.DENSE;
    }

    /** Where the variant's binary lives in {@code cacheDir}. */
    public static Path binaryFile(PublicDataset pd, Path cacheDir, Variant v) {
        String suffix = v == Variant.RAW ? "" : "." + v.id();
        return cacheDir.resolve(pd.id()).resolve(pd.id() + suffix + v.layout(rawLayout(pd)).extension());
    }

    /**
     * The variant's binary, converting (and for the raw file downloading, when allowed) whatever is missing. An
     * existing binary is trusted after a header check; delete it to convert again.
     *
     * @throws IllegalStateException if the download is needed but not cached and {@code allowDownload} is false
     */
    public static Path ensure(PublicDataset pd, Path cacheDir, Variant v, boolean allowDownload)
            throws IOException, InterruptedException {
        Path target = binaryFile(pd, cacheDir, v);
        if (Files.isRegularFile(target)) {
            requireShape(pd, PointFile.readHeader(target), target);
            return target;
        }
        if (v != Variant.RAW) {
            Path raw = ensure(pd, cacheDir, Variant.RAW, allowDownload);
            derive(raw, target, v);
            return target;
        }
        Path download = pd.cached(cacheDir).orElse(null);
        if (download == null) {
            if (!allowDownload) {
                throw new IllegalStateException("public dataset " + pd.id() + " is not cached in " + cacheDir
                        + "; fetch it with 'coreset-testbed datasets fetch " + pd.id() + "' or pass "
                        + "--allow-download");
            }
            download = pd.fetch(cacheDir);
        }
        convertRaw(pd, download, target);
        return target;
    }

    /**
     * Parses the download into {@code target}.
     *
     * @throws IllegalStateException if the shape differs from the descriptor (the binary is then deleted)
     */
    public static Header convertRaw(PublicDataset pd, Path download, Path target) throws IOException {
        Header h;
        try (BufferedReader r = Loaders.open(download, pd.member())) {
            h = pd.format() == Loaders.Format.BOW ? convertBagOfWords(r, target)
                    : convertDense(r, pd.loaderOptions(), pd.d(), target);
        }
        try {
            requireShape(pd, h, target);
        } catch (IllegalStateException e) {
            Files.deleteIfExists(target);
            throw e;
        }
        return h;
    }

    /** Streams a dense text file of {@code d} kept columns into a {@code .cbin}. */
    public static Header convertDense(BufferedReader r, Loaders.Options options, int d, Path target)
            throws IOException {
        try (PointFileWriter w = PointFileWriter.dense(target, d, 0)) {
            Loaders.streamDense(r, options, w::writeRow);
            return w.finish();
        }
    }

    /** Streams a bag-of-words file into a {@code .csr}. */
    public static Header convertBagOfWords(BufferedReader r, Path target) throws IOException {
        Loaders.BagOfWordsShape shape = Loaders.bagOfWordsShape(r);
        try (PointFileWriter w = PointFileWriter.csr(target, shape.docs(), shape.words(), shape.nnz(), 0)) {
            Loaders.streamBagOfWords(r, shape, w::writeRow);
            return w.finish();
        }
    }

    /** Writes the variant of the raw binary {@code raw} to {@code target}. */
    public static Header derive(Path raw, Path target, Variant v) throws IOException {
        Header h = PointFile.readHeader(raw);
        v.layout(h.layout());
        switch (v) {
            case ZSCORE:
                return zscore(raw, h, target);
            case TFIDF:
                return tfidf(raw, h, target);
            default:
                throw new IllegalArgumentException("nothing to derive for " + v.id());
        }
    }

    // Two passes for the moments (mean first, then squared deviations) keep the variance accurate on large n.
    private static Header zscore(Path raw, Header h, Path target) throws IOException {
        final int d = h.d();
        double[] mean = new double[d];
        PointFile.forEachRow(raw, row -> {
            for (int c = 0; c < d; c++) {
                mean[c] += row[c];
            }
        });
        for (int c = 0; c < d; c++) {
            mean[c] /= h.n();
        }
        double[] std = new double[d];
        PointFile.forEachRow(raw, row -> {
            for (int c = 0; c < d; c++) {
                double x = row[c] - mean[c];
                std[c] += x * x;
            }
        });
        for (int c = 0; c < d; c++) {
            std[c] = Math.sqrt(std[c] / h.n());
        }
        double[] out = new double[d];
        try (PointFileWriter w = PointFileWriter.dense(target, d, h.flags() | PointFile.FLAG_ZSCORE)) {
            PointFile.forEachRow(raw, row -> {
                for (int c = 0; c < d; c++) {
                    out[c] = std[c] > 0.0 ? (row[c] - mean[c]) / std[c] : 0.0;
                }
                w.writeRow(out);
            });
            return w.finish();
        }
    }

    private static Header tfidf(Path raw, Header h, Path target) throws IOException {
        long[] df = new long[h.d()];
        PointFile.forEachRow(raw, (idx, val, from, to) -> {
            for (int e = from; e < to; e++) {
                df[idx[e]]++;
            }
        });
        double[] idf = new double[h.d()];
        long nnz = 0;
        for (int t = 0; t < idf.length; t++) {
            if (df[t] > 0 && df[t] < h.n()) { // a word in every document gets idf 0 and disappears
                idf[t] = Math.log((double) h.n() / df[t]);
                nnz += df[t];
            }
        }
        int[] oi = new int[h.d()];
        double[] ov = new double[h.d()];
        try (PointFileWriter w = PointFileWriter.csr(target, h.n(), h.d(), nnz, h.flags() | PointFile.FLAG_TFIDF)) {
            PointFile.forEachRow(raw, (idx, val, from, to) -> {
                int len = 0;
                double norm = 0.0;
                for (int e = from; e < to; e++) {
                    double x = val[e] * idf[idx[e]];
                    if (x != 0.0) {
                        oi[len] = idx[e];
                        ov[len++] = x;
                        norm += x * x;
                    }
                }
                double inv = 1.0 / Math.sqrt(norm);
                for (int e = 0; e < len; e++) {
                    ov[e] *= inv;
                }
                w.writeRow(oi, ov, 0, len);
            });
            return w.finish();
        }
    }

    private static void requireShape(PublicDataset pd, Header h, Path file) {
        if (h.n() != pd.n() || h.d() != pd.d()) {
            throw new IllegalStateException(file + ": " + h.n() + " x " + h.d() + ", descriptor says " + pd.n()
                    + " x " + pd.d());
        }
    }
}
