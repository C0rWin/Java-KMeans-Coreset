package io.github.c0rwin.coreset.testbed.data;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.HexFormat;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeSet;

/**
 * A public dataset that is downloaded on request into a local cache and never redistributed. The digest pins the
 * exact file the descriptor was written against; a mismatch is an error, never a silent re-download. Loading goes
 * through a {@link PointFile} converted once from the download (see {@link Conversion} and docs/datasets.md).
 *
 * @param url      the file downloaded; an official mirror when the original location is gone (see {@code notes})
 * @param bytes    size of the downloaded file
 * @param license  as stated by the source page
 * @param page     human-readable landing page
 * @param member   zip entry holding the data, or empty when the download is the data file itself
 * @param header   whether the (dense) data file starts with a header row
 * @param drop     dense columns removed after reading (ids, class labels, symbolic attributes)
 * @param n        expected number of points after loading
 * @param d        expected dimension after loading
 * @param notes    preprocessing and provenance decisions
 */
public record PublicDataset(String id, String url, String sha256, long bytes, String license, String page,
                            Loaders.Format format, String member, Loaders.Header header, List<String> drop, int n,
                            int d, String citation, String notes) {

    private static final String CC_BY = "CC BY 4.0";

    /** Default cache, {@code ~/.cache/coreset-testbed}. */
    public static Path defaultCacheDir() {
        return Path.of(System.getProperty("user.home"), ".cache", "coreset-testbed");
    }

    /**
     * Digests were computed on 2026-11-08 from the URLs below; every license was read off the dataset's UCI page on
     * that day. Citations follow the BibTeX UCI publishes for each dataset.
     */
    public static final Map<String, PublicDataset> REGISTRY = Map.of(
            "covertype", new PublicDataset("covertype",
                    "https://archive.ics.uci.edu/static/public/31/data.csv",
                    "a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4", 75_751_077L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/31/covertype", Loaders.Format.DENSE, "",
                    Loaders.Header.PRESENT, List.of("Cover_Type"), 581_012, 54,
                    "Blackard, J. (1998). Covertype. UCI Machine Learning Repository. "
                            + "https://doi.org/10.24432/C50K5N",
                    "drops the class label Cover_Type; keeps 10 quantitative and 44 binary (one-hot wilderness and "
                            + "soil) columns as numbers"),
            "census1990", new PublicDataset("census1990",
                    "https://archive.ics.uci.edu/static/public/116/data.csv",
                    "38e99f50855ddc03167b715b36519ef40b77295ae4e85dfef0c6429ac79ed799", 361_344_227L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/116/us+census+data+1990", Loaders.Format.DENSE, "",
                    Loaders.Header.PRESENT, List.of("caseid"), 2_458_285, 68,
                    "Meek, C., Thiesson, B., Heckerman, D. (2001). US Census Data (1990). UCI Machine Learning "
                            + "Repository. https://doi.org/10.24432/C5VP42",
                    "drops the record id caseid; the 68 attributes are integer codes from the source, used as "
                            + "numbers"),
            "kddcup99", new PublicDataset("kddcup99",
                    "https://web.archive.org/web/20070709044442id_/http://kdd.ics.uci.edu/databases/kddcup99/"
                            + "kddcup.data.gz",
                    "3b6c942aa0356c0ca35b7b595a26c89d343652c9db428893e7494f837b274292", 18_115_902L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/130/kdd+cup+1999+data", Loaders.Format.DENSE, "",
                    Loaders.Header.ABSENT, List.of("1", "2", "3", "41"), 4_898_431, 38,
                    "Stolfo, S., Fan, W., Lee, W., Prodromidis, A., Chan, P. (1999). KDD Cup 1999 Data. UCI "
                            + "Machine Learning Repository. https://doi.org/10.24432/C51C7N",
                    "full kddcup.data; drops the symbolic protocol_type, service, flag (columns 1-3; one-hot "
                            + "would add 84 sparse 0/1 columns) and the label (column 41); keeps the 38 numeric "
                            + "columns, 0/1 flags included. UCI's own zip holds only dangling links and "
                            + "kdd.ics.uci.edu refuses downloads, so the URL is the Internet Archive copy of the "
                            + "original UCI KDD Archive file; its digest equals the one scikit-learn pins for its "
                            + "mirror"),
            "yearmsd", new PublicDataset("yearmsd",
                    "https://archive.ics.uci.edu/static/public/203/yearpredictionmsd.zip",
                    "06f801af323bb7798e800583acce4ea1ed2697ac12c23f4424aea0a7a3d09e11", 211_011_981L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/203/yearpredictionmsd", Loaders.Format.DENSE,
                    "YearPredictionMSD.txt", Loaders.Header.ABSENT, List.of("0"), 515_345, 90,
                    "Bertin-Mahieux, T. (2011). Year Prediction MSD. UCI Machine Learning Repository. "
                            + "https://doi.org/10.24432/C50K61",
                    "drops the release year (column 0, the regression target); keeps 12 timbre means and 78 "
                            + "timbre covariances"),
            "susy", new PublicDataset("susy",
                    "https://archive.ics.uci.edu/static/public/279/susy.zip",
                    "dd20f2a756677d369415ef0596c89266e2c5a175958056c22a36cb5321754169", 922_377_831L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/279/susy", Loaders.Format.DENSE, "SUSY.csv.gz",
                    Loaders.Header.ABSENT, List.of("0"), 5_000_000, 18,
                    "Whiteson, D. (2014). SUSY. UCI Machine Learning Repository. https://doi.org/10.24432/C54606",
                    "drops the class label (column 0); keeps 8 low-level and 10 high-level features"),
            "higgs", new PublicDataset("higgs",
                    "https://archive.ics.uci.edu/static/public/280/higgs.zip",
                    "eca8d41530af41bef24854c4eb0e3e315d41b2c837aaff67d1fde908a29dee0f", 2_816_865_137L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/280/higgs", Loaders.Format.DENSE, "HIGGS.csv.gz",
                    Loaders.Header.ABSENT, List.of("0"), 11_000_000, 28,
                    "Whiteson, D. (2014). HIGGS. UCI Machine Learning Repository. https://doi.org/10.24432/C5V312",
                    "drops the class label (column 0); keeps 21 low-level and 7 high-level features"),
            "nytimes", new PublicDataset("nytimes",
                    "https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/docword.nytimes.txt.gz",
                    "6fc63144a5834ddcc255313625a235a85440bf5b702d06a6c676f749262e43b7", 234_225_967L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/164/bag+of+words", Loaders.Format.BOW, "",
                    Loaders.Header.ABSENT, List.of(), 300_000, 102_660,
                    "Newman, D. (2008). Bag of Words. UCI Machine Learning Repository. "
                            + "https://doi.org/10.24432/C5ZG6P",
                    "word counts, one row per document (word ids made 0-based); the tfidf variant is "
                            + "count * ln(N / df) with L2-normalised rows. The URL is the NYTimes docword file in "
                            + "UCI's bag-of-words directory, not the dataset's zip of all corpora"),
            "pendigits", new PublicDataset("pendigits",
                    "https://archive.ics.uci.edu/static/public/81/data.csv",
                    "3a3043879657816e18af4bc0822cb89aff03b30988511590e2c05454ce092656", 539_812L, CC_BY,
                    "https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits",
                    Loaders.Format.DENSE, "", Loaders.Header.PRESENT, List.of("Class"), 10_992, 16,
                    "Alpaydin, E., Alimoglu, F. (1996). Pen-Based Recognition of Handwritten Digits. UCI Machine "
                            + "Learning Repository. https://doi.org/10.24432/C5MG6K",
                    "drops the class label Class"));

    public PublicDataset {
        Objects.requireNonNull(id, "id");
        Objects.requireNonNull(member, "member");
        Objects.requireNonNull(header, "header");
        drop = List.copyOf(drop);
    }

    /** The registered descriptor. @throws IllegalArgumentException listing the known names */
    public static PublicDataset named(String name) {
        PublicDataset pd = REGISTRY.get(name);
        if (pd == null) {
            throw new IllegalArgumentException("unknown public dataset '" + name + "', known: "
                    + new TreeSet<>(REGISTRY.keySet()));
        }
        return pd;
    }

    /** Options that read the downloaded file. */
    public Loaders.Options loaderOptions() {
        return new Loaders.Options(header, drop, 0);
    }

    public Path cachedFile(Path cacheDir) {
        return cacheDir.resolve(id).resolve(fileName());
    }

    private String fileName() {
        String path = URI.create(url).getPath();
        return path.substring(path.lastIndexOf('/') + 1);
    }

    /** The cached file if it exists and matches the digest. @throws IOException if it exists but does not match */
    public Optional<Path> cached(Path cacheDir) throws IOException {
        Path f = cachedFile(cacheDir);
        if (!Files.isRegularFile(f)) {
            return Optional.empty();
        }
        String actual = sha256(f);
        if (!actual.equals(sha256)) {
            throw new IOException(f + ": sha256 " + actual + " does not match the descriptor (" + sha256
                    + "); delete it to download again");
        }
        return Optional.of(f);
    }

    /** Downloads into the cache (unless already there) and verifies the digest before publishing the file. */
    public Path fetch(Path cacheDir) throws IOException, InterruptedException {
        Optional<Path> hit = cached(cacheDir);
        if (hit.isPresent()) {
            return hit.get();
        }
        Path target = cachedFile(cacheDir);
        Files.createDirectories(target.getParent());
        Path part = target.resolveSibling(target.getFileName() + ".part");
        HttpClient client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NORMAL)
                .connectTimeout(Duration.ofSeconds(30)).build();
        HttpResponse<InputStream> resp = client.send(HttpRequest.newBuilder(URI.create(url)).GET().build(),
                HttpResponse.BodyHandlers.ofInputStream());
        if (resp.statusCode() != 200) {
            resp.body().close();
            throw new IOException(url + ": HTTP " + resp.statusCode());
        }
        MessageDigest md = sha256Digest();
        try (InputStream in = new DigestInputStream(resp.body(), md);
             OutputStream out = Files.newOutputStream(part)) {
            in.transferTo(out);
        }
        String actual = HexFormat.of().formatHex(md.digest());
        if (!actual.equals(sha256)) {
            Files.deleteIfExists(part);
            throw new IOException(url + ": downloaded sha256 " + actual + " does not match " + sha256);
        }
        Files.move(part, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        return target;
    }

    public static String sha256(Path file) throws IOException {
        MessageDigest md = sha256Digest();
        try (InputStream in = new DigestInputStream(Files.newInputStream(file), md)) {
            in.transferTo(OutputStream.nullOutputStream());
        }
        return HexFormat.of().formatHex(md.digest());
    }

    static MessageDigest sha256Digest() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required of every JVM", e);
        }
    }
}
