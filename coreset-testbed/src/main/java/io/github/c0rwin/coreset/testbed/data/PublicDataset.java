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

/**
 * A public dataset that is downloaded on request into a local cache and never redistributed. The digest pins the
 * exact file the descriptor was written against; a mismatch is an error, never a silent re-download.
 *
 * @param page     human-readable landing page
 * @param license  as stated by the source; check it before publishing anything derived from the data
 * @param drop     dense columns removed after reading (ids, class labels)
 * @param n        expected number of points after loading
 * @param d        expected dimension after loading
 */
public record PublicDataset(String id, String url, String sha256, long bytes, String license, String page,
                            Loaders.Format format, List<String> drop, int n, int d, String citation) {

    /** Default cache, {@code ~/.cache/coreset-testbed}. */
    public static Path defaultCacheDir() {
        return Path.of(System.getProperty("user.home"), ".cache", "coreset-testbed");
    }

    /**
     * Digests were computed on 2026-11-07 from the URLs below. UCI states CC BY 4.0 on its dataset pages; only the
     * census page was checked by hand, so treat the others as to-be-confirmed.
     */
    public static final Map<String, PublicDataset> REGISTRY = Map.of(
            "covertype", new PublicDataset("covertype",
                    "https://archive.ics.uci.edu/static/public/31/data.csv",
                    "a07902ee1c9d3231c6655f23e6f75a6797d0ba26a2359f533c2c0e65d05c9bd4", 75_751_077L,
                    "CC BY 4.0 (UCI default; unconfirmed)", "https://archive.ics.uci.edu/dataset/31/covertype",
                    Loaders.Format.DENSE, List.of("Cover_Type"), 581_012, 54,
                    "Blackard, J. Covertype. UCI Machine Learning Repository."),
            "census1990", new PublicDataset("census1990",
                    "https://archive.ics.uci.edu/static/public/116/data.csv",
                    "38e99f50855ddc03167b715b36519ef40b77295ae4e85dfef0c6429ac79ed799", 361_344_227L,
                    "CC BY 4.0", "https://archive.ics.uci.edu/dataset/116/us+census+data+1990",
                    Loaders.Format.DENSE, List.of("caseid"), 2_458_285, 68,
                    "Meek, C., Thiesson, B., Heckerman, D. US Census Data (1990). UCI Machine Learning Repository."),
            "pendigits", new PublicDataset("pendigits",
                    "https://archive.ics.uci.edu/static/public/81/data.csv",
                    "3a3043879657816e18af4bc0822cb89aff03b30988511590e2c05454ce092656", 539_812L,
                    "CC BY 4.0 (UCI default; unconfirmed)",
                    "https://archive.ics.uci.edu/dataset/81/pen+based+recognition+of+handwritten+digits",
                    Loaders.Format.DENSE, List.of("Class"), 10_992, 16,
                    "Alpaydin, E., Alimoglu, F. Pen-Based Recognition of Handwritten Digits. UCI Machine Learning "
                            + "Repository."));

    public PublicDataset {
        Objects.requireNonNull(id, "id");
        drop = List.copyOf(drop);
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
