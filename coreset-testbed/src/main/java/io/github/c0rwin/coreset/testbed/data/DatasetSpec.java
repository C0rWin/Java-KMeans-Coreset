package io.github.c0rwin.coreset.testbed.data;

import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

/**
 * Configuration of one dataset: a {@code type} plus string parameters, as written in a testbed config.
 *
 * <ul>
 *   <li>{@code gaussian-mixture}: {@code n, d, clusters, sigma}; optional {@code imbalance} (0), {@code box} (100),
 *       {@code seed} (1).</li>
 *   <li>{@code esa22}: {@code k, alpha}.</li>
 *   <li>{@code sparse-bow}: {@code n, d, topics, docLength}; optional {@code topicVocabulary} (d / topics),
 *       {@code topicMix} (0.8), {@code imbalance} (0), {@code seed} (1).</li>
 *   <li>{@code file}: {@code path}, {@code format} (dense, libsvm, wiki or bow); optional {@code header} (auto, present,
 *       absent), {@code drop} (comma-separated column names or indices), {@code dim} (libsvm), {@code license}.</li>
 *   <li>{@code public}: {@code name} of a {@link PublicDataset} and optional {@code variant} ({@code raw} default,
 *       {@code zscore}, {@code tfidf}; see {@link Conversion.Variant}); read from the cache's binary conversion, which
 *       is created on first use; downloaded only when allowed.</li>
 * </ul>
 *
 * @param id label used in reports; defaults to the public dataset name for {@code public}
 */
public record DatasetSpec(String id, String type, Map<String, String> params) {

    public DatasetSpec {
        Objects.requireNonNull(type, "type");
        params = params == null ? Map.of() : Map.copyOf(new TreeMap<>(params));
        if (id == null || id.isBlank()) {
            if (!"public".equals(type) || !params.containsKey("name")) {
                throw new IllegalArgumentException("dataset of type " + type + " needs an id");
            }
            id = params.get("name");
        }
    }
}
