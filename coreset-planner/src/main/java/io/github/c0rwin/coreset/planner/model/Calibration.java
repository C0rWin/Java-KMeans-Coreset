package io.github.c0rwin.coreset.planner.model;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.TreeMap;

/**
 * Empirical constants of the resilience model, read from a versioned JSON file whose {@code provenance} names the
 * experiment run they were fitted on. The bundled file is {@code calibration-v1.json} next to this package.
 *
 * @param version file version, echoed in every plan
 * @param provenance free-form provenance entries
 * @param constants name to value
 */
public record Calibration(String version, Map<String, String> provenance, Map<String, Double> constants) {

    public static final String BUNDLED = "/io/github/c0rwin/coreset/planner/calibration-v1.json";

    public static Calibration bundled() {
        try (InputStream in = Calibration.class.getResourceAsStream(BUNDLED)) {
            if (in == null) {
                throw new IllegalStateException("missing resource " + BUNDLED);
            }
            return parse(new ObjectMapper().readTree(in));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static Calibration load(Path file) throws IOException {
        return parse(new ObjectMapper().readTree(Files.readAllBytes(file)));
    }

    static Calibration parse(JsonNode root) {
        Map<String, String> prov = new TreeMap<>();
        root.path("provenance").fields().forEachRemaining(e -> prov.put(e.getKey(), e.getValue().asText()));
        Map<String, Double> consts = new TreeMap<>();
        root.path("constants").fields().forEachRemaining(e -> {
            JsonNode v = e.getValue().isObject() ? e.getValue().get("value") : e.getValue();
            if (v == null || !v.isNumber()) {
                throw new IllegalArgumentException("calibration constant " + e.getKey() + " has no numeric value");
            }
            consts.put(e.getKey(), v.asDouble());
        });
        String version = root.path("version").asText(null);
        if (version == null) {
            throw new IllegalArgumentException("calibration file has no version");
        }
        return new Calibration(version, Map.copyOf(prov), Map.copyOf(consts));
    }

    public double get(String name) {
        Double v = constants.get(name);
        if (v == null) {
            throw new IllegalArgumentException("calibration " + version + " lacks constant " + name);
        }
        return v;
    }
}
