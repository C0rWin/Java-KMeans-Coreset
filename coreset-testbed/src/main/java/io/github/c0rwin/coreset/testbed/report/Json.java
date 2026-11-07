package io.github.c0rwin.coreset.testbed.report;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.json.JsonReadFeature;
import com.fasterxml.jackson.core.json.JsonWriteFeature;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.databind.json.JsonMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLMapper;
import io.github.c0rwin.coreset.testbed.run.TestbedConfig;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/** JSON/YAML mapping of configs and reports. Non-finite doubles are written as the strings "Infinity"/"NaN". */
public final class Json {

    private static final JsonMapper JSON = JsonMapper.builder()
            .enable(JsonWriteFeature.WRITE_NAN_AS_STRINGS)
            .enable(JsonReadFeature.ALLOW_NON_NUMERIC_NUMBERS)
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS)
            .build();

    private static final YAMLMapper YAML = YAMLMapper.builder()
            .enable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES)
            .enable(JsonParser.Feature.STRICT_DUPLICATE_DETECTION)
            .build();

    private Json() {
    }

    public static ObjectMapper mapper() {
        return JSON;
    }

    public static String toJson(Object value) throws IOException {
        return JSON.writerWithDefaultPrettyPrinter().writeValueAsString(value);
    }

    public static String toJsonLine(Object value) throws IOException {
        return JSON.writeValueAsString(value);
    }

    public static ValidityReport parseReport(String json) throws IOException {
        return JSON.readValue(json, ValidityReport.class);
    }

    public static void writeReport(Path file, ValidityReport report) throws IOException {
        Files.writeString(file, toJson(report) + "\n", StandardCharsets.UTF_8);
    }

    public static ValidityReport readReport(Path file) throws IOException {
        return parseReport(Files.readString(file, StandardCharsets.UTF_8));
    }

    /** Reads {@code .yaml}/{@code .yml} as YAML and anything else as JSON. */
    public static TestbedConfig readConfig(Path file) throws IOException {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);
        ObjectMapper m = name.endsWith(".yaml") || name.endsWith(".yml") ? YAML : JSON;
        return m.readValue(file.toFile(), TestbedConfig.class);
    }

    public static TestbedConfig parseConfig(String text, boolean yaml) throws IOException {
        return (yaml ? YAML : JSON).readValue(text, TestbedConfig.class);
    }

    /** Opens a gzip-compressed JSON-lines writer, one report per line. */
    public static Writer jsonLinesGzip(Path file) throws IOException {
        return new BufferedWriter(new OutputStreamWriter(new GZIPOutputStream(Files.newOutputStream(file), 1 << 16),
                StandardCharsets.UTF_8));
    }

    /** Reads every report from a {@code .jsonl.gz} file. */
    public static List<ValidityReport> readJsonLinesGzip(Path file) throws IOException {
        List<ValidityReport> out = new ArrayList<>();
        try (BufferedReader r = new BufferedReader(new InputStreamReader(
                new GZIPInputStream(Files.newInputStream(file), 1 << 16), StandardCharsets.UTF_8))) {
            for (String line = r.readLine(); line != null; line = r.readLine()) {
                if (!line.isBlank()) {
                    out.add(parseReport(line));
                }
            }
        }
        return out;
    }
}
