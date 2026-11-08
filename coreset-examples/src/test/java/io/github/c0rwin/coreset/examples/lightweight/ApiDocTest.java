package io.github.c0rwin.coreset.examples.lightweight;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.WeightedPoint;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Test;

/** docs/api.md shows exactly the compiled regions of ApiDocSnippets, and the runnable ones work. */
class ApiDocTest {

    private static final Path DOC = Path.of("..", "docs", "api.md");
    private static final Path SNIPPETS =
            Path.of("src/test/java/io/github/c0rwin/coreset/examples/lightweight/ApiDocSnippets.java");

    private static String dedent(List<String> lines) {
        int indent = lines.stream().filter(l -> !l.isBlank())
                .mapToInt(l -> l.length() - l.stripLeading().length()).min().orElse(0);
        StringBuilder out = new StringBuilder();
        for (String l : lines) {
            out.append(l.isBlank() ? "" : l.substring(indent)).append('\n');
        }
        return out.toString();
    }

    @Test
    void everyJavaBlockOfTheDocIsACompiledRegion() throws Exception {
        Map<String, String> regions = new LinkedHashMap<>();
        List<String> current = null;
        String name = null;
        for (String line : Files.readAllLines(SNIPPETS)) {
            String t = line.strip();
            if (t.startsWith("// tag::")) {
                name = t.substring(8, t.indexOf('['));
                current = new ArrayList<>();
            } else if (t.startsWith("// end::")) {
                regions.put(name, dedent(current));
                current = null;
            } else if (current != null) {
                current.add(line);
            }
        }
        List<String> blocks = new ArrayList<>();
        Matcher m = Pattern.compile("```java\\n(.*?)```", Pattern.DOTALL).matcher(Files.readString(DOC));
        while (m.find()) {
            blocks.add(m.group(1));
        }
        assertEquals(List.copyOf(regions.values()), blocks, "docs/api.md java blocks vs ApiDocSnippets regions");
    }

    @Test
    void theRunnableSnippetsRun() throws Exception {
        List<WeightedPoint> points = Blobs.points(12_000, 12);
        Coreset full = Coreset.of(points);
        assertEquals(8, ApiDocSnippets.embed(points).centers().size());
        List<List<WeightedPoint>> batches = new ArrayList<>();
        for (int from = 0; from < points.size(); from += 1_500) {
            batches.add(points.subList(from, from + 1_500));
        }
        Coreset streamed = ApiDocSnippets.stream(batches);
        Coreset distributed = ApiDocSnippets.distributed(points);
        for (Coreset c : List.of(streamed, distributed)) {
            double[] quality = ApiDocSnippets.evaluate(full, c);
            assertTrue(quality[0] < LightweightCoresetTest.MAX_STREAM_DISTORTION, "distortion " + quality[0]);
            assertTrue(quality[1] < LightweightCoresetTest.MAX_WEIGHT_REL_ERR, "weight rel err " + quality[1]);
        }
    }
}
