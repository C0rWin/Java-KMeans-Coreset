package io.github.c0rwin.coreset.core;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.StringWriter;
import java.lang.module.Configuration;
import java.lang.module.ModuleDescriptor;
import java.lang.module.ModuleFinder;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.ServiceLoader;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.Collectors;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.ToolProvider;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The compiled module descriptor, exercised the way a modular application would use it. */
class ModuleBoundaryTest {

    private static final String MODULE = "io.github.c0rwin.coreset.core";
    private static final Set<String> API = Set.of(MODULE, MODULE + ".algo", MODULE + ".kmeans", MODULE + ".metrics",
            MODULE + ".sampling", MODULE + ".stream", MODULE + ".validate");

    private static Path moduleDir() throws URISyntaxException {
        return Path.of(CoresetAlgorithms.class.getProtectionDomain().getCodeSource().getLocation().toURI());
    }

    @Test
    void exportsTheApiPackagesOnlyAndProvidesTheBuiltIns() throws Exception {
        ModuleDescriptor d = ModuleFinder.of(moduleDir()).find(MODULE).orElseThrow().descriptor();
        assertEquals(API, d.exports().stream().map(ModuleDescriptor.Exports::source).collect(Collectors.toSet()));
        assertTrue(d.exports().stream().noneMatch(ModuleDescriptor.Exports::isQualified));
        assertTrue(d.packages().contains(MODULE + ".internal"));
        assertEquals(Set.of(CoresetAlgorithmProvider.class.getName()), d.uses());
        assertEquals(List.of("io.github.c0rwin.coreset.core.algo.BuiltInAlgorithms"),
                d.provides().iterator().next().providers());
    }

    @Test
    void aModuleLayerResolvesTheBuiltInsThroughProvides() throws Exception {
        Configuration cf = ModuleLayer.boot().configuration()
                .resolve(ModuleFinder.of(moduleDir()), ModuleFinder.of(), Set.of(MODULE));
        ModuleLayer layer = ModuleLayer.boot().defineModulesWithOneLoader(cf, ClassLoader.getPlatformClassLoader());
        Module module = layer.findModule(MODULE).orElseThrow();
        assertFalse(module.isExported(MODULE + ".internal"));
        ClassLoader loader = layer.findLoader(MODULE);
        Class<?> registry = loader.loadClass(CoresetAlgorithms.class.getName());
        assertNotEquals(CoresetAlgorithms.class, registry); // the layer's own copy, not the class-path one
        Object algorithms = registry.getMethod("load", ClassLoader.class).invoke(null, loader);
        @SuppressWarnings("unchecked")
        SortedSet<String> ids = (SortedSet<String>) registry.getMethod("ids").invoke(algorithms);
        assertEquals(new TreeSet<>(Set.of("mean", "sensitivity", "uniform")), ids);
        Class<?> service = loader.loadClass(CoresetAlgorithmProvider.class.getName());
        assertEquals(1, ServiceLoader.load(layer, service).stream().count());
    }

    @Test
    void aConsumerModuleCompilesAgainstTheApiButNotTheInternals(@TempDir Path dir) throws Exception {
        Path src = Files.createDirectories(dir.resolve("src/consumer/app"));
        Files.writeString(dir.resolve("src/consumer/module-info.java"),
                "module consumer { requires io.github.c0rwin.coreset.core; }");
        Files.writeString(src.resolve("Api.java"), "package app; class Api { Object a = "
                + "io.github.c0rwin.coreset.core.CoresetAlgorithms.load().create(\"uniform\", java.util.Map.of(\"m\", "
                + "\"4\")); }");
        assertEquals("", compile(dir, src.resolve("Api.java")));
        Files.writeString(src.resolve("Internal.java"), "package app; class Internal { Object k = "
                + "new io.github.c0rwin.coreset.core.internal.RowCandidates(1, 1, false); }");
        String errors = compile(dir, src.resolve("Internal.java"));
        assertTrue(errors.contains("package io.github.c0rwin.coreset.core.internal is not visible")
                || errors.contains("does not export io.github.c0rwin.coreset.core.internal"), errors);
    }

    private static String compile(Path dir, Path source) throws Exception {
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        boolean ok = javac.getTask(new StringWriter(), null, diagnostics,
                List.of("--module-path", moduleDir().toString(), "--module-source-path", dir.resolve("src").toString(),
                        "-d", dir.resolve("out").toString()),
                null, javac.getStandardFileManager(null, null, null).getJavaFileObjects(
                        dir.resolve("src/consumer/module-info.java"), source)).call();
        return ok ? "" : diagnostics.getDiagnostics().stream().map(d -> d.getMessage(null))
                .collect(Collectors.joining("\n", "failed: ", ""));
    }
}
