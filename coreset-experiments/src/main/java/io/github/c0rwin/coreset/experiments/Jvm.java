package io.github.c0rwin.coreset.experiments;

import java.io.File;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.stream.Collectors;

/** How child JVMs (cells, nodes, Spark executors) are launched: this JVM's java binary and class path. */
public final class Jvm {

    private Jvm() {
    }

    public static String java() {
        return Path.of(System.getProperty("java.home"), "bin", "java").toString();
    }

    /** This JVM's class path with absolute entries: children may run in another directory (Spark work dirs). */
    public static String classPath() {
        return Arrays.stream(System.getProperty("java.class.path").split(File.pathSeparator))
                .filter(s -> !s.isEmpty())
                .map(s -> Path.of(s).toAbsolutePath().toString())
                .collect(Collectors.joining(File.pathSeparator));
    }
}
