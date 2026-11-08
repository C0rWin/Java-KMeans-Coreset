package io.github.c0rwin.coreset.core;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * Serializable recipe for a {@link CoresetAlgorithm}, shipped to the JVMs that compress (Spark executors, Flink task
 * managers) and called there. Use {@link RegisteredAlgorithm} to resolve a registered id on each JVM, or implement
 * this interface for an algorithm supplied in code; the implementation and everything it references must then be on
 * the job's class path and serializable.
 */
@FunctionalInterface
public interface CoresetAlgorithmFactory extends Serializable {

    /**
     * May be called many times, on any JVM of the job.
     *
     * @return a non-null, stateless, thread-safe algorithm
     * @throws IllegalArgumentException if the recipe is invalid
     */
    CoresetAlgorithm create();

    /**
     * What this recipe builds, as stored in engine checkpoints (Flink operator state, the Spark stream sink) so that
     * state written with one algorithm is never resumed with another. The default is the class name plus the SHA-256
     * of the serialized form, which changes whenever a field value or the class's serialized shape does; override it
     * to keep checkpoints compatible across such changes.
     *
     * @throws IllegalStateException if the recipe cannot be serialized
     */
    default String identity() {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        try (ObjectOutputStream out = new ObjectOutputStream(bytes)) {
            out.writeObject(this);
        } catch (IOException e) {
            throw new IllegalStateException("algorithm factory " + getClass().getName() + " is not serializable", e);
        }
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes.toByteArray());
            return getClass().getName() + "@sha256:" + HexFormat.of().formatHex(digest);
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 is required of every Java platform", e);
        }
    }
}
