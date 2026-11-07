package io.github.c0rwin.coreset.testbed.run;

import java.nio.charset.StandardCharsets;

/** Deterministic seed derivation (SplitMix64 finaliser over FNV-1a string hashes). */
final class Seeds {

    private Seeds() {
    }

    static long mix64(long z) {
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    /** Order-sensitive combination of seeds. */
    static long mix(long... parts) {
        long h = 0x9E3779B97F4A7C15L;
        for (long p : parts) {
            h = mix64(h ^ mix64(p + 0x9E3779B97F4A7C15L));
        }
        return h;
    }

    static long hash(String s) {
        long h = 0xcbf29ce484222325L;
        for (byte b : s.getBytes(StandardCharsets.UTF_8)) {
            h ^= b & 0xff;
            h *= 0x100000001b3L;
        }
        return h;
    }
}
