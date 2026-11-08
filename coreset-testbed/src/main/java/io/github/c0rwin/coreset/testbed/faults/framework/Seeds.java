package io.github.c0rwin.coreset.testbed.faults.framework;

import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import java.nio.charset.StandardCharsets;
import java.util.SplittableRandom;

/**
 * Seed streams of one run. Every experiment keeps data, sampling, packing and channel randomness on separate streams
 * keyed by a purpose string and two indices, so results do not depend on the thread count or on the order of tasks.
 */
final class Seeds {

    private Seeds() {
    }

    static long of(long base, String purpose, long a, long b) {
        return MergeRandomness.derive(base, hash(purpose), a, b);
    }

    static SplittableRandom rng(long base, String purpose, long a, long b) {
        return new SplittableRandom(of(base, purpose, a, b));
    }

    /** FNV-1a over UTF-8. */
    static long hash(String s) {
        long h = 0xcbf29ce484222325L;
        for (byte x : s.getBytes(StandardCharsets.UTF_8)) {
            h = (h ^ (x & 0xff)) * 0x100000001b3L;
        }
        return h;
    }
}
