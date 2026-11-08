package io.github.c0rwin.coreset.testbed.faults;

import io.github.c0rwin.coreset.core.stream.MergeRandomness;
import java.nio.charset.StandardCharsets;
import java.util.SplittableRandom;

/**
 * Independent seed streams for one lab run: data, partitions, sampling, faults and solvers each get their own stream,
 * so a defense compared against another sees the same partition, sample and fault realization.
 */
final class LabSeeds {

    private LabSeeds() {
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
