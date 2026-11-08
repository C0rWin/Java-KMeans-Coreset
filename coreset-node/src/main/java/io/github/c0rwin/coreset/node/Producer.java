package io.github.c0rwin.coreset.node;

/**
 * Per-producer state of a session: seqs are deduplicated per {@code (producer, seq)}, so independent producers (for
 * example two coordinator runs) may both number their batches from 0 without colliding.
 */
final class Producer {

    /** Producer of callers that do not name one; salt 0 keeps its batch generators and routing as before. */
    static final String DEFAULT_ID = "";

    final String id;
    /** Mixed into the batch seed so equal seqs of different producers get independent generators. */
    final long salt;
    final SeqDeduplicator dedupe;

    Producer(String id, int seqWindow) {
        this.id = id;
        this.salt = id.isEmpty() ? 0L : Seeds.mix64(fnv1a64(id));
        this.dedupe = new SeqDeduplicator(seqWindow);
    }

    /** Worker for {@code seq}: the salt rotates the start so concurrent producers spread over workers. */
    int worker(long seq, int workers) {
        return (int) Math.floorMod(seq + salt, (long) workers);
    }

    private static long fnv1a64(String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0, n = s.length(); i < n; i++) {
            h = (h ^ s.charAt(i)) * 0x100000001b3L;
        }
        return h;
    }
}
