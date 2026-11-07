package io.github.c0rwin.coreset.node;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Lock-free exactly-once filter over non-negative batch sequence numbers.
 *
 * <p>Memory stays bounded for (mostly) dense seqs: every seq below {@code floor} is known committed, and only seqs
 * at or above it are kept in a map, as PENDING (claimed, enqueue in progress) or COMMITTED. Committing the seq at the
 * floor advances it over the committed run and drops those entries. A gap that is never filled pins the floor.
 *
 * <p>Invariant: the floor only passes a seq whose entry is COMMITTED, so a PENDING claim can never be overtaken.
 */
final class SeqDeduplicator {

    enum Claim { CLAIMED, DUPLICATE, IN_FLIGHT }

    private final AtomicLong floor = new AtomicLong();
    private final ConcurrentHashMap<Long, Boolean> above = new ConcurrentHashMap<>();

    /** Reserves {@code seq}; a {@link Claim#CLAIMED} result must be followed by {@link #commit} or {@link #release}. */
    Claim claim(long seq) {
        if (seq < floor.get()) {
            return Claim.DUPLICATE;
        }
        Boolean prev = above.putIfAbsent(seq, Boolean.FALSE);
        if (prev != null) {
            return prev ? Claim.DUPLICATE : Claim.IN_FLIGHT;
        }
        // The floor may have passed seq (committed and compacted) between the first check and putIfAbsent.
        if (seq < floor.get()) {
            above.remove(seq, Boolean.FALSE);
            return Claim.DUPLICATE;
        }
        return Claim.CLAIMED;
    }

    void commit(long seq) {
        above.put(seq, Boolean.TRUE);
        for (;;) {
            long f = floor.get();
            if (!Boolean.TRUE.equals(above.get(f))) {
                return;
            }
            if (floor.compareAndSet(f, f + 1)) {
                above.remove(f);
            }
        }
    }

    void release(long seq) {
        above.remove(seq, Boolean.FALSE);
    }

    /** Seqs tracked above the floor (memory footprint indicator). */
    int tracked() {
        return above.size();
    }

    long floor() {
        return floor.get();
    }
}
