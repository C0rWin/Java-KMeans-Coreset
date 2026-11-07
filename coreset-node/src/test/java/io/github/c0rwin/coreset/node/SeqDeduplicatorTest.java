package io.github.c0rwin.coreset.node;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;

class SeqDeduplicatorTest {

    @Test
    void compactsCommittedPrefixAndKeepsGaps() {
        SeqDeduplicator d = new SeqDeduplicator(Long.MAX_VALUE);
        for (long seq : new long[] {2, 0, 3}) {
            assertEquals(SeqDeduplicator.Claim.CLAIMED, d.claim(seq));
            d.commit(seq);
        }
        assertEquals(1, d.floor()); // 1 is missing, so 2 and 3 stay tracked
        assertEquals(2, d.tracked());
        assertEquals(SeqDeduplicator.Claim.DUPLICATE, d.claim(0));
        assertEquals(SeqDeduplicator.Claim.DUPLICATE, d.claim(3));
        assertEquals(SeqDeduplicator.Claim.CLAIMED, d.claim(1));
        d.commit(1);
        assertEquals(4, d.floor());
        assertEquals(0, d.tracked());
    }

    @Test
    void pendingClaimBlocksConcurrentRetryAndReleaseAllowsIt() {
        SeqDeduplicator d = new SeqDeduplicator(Long.MAX_VALUE);
        assertEquals(SeqDeduplicator.Claim.CLAIMED, d.claim(0));
        assertEquals(SeqDeduplicator.Claim.IN_FLIGHT, d.claim(0));
        d.release(0); // e.g. the queue was full
        assertEquals(SeqDeduplicator.Claim.CLAIMED, d.claim(0));
        d.commit(0);
        assertEquals(SeqDeduplicator.Claim.DUPLICATE, d.claim(0));
        assertEquals(1, d.floor());
    }

    @Test
    void concurrentClaimsCommitEachSeqExactlyOnce() throws Exception {
        SeqDeduplicator d = new SeqDeduplicator(Long.MAX_VALUE);
        int threads = 8;
        int seqs = 20_000;
        CyclicBarrier start = new CyclicBarrier(threads);
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            List<Future<Long>> wins = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final int offset = t;
                wins.add(pool.submit(() -> {
                    start.await();
                    long won = 0;
                    // Every thread tries every seq, starting at a different point to maximise overlap races.
                    for (int i = 0; i < seqs; i++) {
                        long seq = (i + (long) offset * seqs / threads) % seqs;
                        SeqDeduplicator.Claim c;
                        while ((c = d.claim(seq)) == SeqDeduplicator.Claim.IN_FLIGHT) {
                            Thread.onSpinWait();
                        }
                        if (c == SeqDeduplicator.Claim.CLAIMED) {
                            d.commit(seq);
                            won++;
                        }
                    }
                    return won;
                }));
            }
            long total = 0;
            for (Future<Long> f : wins) {
                total += f.get(60, TimeUnit.SECONDS);
            }
            assertEquals(seqs, total);
            assertEquals(seqs, d.floor());
            assertEquals(0, d.tracked());
        } finally {
            pool.shutdownNow();
        }
    }
}
