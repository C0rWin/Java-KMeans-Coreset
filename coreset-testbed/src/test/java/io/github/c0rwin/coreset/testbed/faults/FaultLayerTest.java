package io.github.c0rwin.coreset.testbed.faults;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.PointSet;
import io.github.c0rwin.coreset.core.algo.MeanCoreset;
import io.github.c0rwin.coreset.core.algo.SampledCoreset;
import io.github.c0rwin.coreset.core.algo.Seeding;
import io.github.c0rwin.coreset.core.algo.SensitivityCoreset;
import io.github.c0rwin.coreset.core.algo.UniformCoreset;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import io.github.c0rwin.coreset.testbed.data.Dataset;
import io.github.c0rwin.coreset.testbed.data.SyntheticData;
import io.github.c0rwin.coreset.testbed.faults.Partitions.Kind;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.SplittableRandom;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class FaultLayerTest {

    static Instance inst;

    @BeforeAll
    static void prepare() {
        Dataset ds = Dataset.of("g", SyntheticData.gaussianMixture(3000, 4, 5, 1.0, 1.5, 100.0, 3L), "", "generated",
                "test", List.of(), Optional.empty());
        inst = Instance.prepare("g", ds, 5, 2, 10, 2, 11L, null);
    }

    @Test
    void bernoulliLossRateAndNesting() {
        int n = 200_000;
        boolean[] a = new boolean[n];
        boolean[] b = new boolean[n];
        Channel.bernoulli(0.1).erase(a, n, new SplittableRandom(5));
        Channel.bernoulli(0.3).erase(b, n, new SplittableRandom(5));
        int la = 0;
        int lb = 0;
        for (int i = 0; i < n; i++) {
            la += a[i] ? 1 : 0;
            lb += b[i] ? 1 : 0;
            assertTrue(!a[i] || b[i], "loss at 0.1 must imply loss at 0.3");
        }
        assertEquals(0.1, (double) la / n, 0.004);
        assertEquals(0.3, (double) lb / n, 0.004);
    }

    @Test
    void gilbertElliottStationaryRateAndBurstLength() {
        int n = 400_000;
        boolean[] lost = new boolean[n];
        Channel.gilbertElliott(0.2, 20).erase(lost, n, new SplittableRandom(9));
        int l = 0;
        int bursts = 0;
        for (int i = 0; i < n; i++) {
            l += lost[i] ? 1 : 0;
            bursts += lost[i] && (i == 0 || !lost[i - 1]) ? 1 : 0;
        }
        assertEquals(0.2, (double) l / n, 0.02);
        assertEquals(20.0, (double) l / bursts, 1.5);
    }

    @Test
    void realizedCountDecoderIsUnbiasedOnAFixedQuery() {
        Sample smp = Sample.sensitivity(inst.points, Partitions.identity(inst.n), inst.k, 300,
                new SplittableRandom(1));
        int cols = inst.candidates();
        double[] unit = F3Erasure.unitCosts(inst, smp);
        int[] all = smp.counts.clone();
        double[] truth = F3Erasure.receivedCost(all, unit, cols);
        Packets pk = Packets.of(Packets.Layout.PER_DRAW, smp, inst.cluster, new SplittableRandom(2));
        boolean[] lost = new boolean[pk.packets()];
        int[] recv = new int[smp.size()];
        int reps = 20_000;
        double realized = 0.0;
        double prior = 0.0;
        double none = 0.0;
        SplittableRandom rng = new SplittableRandom(3);
        int j = inst.ref;
        for (int t = 0; t < reps; t++) {
            Channel.bernoulli(0.3).erase(lost, pk.packets(), rng);
            int got = pk.receive(lost, recv);
            double c = F3Erasure.receivedCost(recv, unit, cols)[j];
            realized += c * F3Erasure.scale("ht-realized", 0.3, smp.m, got);
            prior += c * F3Erasure.scale("ht-prior", 0.3, smp.m, got);
            none += c;
        }
        assertEquals(1.0, realized / reps / truth[j], 0.005);
        assertEquals(1.0, prior / reps / truth[j], 0.005);
        assertEquals(0.7, none / reps / truth[j], 0.005);
    }

    @Test
    void partitionsCoverEveryRowOnce() {
        for (Kind kind : Kind.values()) {
            int[][] parts = Partitions.split(kind, inst.cluster, inst.points, 7, new SplittableRandom(4));
            int[] seen = new int[inst.n];
            for (int[] p : parts) {
                for (int r : p) {
                    seen[r]++;
                }
            }
            for (int c : seen) {
                assertEquals(1, c, kind.id());
            }
            if (kind != Kind.ZIPF_SIZES) {
                for (int[] p : parts) {
                    assertTrue(Math.abs(p.length - inst.n / 7.0) <= 1.0, kind.id());
                }
            }
        }
        int[][] zipf = Partitions.split(Kind.ZIPF_SIZES, inst.cluster, inst.points, 7, new SplittableRandom(4));
        int max = Arrays.stream(zipf).mapToInt(p -> p.length).max().orElseThrow();
        int min = Arrays.stream(zipf).mapToInt(p -> p.length).min().orElseThrow();
        assertTrue(max >= 5 * min, "zipf sizes must be unequal");
        // cluster-sorted parts: every cluster occupies a contiguous run of nodes
        int[][] sorted = Partitions.split(Kind.CLUSTER_SORTED, inst.cluster, inst.points, 10, new SplittableRandom(4));
        for (int c = 0; c < inst.clusterSize.length; c++) {
            int first = -1;
            int last = -1;
            for (int v = 0; v < sorted.length; v++) {
                for (int r : sorted[v]) {
                    if (inst.cluster[r] == c) {
                        first = first < 0 ? v : first;
                        last = v;
                    }
                }
            }
            int span = last - first + 1;
            assertTrue(span <= inst.clusterSize[c] / (inst.n / 10) + 2, "cluster " + c + " spans " + span);
        }
    }

    @Test
    void coresetReplicationLossRateIsFToTheRPlusOne() {
        int n = 16;
        int[] ring = Partitions.permutation(n, new SplittableRandom(6));
        int[] pos = new int[n];
        for (int i = 0; i < n; i++) {
            pos[ring[i]] = i;
        }
        double f = 0.3;
        SplittableRandom rng = new SplittableRandom(7);
        int trials = 40_000;
        int[] lost = new int[3];
        for (int t = 0; t < trials; t++) {
            boolean[] failed = NodeFaults.Loss.iid(f).realize(n, rng);
            for (int r = 0; r <= 2; r++) {
                for (int v = 0; v < n; v++) {
                    boolean any = false;
                    for (int c = 0; c <= r; c++) {
                        any |= !failed[ring[(pos[v] + c) % n]];
                    }
                    assertEquals(any, F2PostLoss.delivered(ring, pos, v, r, failed));
                    lost[r] += any ? 0 : 1;
                }
            }
        }
        for (int r = 0; r <= 2; r++) {
            assertEquals(Math.pow(f, r + 1), (double) lost[r] / (trials * n), 0.1 * Math.pow(f, r + 1));
        }
    }

    @Test
    void fixedLossLosesExactlyPhiN() {
        boolean[] lost = NodeFaults.Loss.fixed(0.25).realize(16, new SplittableRandom(8));
        assertEquals(4, Placement.count(lost));
    }

    @Test
    void samplesReplayTheCoreAlgorithms() {
        int[] rows = Partitions.identity(inst.n);
        SampledCoreset sc = new SensitivityCoreset(inst.k, 200, Seeding.kMeansPlusPlus(0))
                .compressWithProvenance(Coreset.of(inst.points), new SplittableRandom(21), MetricsSink.NOOP);
        Sample s = Sample.sensitivity(inst.points, rows, inst.k, 200, new SplittableRandom(21));
        assertCoresetEquals(sc.coreset(), s);
        assertArrayEquals(sc.drawCounts(), s.counts);
        Coreset u = new UniformCoreset(150).compress(Coreset.of(inst.points), new SplittableRandom(22),
                MetricsSink.NOOP);
        assertCoresetEquals(u, Sample.uniform(inst.points, rows, 150, new SplittableRandom(22)));
    }

    private static void assertCoresetEquals(Coreset expected, Sample s) {
        PointSet got = PointSet.of(s.materialize(inst.points, 1.0));
        assertEquals(expected.size(), got.size());
        for (int i = 0; i < got.size(); i++) {
            assertEquals(expected.pointSet().vector(i), got.vector(i));
            assertEquals(expected.pointSet().weight(i), got.weight(i), 1e-12 * got.weight(i));
        }
    }

    @Test
    void meanCellsReplayMeanCoresetAndSplitItsOffset() {
        Coreset mc = new MeanCoreset(30, 3).compress(Coreset.of(inst.points), new SplittableRandom(31),
                MetricsSink.NOOP);
        MeanCells cells = MeanCells.build(inst.points, 30, 3, new SplittableRandom(31));
        assertEquals(mc.pointSet(), PointSet.of(cells.means));
        assertEquals(mc.offset(), cells.offset);
        assertEquals(cells.offset, Arrays.stream(cells.phi).sum(), 1e-9 * cells.offset);
    }

    @Test
    void replicationCopiesAndFecSizes() {
        assertEquals(1, F1NodeLoss.copies(1e-4, 1e-3, 0.1, 16));
        assertEquals(2, F1NodeLoss.copies(5e-3, 1e-3, 0.1, 16)); // log_10(5) < 1
        assertEquals(3, F1NodeLoss.copies(5e-2, 1e-3, 0.1, 16));
        int t = F3Erasure.fecPackets(200, 0.1, 0.01);
        assertTrue(t > 200 / 0.9 && t < 200 / 0.9 + 40, "fec packets " + t);
        assertEquals(6, F6Tree.depth(64, 2));
        assertEquals(1, F6Tree.depth(64, 64));
    }

    @Test
    void realInstanceIdsRoundTrip() {
        assertEquals(new RealData.Id("higgs", 1_000_000, 10), RealData.Id.parse("higgs-1m-k10"));
        assertEquals(new RealData.Id("kddcup99", 500_000, 25), RealData.Id.parse("kddcup99-500k-k25"));
        assertEquals(new RealData.Id("covertype", 0, 10), RealData.Id.parse("covertype-k10"));
        assertEquals("census1990-500k-k25", RealData.Id.of("census1990", 500_000, 25));
        assertEquals("higgs-1m-k10", RealData.Id.of("higgs", 1_000_000, 10));
        assertTrue(RealData.isReal("covertype-k25"));
        assertTrue(!RealData.isReal("zipf-heavy") && !RealData.isReal("zipf-400k"));
    }
}
