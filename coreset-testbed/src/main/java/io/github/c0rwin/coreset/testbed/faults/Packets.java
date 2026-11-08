package io.github.c0rwin.coreset.testbed.faults;

import java.util.Arrays;
import java.util.Locale;
import java.util.random.RandomGenerator;

/**
 * A packetization of a merged sample: transmission units (one draw, or one merged item with its count) grouped into
 * packets. Losing a packet loses its units; {@link #receive} turns a loss pattern into received draws per item.
 */
final class Packets {

    enum Layout {
        /** One draw per packet, draws in random order (split: duplicates not merged). */
        PER_DRAW(1),
        /** One item (point and its draw count) per packet, in coreset order. */
        MERGED(1),
        /** B draws per packet after a random permutation. */
        RANDOM_B8(8),
        RANDOM_B64(64),
        /** B draws per packet, draws sorted by reference cluster: content-dependent packing. */
        SORTED_B8(8);

        final int perPacket;

        Layout(int perPacket) {
            this.perPacket = perPacket;
        }

        String id() {
            return name().toLowerCase(Locale.ROOT).replace('_', '-');
        }
    }

    final Layout layout;
    final int[] unitItem;
    final int[] unitDraws;
    final int[] packetStart;

    private Packets(Layout layout, int[] unitItem, int[] unitDraws, int[] packetStart) {
        this.layout = layout;
        this.unitItem = unitItem;
        this.unitDraws = unitDraws;
        this.packetStart = packetStart;
    }

    int packets() {
        return packetStart.length - 1;
    }

    static Packets of(Layout layout, Sample s, int[] cluster, RandomGenerator rng) {
        if (layout == Layout.MERGED) {
            int[] start = new int[s.size() + 1];
            for (int i = 0; i <= s.size(); i++) {
                start[i] = i;
            }
            return new Packets(layout, Partitions.identity(s.size()), s.counts.clone(), start);
        }
        int[] seq = s.drawSequence(rng);
        if (layout == Layout.SORTED_B8) {
            Integer[] boxed = Arrays.stream(seq).boxed().toArray(Integer[]::new);
            Arrays.sort(boxed, (a, b) -> {
                int c = Integer.compare(cluster[s.rows[a]], cluster[s.rows[b]]);
                return c != 0 ? c : Integer.compare(s.rows[a], s.rows[b]);
            });
            for (int i = 0; i < seq.length; i++) {
                seq[i] = boxed[i];
            }
        }
        int b = layout.perPacket;
        int packets = (seq.length + b - 1) / b;
        int[] start = new int[packets + 1];
        for (int p = 0; p <= packets; p++) {
            start[p] = Math.min(seq.length, p * b);
        }
        int[] ones = new int[seq.length];
        Arrays.fill(ones, 1);
        return new Packets(layout, seq, ones, start);
    }

    /** Received draws per item into {@code recv} (cleared first); returns the total received draws m'. */
    int receive(boolean[] lost, int[] recv) {
        Arrays.fill(recv, 0);
        int got = 0;
        for (int p = 0; p < packets(); p++) {
            if (!lost[p]) {
                for (int u = packetStart[p]; u < packetStart[p + 1]; u++) {
                    recv[unitItem[u]] += unitDraws[u];
                    got += unitDraws[u];
                }
            }
        }
        return got;
    }

    /** Bytes on the wire: one header per packet, one point payload per unit (merged units also carry a count). */
    long bytes(long pointBytes) {
        long extra = layout == Layout.MERGED ? 4L : 0L;
        return (long) packets() * Measure.PACKET_HEADER + (long) unitItem.length * (pointBytes + extra);
    }
}
