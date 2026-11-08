package io.github.c0rwin.coreset.node;

import java.util.Objects;

/**
 * How a session's snapshot is shaped into a message of records and packets ({@link NodeMessage}).
 *
 * @param packetization    record form; {@link Packetization#SUMMARY} is the non-resilient baseline
 * @param size             draws m for {@link Packetization#sampled()} forms, cells for {@link Packetization#cells()}
 *                         forms; ignored (0) for SUMMARY
 * @param k                centers of the sensitivity seeding of the sampled forms; ignored otherwise
 * @param recordsPerPacket records per packet B (the last packet may hold fewer)
 * @param lloydIterations  Lloyd steps of the seeding (sampled forms) or of the cell partition (cell forms)
 */
public record MessageSpec(Packetization packetization, int size, int k, int recordsPerPacket, int lloydIterations) {

    public static final int DEFAULT_RECORDS_PER_PACKET = 16;
    public static final int MAX_LLOYD_ITERATIONS = 1_000;

    /** The baseline: the summary as is, {@link #DEFAULT_RECORDS_PER_PACKET} records per packet. */
    public static final MessageSpec SUMMARY =
            new MessageSpec(Packetization.SUMMARY, 0, 0, DEFAULT_RECORDS_PER_PACKET, 0);

    public MessageSpec {
        Objects.requireNonNull(packetization, "packetization");
        if (recordsPerPacket < 1) {
            throw new IllegalArgumentException("recordsPerPacket must be >= 1: " + recordsPerPacket);
        }
        if (lloydIterations < 0 || lloydIterations > MAX_LLOYD_ITERATIONS) {
            throw new IllegalArgumentException("lloydIterations must be in [0, " + MAX_LLOYD_ITERATIONS + "]: "
                    + lloydIterations);
        }
        if (packetization != Packetization.SUMMARY && size < 1) {
            throw new IllegalArgumentException(packetization.id() + " needs size >= 1: " + size);
        }
        if (packetization.sampled() && k < 1) {
            throw new IllegalArgumentException(packetization.id() + " needs k >= 1: " + k);
        }
    }

    public static MessageSpec perDraw(int draws, int k, int recordsPerPacket) {
        return new MessageSpec(Packetization.PER_DRAW, draws, k, recordsPerPacket, 0);
    }

    public static MessageSpec merged(int draws, int k, int recordsPerPacket) {
        return new MessageSpec(Packetization.MERGED, draws, k, recordsPerPacket, 0);
    }

    public static MessageSpec cfTriples(int cells, int recordsPerPacket, int lloydIterations) {
        return new MessageSpec(Packetization.MEAN_CF_TRIPLES, cells, 0, recordsPerPacket, lloydIterations);
    }

    public static MessageSpec globalOffset(int cells, int recordsPerPacket, int lloydIterations) {
        return new MessageSpec(Packetization.MEAN_GLOBAL_OFFSET, cells, 0, recordsPerPacket, lloydIterations);
    }
}
