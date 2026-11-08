package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.PointBlock;
import java.util.Objects;

/**
 * A node's snapshot as a message (model.tex): records in send order plus the header fields a receiver needs to decode
 * any subset of them. Packet {@code j} carries records {@code [j*B, min((j+1)*B, records))}, so the packet layout is
 * a function of the record count and B alone.
 *
 * @param packetization    record form
 * @param stream           sampling stream the draws came from (replicas of one group use distinct streams)
 * @param draws            draws m (sampled forms), cells (cell forms) or records (SUMMARY): the realized-count
 *                         decoder's numerator
 * @param recordsPerPacket B
 * @param nodeWeight       input weight the summary stands for (survivor reweighting)
 * @param nodePoints       input points accepted by the node
 * @param complete         the summary reflected every accepted batch ({@link NodeSnapshot#complete()})
 * @param offset           reliable additive offset: the summary's offset, plus every cell offset for
 *                         MEAN_GLOBAL_OFFSET
 * @param records          points and weights in send order; arrays are shared, not copied
 * @param tags             per record: draw count (MERGED), index of the distinct draw (PER_DRAW); empty otherwise
 * @param cellOffsets      per record: the cell's offset (MEAN_CF_TRIPLES); empty otherwise
 */
public record NodeMessage(Packetization packetization, long stream, int draws, int recordsPerPacket,
                          double nodeWeight, long nodePoints, boolean complete, double offset, PointBlock records,
                          int[] tags, double[] cellOffsets) {

    private static final int[] NO_TAGS = new int[0];
    private static final double[] NO_OFFSETS = new double[0];

    public NodeMessage {
        Objects.requireNonNull(packetization, "packetization");
        Objects.requireNonNull(records, "records");
        tags = tags == null ? NO_TAGS : tags;
        cellOffsets = cellOffsets == null ? NO_OFFSETS : cellOffsets;
        if (recordsPerPacket < 1) {
            throw new IllegalArgumentException("recordsPerPacket must be >= 1: " + recordsPerPacket);
        }
        if (draws < 0 || nodePoints < 0 || !(nodeWeight >= 0.0) || !(offset >= 0.0)) {
            throw new IllegalArgumentException("draws, nodePoints, nodeWeight and offset must be >= 0");
        }
        if (tags.length != 0 && tags.length != records.size()
                || cellOffsets.length != 0 && cellOffsets.length != records.size()) {
            throw new IllegalArgumentException("tags and cell offsets need one entry per record");
        }
    }

    public int recordCount() {
        return records.size();
    }

    public int packets() {
        return (records.size() + recordsPerPacket - 1) / recordsPerPacket;
    }

    /** Sum of the record weights as sent. */
    public double recordWeight() {
        return records.totalWeight();
    }
}
