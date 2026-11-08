package io.github.c0rwin.coreset.planner.model;

import io.github.c0rwin.coreset.planner.Configuration;
import io.github.c0rwin.coreset.planner.Configuration.Packetization;
import io.github.c0rwin.coreset.planner.EnvironmentProfile;
import io.github.c0rwin.coreset.planner.WorkloadProfile;

/**
 * Wire sizes and transfer times of one hop; a calibrated cost model, not a guarantee.
 * <ul>
 *   <li>Datagrams: IPv4+UDP header plus a 16-byte reliable application header (sender, sequence number, draw count)
 *       per packet. Block FEC adds parity packets per block. RaptorQ (RFC 6330) sends the least number of
 *       source+repair packets whose channel quantile reaches K + 2 received symbols. Early close (LTP-style) streams
 *       until acknowledged: K/(1 - rho) packets plus one bandwidth-delay product of overshoot, plus one RTT.</li>
 *   <li>Reliable: TCP/IPv4 headers per 1460-byte segment, a 5-byte gRPC prefix per message, retransmissions at rate
 *       rho. Time follows the short-flow model of Cardwell, Savage and Anderson (INFOCOM 2000): slow start from an
 *       initial window of 10 segments (RFC 6928) with delayed ACKs until the first loss, then the square-root
 *       throughput of Mathis et al. (CCR 1997) with one congestion event per erasure burst. Timeouts are omitted.</li>
 * </ul>
 */
public final class NetworkCostModel {

    public static final int DATAGRAM_HEADER = 28 + 16;
    public static final int TCP_HEADER = 40;
    public static final int TCP_MSS = 1460;
    public static final int GRPC_PREFIX = 5;
    static final int INITIAL_WINDOW = 10;
    static final double SLOW_START_GROWTH = 1.5; // 1 + 1/b with delayed ACKs (b = 2)
    static final int RAPTORQ_OVERHEAD = 2;

    /** Bytes of one hop and its duration. */
    public record Hop(double senderBytes, double ingressBytes, double seconds) {
    }

    /** Bytes of one record: the point (dense or sparse, whichever is smaller) plus weight and tag fields. */
    public int recordBytes(WorkloadProfile w, Configuration c) {
        return recordBytes(w.density(), w.d(), c.packetization());
    }

    public int recordBytes(double density, int d, Packetization packetization) {
        double nnz = Math.max(1.0, density * d);
        int point = (int) Math.ceil(Math.min(8.0 * d, 12.0 * nnz + 4.0));
        return switch (packetization) {
            case PER_DRAW -> point + 8 + 8;            // weight, draw tag
            case MERGED, SPLIT -> point + 8 + 8 + 4;   // weight, tag, count
            case CELLS -> point + 8;                   // mean, cell weight
            case CELLS_CF -> point + 8 + 8;            // LS, n, SS (mean derived)
        };
    }

    /** Records that fit in one datagram, 0 if a single record does not. */
    public int recordsPerPacket(EnvironmentProfile e, int recordBytes) {
        return Math.max(0, (e.maxMessageBytes() - DATAGRAM_HEADER) / recordBytes);
    }

    /**
     * Wire bytes for {@code records} records from one sender; {@code eta} is the per-sender channel budget used to
     * size RaptorQ repair symbols.
     */
    public double wireBytes(EnvironmentProfile e, Configuration c, int recordBytes, long records, double eta) {
        if (records <= 0) {
            return 0.0;
        }
        if (c.transport().datagram()) {
            int b = recordsPerPacket(e, recordBytes);
            long packets = (records + b - 1) / b;
            double packetBytes = DATAGRAM_HEADER + (double) b * recordBytes;
            double bytes = records * (double) recordBytes + packets * (double) DATAGRAM_HEADER;
            switch (c.transport()) {
                case DATAGRAM_FEC -> {
                    long blocks = (packets + c.fecData() - 1) / c.fecData();
                    bytes += blocks * (double) c.fecParity() * packetBytes;
                }
                case DATAGRAM_RAPTORQ -> bytes += (raptorqPackets(e, packets, eta) - packets) * packetBytes;
                case EARLY_CLOSE -> {
                    double extra = packets * e.rho() / (1.0 - e.rho());
                    double overshoot = Math.ceil(e.bandwidthBytesPerSecond() * e.rttSeconds() / packetBytes);
                    bytes += (extra + (e.rho() > 0.0 ? overshoot : 0.0)) * packetBytes + DATAGRAM_HEADER;
                }
                default -> {
                }
            }
            return bytes;
        }
        int perMessage = Math.max(1, (e.maxMessageBytes() - GRPC_PREFIX) / recordBytes);
        long messages = (records + perMessage - 1) / perMessage;
        double payload = records * (double) recordBytes + messages * (double) GRPC_PREFIX;
        double segments = Math.ceil(payload / TCP_MSS);
        return (payload + segments * TCP_HEADER) / (1.0 - e.rho()) + 3.0 * TCP_HEADER;
    }

    /** Least source+repair packet count whose eta-quantile of deliveries reaches K + overhead. */
    public static long raptorqPackets(EnvironmentProfile e, long sourcePackets, double eta) {
        if (e.rho() == 0.0) {
            return sourcePackets;
        }
        double lambda = e.meanBurst() > 1.0 ? e.gilbertElliott().lambda() : 0.0;
        long need = sourcePackets + RAPTORQ_OVERHEAD;
        long hi = need;
        while (Probability.deliveredLowerBound(hi, e.rho(), lambda, eta) < need) {
            hi *= 2;
        }
        long lo = need;
        while (lo < hi) {
            long mid = (lo + hi) >>> 1;
            if (Probability.deliveredLowerBound(mid, e.rho(), lambda, eta) >= need) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        return lo;
    }

    /** A hop where the busiest sender emits {@code senderBytes} and the receiver takes {@code ingressBytes}. */
    public Hop hop(EnvironmentProfile e, Configuration c, double senderBytes, double ingressBytes) {
        double bw = e.bandwidthBytesPerSecond();
        double rtt = e.rttSeconds();
        if (c.transport().datagram()) {
            double t = rtt / 2.0 + Math.max(senderBytes, ingressBytes) / bw;
            if (c.transport() == Configuration.Transport.EARLY_CLOSE) {
                t += rtt;
            }
            return new Hop(senderBytes, ingressBytes, t);
        }
        double p = e.rho() / e.meanBurst();
        double d = Math.ceil(senderBytes / (TCP_MSS + TCP_HEADER));
        double dss = p == 0.0 ? d : Math.min(d, (1.0 - Math.pow(1.0 - p, d)) * (1.0 - p) / p + 1.0);
        double ss = rtt * Math.log(dss * (SLOW_START_GROWTH - 1.0) / INITIAL_WINDOW + 1.0)
                / Math.log(SLOW_START_GROWTH);
        double rest = 0.0;
        if (d > dss) {
            double flow = rtt > 0.0 ? Math.min(bw, TCP_MSS / rtt * Math.sqrt(1.5 / p)) : bw;
            rest = rtt + (d - dss) * (TCP_MSS + TCP_HEADER) / flow; // one fast-recovery round, then congestion avoidance
        }
        double t = rtt + Math.max(ingressBytes / bw, Math.max(senderBytes / bw, ss + rest));
        return new Hop(senderBytes, ingressBytes, t);
    }

    /** Bytes of one input row in transit (for data placement under data-level redundancy). */
    public double rowBytes(WorkloadProfile w) {
        double nnz = Math.max(1.0, w.density() * w.d());
        return Math.min(8.0 * w.d(), 12.0 * nnz + 4.0) + 8.0;
    }

    static boolean merged(Packetization p) {
        return p == Packetization.MERGED || p == Packetization.SPLIT;
    }
}
