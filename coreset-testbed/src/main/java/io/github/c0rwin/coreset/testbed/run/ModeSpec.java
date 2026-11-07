package io.github.c0rwin.coreset.testbed.run;

import io.github.c0rwin.coreset.node.TreeMode;
import java.util.Locale;
import java.util.Objects;

/**
 * How a coreset is built from the dataset.
 *
 * <ul>
 *   <li>{@code offline}: one {@code compress} of the whole input.</li>
 *   <li>{@code stream}: the input in {@code blockCount} consecutive blocks, each compressed and added to a
 *       {@code MergeReduceTree}; distortion is measured on the summary of the first {@code 2^l} blocks for every
 *       level {@code l}.</li>
 *   <li>{@code distributed}: the blocks dealt round-robin to {@code nodes} in-process {@code CoresetNode}s, either
 *       called directly ({@code transport: node}) or over gRPC on loopback ({@code transport: grpc}); the node
 *       snapshots are unioned.</li>
 * </ul>
 *
 * @param finalReduce stream/distributed: compress the final union once more to the target size (default true)
 */
public record ModeSpec(String type, Integer blockCount, Boolean finalReduce, Integer nodes, String transport,
                       Integer workers, String treeMode, String label) {

    public static final String OFFLINE = "offline";
    public static final String STREAM = "stream";
    public static final String DISTRIBUTED = "distributed";

    public ModeSpec {
        Objects.requireNonNull(type, "mode type");
        type = type.toLowerCase(Locale.ROOT);
        switch (type) {
            case OFFLINE -> {
                if (blockCount != null || nodes != null || transport != null || workers != null || treeMode != null
                        || finalReduce != null) {
                    throw new IllegalArgumentException("offline mode takes no stream or distributed settings");
                }
            }
            case STREAM -> {
                blockCount = blockCount == null ? 64 : blockCount;
                finalReduce = finalReduce == null || finalReduce;
                if (nodes != null || transport != null || workers != null || treeMode != null) {
                    throw new IllegalArgumentException("stream mode takes blockCount and finalReduce only");
                }
            }
            case DISTRIBUTED -> {
                blockCount = blockCount == null ? 64 : blockCount;
                finalReduce = finalReduce == null || finalReduce;
                nodes = nodes == null ? 4 : nodes;
                transport = transport == null ? "node" : transport.toLowerCase(Locale.ROOT);
                workers = workers == null ? 1 : workers;
                treeMode = treeMode == null ? TreeMode.SINGLE_WRITER.id() : TreeMode.fromId(treeMode).id();
                if (!transport.equals("node") && !transport.equals("grpc")) {
                    throw new IllegalArgumentException("transport must be node or grpc: " + transport);
                }
                if (nodes < 1 || workers < 1) {
                    throw new IllegalArgumentException("nodes and workers must be >= 1");
                }
            }
            default -> throw new IllegalArgumentException("unknown mode '" + type + "', expected offline, stream or "
                    + "distributed");
        }
        if (blockCount != null && blockCount < 1) {
            throw new IllegalArgumentException("blockCount must be >= 1: " + blockCount);
        }
        if (label == null || label.isBlank()) {
            label = switch (type) {
                case STREAM -> "stream-b" + blockCount;
                case DISTRIBUTED -> "distributed-" + transport + "-n" + nodes + "-b" + blockCount;
                default -> OFFLINE;
            };
        }
    }

    public static ModeSpec offline() {
        return new ModeSpec(OFFLINE, null, null, null, null, null, null, null);
    }

    public static ModeSpec stream(int blockCount) {
        return new ModeSpec(STREAM, blockCount, null, null, null, null, null, null);
    }

    public static ModeSpec distributed(int nodes, String transport, int blockCount) {
        return new ModeSpec(DISTRIBUTED, blockCount, null, nodes, transport, null, null, null);
    }
}
