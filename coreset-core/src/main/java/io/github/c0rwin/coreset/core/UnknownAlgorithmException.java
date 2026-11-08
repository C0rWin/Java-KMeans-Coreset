package io.github.c0rwin.coreset.core;

import java.util.Collections;
import java.util.SortedSet;
import java.util.TreeSet;

/** Thrown by {@link CoresetAlgorithms#create} for an id no provider owns; the message lists the available ids. */
public class UnknownAlgorithmException extends IllegalArgumentException {

    private static final long serialVersionUID = 1L;

    private final String algorithmId;
    private final TreeSet<String> availableIds;

    public UnknownAlgorithmException(String algorithmId, SortedSet<String> availableIds) {
        super("unknown coreset algorithm '" + algorithmId + "', available: " + availableIds);
        this.algorithmId = algorithmId;
        this.availableIds = new TreeSet<>(availableIds);
    }

    public String algorithmId() {
        return algorithmId;
    }

    public SortedSet<String> availableIds() {
        return Collections.unmodifiableSortedSet(availableIds);
    }
}
