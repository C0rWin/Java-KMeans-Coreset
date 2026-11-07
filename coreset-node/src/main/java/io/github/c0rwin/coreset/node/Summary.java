package io.github.c0rwin.coreset.node;

import io.github.c0rwin.coreset.core.Coreset;

/**
 * Immutable coreset tagged with what it summarises; the unit published by workers and stored in CAS slots.
 *
 * @param batches     ingested batches folded into {@code coreset}
 * @param inputWeight total input weight of those batches
 */
record Summary(Coreset coreset, long batches, double inputWeight) {

    static final Summary EMPTY = new Summary(Coreset.of(java.util.List.of()), 0, 0.0);
}
