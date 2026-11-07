package io.github.c0rwin.coreset.node.testing;

import io.github.c0rwin.coreset.core.Coreset;
import io.github.c0rwin.coreset.core.CoresetAlgorithm;
import io.github.c0rwin.coreset.core.SparseVector;
import io.github.c0rwin.coreset.core.WeightedPoint;
import io.github.c0rwin.coreset.core.metrics.MetricsSink;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.random.RandomGenerator;

/** Test doubles for {@link CoresetAlgorithm}; not coreset constructions, only predictable behaviours. */
public final class TestAlgorithms {

    private TestAlgorithms() {
    }

    /** Returns its input: exact weight preservation, unbounded size. */
    public static final CoresetAlgorithm IDENTITY = (input, rng, metrics) -> input;

    /** Sums the weights of equal vectors (first-seen order): weight-preserving and bounded by distinct vectors. */
    public static final CoresetAlgorithm MERGE_EQUAL = (input, rng, metrics) -> {
        Map<SparseVector, Double> acc = new LinkedHashMap<>();
        for (WeightedPoint p : input.points()) {
            acc.merge(p.vector(), p.weight(), Double::sum);
        }
        List<WeightedPoint> out = new ArrayList<>(acc.size());
        acc.forEach((v, wt) -> out.add(new WeightedPoint(v, wt)));
        return new Coreset(out, input.offset());
    };

    /** Draws {@code m} points with replacement using {@code rng}; output depends on every random draw. */
    public static CoresetAlgorithm sample(int m) {
        return (input, rng, metrics) -> {
            if (input.size() <= m) {
                return input;
            }
            double total = input.totalWeight();
            List<WeightedPoint> out = new ArrayList<>(m);
            for (int i = 0; i < m; i++) {
                WeightedPoint p = input.points().get(rng.nextInt(input.size()));
                out.add(new WeightedPoint(p.vector(), total / m));
            }
            return new Coreset(out, input.offset());
        };
    }

    /** Throws when the input contains a point of weight {@code poison}. */
    public static CoresetAlgorithm failingOn(double poison) {
        return (input, rng, metrics) -> {
            for (WeightedPoint p : input.points()) {
                if (p.weight() == poison) {
                    throw new IllegalStateException("poisoned batch (weight " + poison + ")");
                }
            }
            return input;
        };
    }

    /**
     * Wraps a delegate so every call first signals {@link #entered} and then waits for {@link #release()}. The wait
     * is bounded so a broken test fails instead of hanging.
     */
    public static final class Gated implements CoresetAlgorithm {
        private final CoresetAlgorithm delegate;
        private final CountDownLatch entered = new CountDownLatch(1);
        private final CountDownLatch released = new CountDownLatch(1);

        public Gated(CoresetAlgorithm delegate) {
            this.delegate = delegate;
        }

        @Override
        public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
            entered.countDown();
            try {
                if (!released.await(30, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("gate never released");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted at gate", e);
            }
            return delegate.compress(input, rng, metrics);
        }

        /** Blocks until a worker is inside {@link #compress}. */
        public void awaitEntered() throws InterruptedException {
            if (!entered.await(30, TimeUnit.SECONDS)) {
                throw new IllegalStateException("no worker reached the gate");
            }
        }

        public void release() {
            released.countDown();
        }
    }

    /** Records every (input, output) pair, in call order. */
    public static final class Recording implements CoresetAlgorithm {
        public record Call(Coreset input, Coreset output) {
        }

        private final CoresetAlgorithm delegate;
        private final List<Call> calls = new CopyOnWriteArrayList<>();

        public Recording(CoresetAlgorithm delegate) {
            this.delegate = delegate;
        }

        @Override
        public Coreset compress(Coreset input, RandomGenerator rng, MetricsSink metrics) {
            Coreset out = delegate.compress(input, rng, metrics);
            calls.add(new Call(input, out));
            return out;
        }

        public List<Call> calls() {
            return List.copyOf(calls);
        }

        /** Output of the unique call whose input points equal {@code points}. */
        public Coreset outputFor(List<WeightedPoint> points) {
            List<Coreset> found = calls.stream().filter(c -> c.input().points().equals(points)).map(Call::output)
                    .toList();
            if (found.size() != 1) {
                throw new AssertionError("expected one call with that input, found " + found.size());
            }
            return found.get(0);
        }
    }
}
