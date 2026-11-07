package univ.ml.distributed.coreset;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

import org.apache.thrift.TApplicationException;
import org.apache.thrift.TException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;
import com.google.common.util.concurrent.ThreadFactoryBuilder;

import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;

public class StreamingCoresetHandler implements CoresetService.Iface, AutoCloseable {

    private final static Logger log = LoggerFactory.getLogger(StreamingCoresetHandler.class);

    private Map<Integer, List<SparseWeightableVector>> coresetTree = new HashMap<>();

    // Daemon threads, so a handler that is never closed does not keep the JVM alive.
    private final ExecutorService mergePool = Executors.newSingleThreadExecutor(
            new ThreadFactoryBuilder().setDaemon(true).setNameFormat("coreset-merge-%d").build());

    private final ExecutorService compressPool = Executors.newFixedThreadPool(20,
            new ThreadFactoryBuilder().setDaemon(true).setNameFormat("coreset-compress-%d").build());

    private AtomicLong counter = new AtomicLong();

    private final AtomicReference<Throwable> failure = new AtomicReference<>();

    private ReentrantLock lock = new ReentrantLock();

    // Set by initialize() on a server thread, read by the worker threads.
    private volatile SparseCoresetAlgorithm algorithm;

    public StreamingCoresetHandler() {
        log.info("Creating new instance of [StreamingCoresetHandler].");
    }

    private void merge(List<SparseWeightableVector> coreset) {
        lock.lock();
        try {
            int treeLevel = 0;
            List<SparseWeightableVector> leaf = coresetTree.get(treeLevel);

            while (null != leaf) {
                coresetTree.remove(treeLevel++);
                coreset.addAll(leaf);
                double weightBefore = 0d;
                if (log.isDebugEnabled()) {
                    weightBefore = coreset.stream().map(x -> x.getWeight()).collect(Collectors.summingDouble(x -> x));
                }
                coreset = algorithm.takeSample(coreset);
                if (log.isDebugEnabled()) {
                    double weightAfter = coreset.stream().map(x -> x.getWeight()).collect(Collectors.summingDouble(x -> x));
                    log.debug("Total weight of two nodes before merging = {}, after merging = {}, level = {}, next level = {}",
                            weightBefore,
                            weightAfter,
                            treeLevel - 1,
                            treeLevel);
                }

                leaf = coresetTree.get(treeLevel);
            }

            coresetTree.put(treeLevel, coreset);
        } finally {
            lock.unlock();
        }
    }

    @Override
    public boolean initialize(int k, int sampleSize, final CoresetAlgorithm coresetAlgorithm) throws TException {
        log.info("Initializing streaming coreset algorithm, " +
                "for k = {}, sample size = {}, algorithm = {}.", k, sampleSize, coresetAlgorithm);

        switch (coresetAlgorithm) {
            case UNIFORM: {
                log.debug("Initializing steaming algorithm with uniform algorithm.");
                algorithm = CoresetAlgorithmFactory.createUniformAlgorithm(sampleSize);
                break;
            }
            case NON_UNIFORM: {
                log.debug("Initializing streaming coreset with Non-Uniform algorithm.");
                algorithm = CoresetAlgorithmFactory.createNonUniformAlgorithm(k, sampleSize);
                break;
            }
            case KMEANS_PLUS_PLUS: {
                throw new UnsupportedOperationException();
            }
        }
        return true;
    }

    @Override
    public void compressPoints(CoresetPoints coresetPoints) throws TException {
        requireInitialized();
        counter.incrementAndGet();
        // Batches compress in parallel; a single merge thread keeps tree updates sequential.
        CompletableFuture
                .supplyAsync(() -> algorithm.takeSample(CoresetUtils.toSparseWeightableVectors(coresetPoints)), compressPool)
                .thenAcceptAsync(this::merge, mergePool)
                .whenComplete((ignored, error) -> {
                    if (error != null) {
                        final Throwable cause = error instanceof CompletionException ? error.getCause() : error;
                        log.error("Failed to compress points id = {}.", coresetPoints.getId(), cause);
                        failure.compareAndSet(null, cause);
                    }
                    counter.decrementAndGet();
                });
    }

    @Override
    public CoresetPoints getTotalCoreset() throws TException {
        requireInitialized();
        final Throwable error = failure.get();
        if (error != null) {
            throw new TApplicationException(TApplicationException.INTERNAL_ERROR,
                    "A batch of points failed to compress, the coreset would be incomplete: " + error);
        }
        lock.lock();
        try {
            if (coresetTree.size() == 1) {
                log.debug("There is only one node in the tree, returning as a final coreset.");
                for (List<SparseWeightableVector> coreset : coresetTree.values()) {
                    return CoresetUtils.toCoresetPoints(coreset);
                }
            }
            final List<SparseWeightableVector> treeCoresetView = Lists.newArrayList();
            for (Map.Entry<Integer, List<SparseWeightableVector>> each : coresetTree.entrySet()) {
                log.debug("Adding coreset of size = {} from level = {}, weight = {}",
                        each.getValue().size(),
                        each.getKey(),
                        each.getValue().stream()
                                .map(x -> x.getWeight()).collect(Collectors.summingDouble(x -> x)));
                // Merging level points
                treeCoresetView.addAll(each.getValue());
            }
            double weightBefore = treeCoresetView
                    .stream()
                    .map(x -> x.getWeight())
                    .collect(Collectors.summingDouble(x -> x));

            final CoresetPoints totalCoreset = CoresetUtils.toCoresetPoints(algorithm.takeSample(treeCoresetView));
            if (log.isDebugEnabled()) {
                log.debug("***** >>> Compressing all tree nodes {} into the final coreset of size = {}, " +
                                "total weight before = {}, after = {}",
                        treeCoresetView.size(),
                        totalCoreset.getPoints().size(),
                        weightBefore,
                        totalCoreset.getPoints().stream().map(x -> x.getWeight()).collect(Collectors.summingDouble(x -> x)));
            }
            return totalCoreset;
        } finally {
            lock.unlock();
        }
    }

    @Override
    public double getEnergy(CoresetPoints centers, CoresetPoints points) throws TException {
        double energy = CoresetUtils.getEnergy(centers, points);
/*
        log.debug("Computed energy for chunk = {}  is [{}], centers and {} points to compute energy, points chunk id = {}",
                points.getId(),
                energy,
                centers.getPoints().size(),
                points.getPoints().size());
*/
        return energy;
    }

    @Override
    public boolean isDone() throws TException {
        return counter.get() == 0;
    }

    /**
     * Stops the worker threads; the handler cannot be used afterwards.
     */
    @Override
    public void close() {
        compressPool.shutdownNow();
        mergePool.shutdownNow();
    }

    // Unlike other exceptions, a TApplicationException reaches the Thrift client with its message.
    private void requireInitialized() throws TApplicationException {
        if (algorithm == null) {
            throw new TApplicationException(TApplicationException.INTERNAL_ERROR,
                    "Coreset service is not initialized, call initialize() first.");
        }
    }
}
