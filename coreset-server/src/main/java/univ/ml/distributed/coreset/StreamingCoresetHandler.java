package univ.ml.distributed.coreset;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

import org.apache.thrift.TException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.collect.Lists;

import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;

public class StreamingCoresetHandler implements CoresetService.Iface {

    private final static Logger log = LoggerFactory.getLogger(StreamingCoresetHandler.class);

    private Map<Integer, List<SparseWeightableVector>> coresetTree = new HashMap<>();

    private ConcurrentLinkedQueue<List<SparseWeightableVector>> samples = new ConcurrentLinkedQueue<>();

    private ConcurrentLinkedQueue<CoresetPoints> pointsQueue = new ConcurrentLinkedQueue<>();

    private Executor pool = Executors.newFixedThreadPool(1);

    private Executor compressPool = Executors.newFixedThreadPool(20);

    private AtomicLong counter = new AtomicLong();

    private ReentrantLock lock = new ReentrantLock();

    private SparseCoresetAlgorithm algorithm;

    public StreamingCoresetHandler() {
        log.info("Creating new instance of [StreamingCoresetHandler].");
        pool.execute(() -> {
            log.info("Submitting thread to pool for new coreset nodes and merge them into the tree.");
            while (true) {
                List<SparseWeightableVector> coreset = samples.poll();
                if (coreset == null) {
                    try {
                        TimeUnit.MILLISECONDS.sleep(200);
                        continue;
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
                try {
                    int treeLevel = 0;
                    lock.lock();
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
                    counter.decrementAndGet();
                } finally {
                    lock.unlock();
                }
            }
        });

        for (int i = 0; i < 20; i++) {
            compressPool.execute(() -> {
                while (true) {
                    final CoresetPoints points = pointsQueue.poll();
                    if (points == null) {
                        try {
                            TimeUnit.MILLISECONDS.sleep(200);
                            continue;
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }

                    final List<SparseWeightableVector> sample = algorithm.takeSample(
                            CoresetUtils.toSparseWeightableVectors(points));
                    samples.add(sample);
                }
            });
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
        counter.incrementAndGet();
        pointsQueue.add(coresetPoints);
    }

    @Override
    public CoresetPoints getTotalCoreset() throws TException {
        try {
            lock.lock();
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
            lock.lock();
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
}
