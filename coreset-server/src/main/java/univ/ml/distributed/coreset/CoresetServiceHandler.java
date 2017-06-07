package univ.ml.distributed.coreset;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import org.apache.thrift.TException;
import org.apache.thrift.server.TNonblockingServer;
import org.apache.thrift.server.TServer;
import org.apache.thrift.transport.TNonblockingServerSocket;
import org.apache.thrift.transport.TNonblockingServerTransport;
import org.apache.thrift.transport.TTransportException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import univ.ml.distributed.coreset.CoresetService.Processor;
import univ.ml.sparse.SparseWeightableVector;
import univ.ml.sparse.algorithm.SparseCoresetAlgorithm;
import univ.ml.sparse.algorithm.streaming.StreamingAlgorithm;

public class CoresetServiceHandler implements CoresetService.Iface {

    private static final Logger LOG = LoggerFactory.getLogger(CoresetServiceHandler.class);

    private final Lock lock = new ReentrantLock();

    private StreamingAlgorithm streaming;

    private AtomicLong counter = new AtomicLong(0);

    public CoresetServiceHandler() {
        LOG.info("Coreset service handler has been created and waiting for initialization.");
    }

    @Override
    public boolean initialize(int k, int sampleSize, CoresetAlgorithm coresetAlgorithm) throws TException {
        SparseCoresetAlgorithm algorithm = null;
        LOG.info("Initializing streaming coreset algorithm, " +
                "for k = {}, sample size = {}, algorithm = {}.", k, sampleSize, coresetAlgorithm);

        switch (coresetAlgorithm) {
            case UNIFORM: {
                LOG.debug("Initializing steaming algorithm with uniform algorithm.");
                algorithm = CoresetAlgorithmFactory.createUniformAlgorithm(sampleSize);
                break;
            }
            case NON_UNIFORM: {
                LOG.debug("Initializing streaming coreset with Non-Uniform algorithm.");
                algorithm = CoresetAlgorithmFactory.createNonUniformAlgorithm(k, sampleSize);
                break;
            }
            case KMEANS_PLUS_PLUS: {
                LOG.debug("Initializing streaming coreset with KMeans Coreset algorithm.");
                algorithm = CoresetAlgorithmFactory.createKmeansAlgorithm(sampleSize);
                break;
            }
        }
        if (algorithm != null) {
            streaming = new StreamingAlgorithm(algorithm);
            LOG.debug("Coreset algorithm has been initialized successfully.");
            return true;
        }
        LOG.warn("Wasn't able to initialize streaming algorithm.");
        return false;
    }

    /**
     * @param points
     * @throws TException
     */
    public void compressPoints(final CoresetPoints points) throws TException {
        LOG.info("Compressing points id = [{}], num of points [{}]", points.getId(), points.getPoints().size());
        counter.incrementAndGet();
        try {
            lock.lock();
            streaming.addDataset(CoresetUtils.toSparseWeightableVectors(points.getPoints()));
            counter.decrementAndGet();
        } finally {
            lock.unlock();
        }
    }

    /**
     * @return
     * @throws TException
     */
    public CoresetPoints getTotalCoreset() throws TException {
        // TODO: Need to make sure how I can actually get final and total coreset and
        // no compression currently executed (coreset streaming tree construction).
        List<SparseWeightableVector> totalCoreset = Collections.emptyList();
        LOG.debug("Computing final coreset.");
        try {
            lock.lock();
            totalCoreset = streaming.getTotalCoreset();
        } finally {
            lock.unlock();
        }
        if (totalCoreset.isEmpty()) {
            LOG.warn("Final coreset is empty.");
        } else {
            LOG.info("Final coreset consist of {} point.", totalCoreset.size());
        }

        LOG.debug("Sending back final coreset results.");
        return CoresetUtils.toCoresetPoints(totalCoreset);
    }

    @Override
    public double getEnergy(final CoresetPoints centers, final CoresetPoints points) throws TException {
        double energy = CoresetUtils.getEnergy(centers, points);
        LOG.debug("Computed energy for chunk = {}  is [{}], centers and {} points to compute energy, points chunk id = {}",
                points.getId(),
                energy,
                centers.getPoints().size(),
                points.getPoints().size());
        return energy;
    }

    @Override
    public boolean isDone() throws TException {
        return counter.get() == 0;
    }

    /**
     * @param args
     * @throws TTransportException
     */
    public static void main(String[] args) throws TTransportException {
        // Read socket number from command line argument
        final TNonblockingServerTransport socket = new TNonblockingServerSocket(Integer.valueOf(args[0]));

        final Processor<CoresetServiceHandler> serverProcessor = new Processor<>(new CoresetServiceHandler());
        final TServer server = new TNonblockingServer(new TNonblockingServer.Args(socket)
                .processor(serverProcessor));

        server.serve();
    }
}
