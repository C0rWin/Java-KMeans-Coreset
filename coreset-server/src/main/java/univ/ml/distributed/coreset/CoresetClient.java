package univ.ml.distributed.coreset;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.thrift.TException;
import org.apache.thrift.protocol.TBinaryProtocol;
import org.apache.thrift.transport.TFramedTransport;
import org.apache.thrift.transport.TSocket;
import org.apache.thrift.transport.TTransport;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import univ.ml.sparse.SparseCentroidCluster;
import univ.ml.sparse.algorithm.SparseWeightedKMeansPlusPlus;

public class CoresetClient {

    private static final Logger log = LoggerFactory.getLogger(CoresetClient.class);

    /**
     * Parameter for the input files path
     */
    public static final String FILE_ARG = "file";

    /**
     * Parameter which keep list of hosts to connect to, e.g. "hosts=server1:port1,server2:port2,..."
     */
    public static final String HOSTS_ARG = "hosts";

    /**
     * Coreset algorithm name to use
     */
    public static final String ALGORITHM_ARG = "algorithm";

    /**
     * k parameter of k-means algorithm
     */
    public static final String K_ARG = "k";

    /**
     * Coreset sample size
     */
    public static final String SAMPLE_SIZE_ARG = "sampleSize";

    /**
     * Points batch size, amount of points to sent for compression
     */
    public static final String BATCH_SIZE_ARG = "batchSize";

    private final Integer kValue;

    private final CoresetAlgorithm coresetAlgName;

    private final Integer sampleSize;

    private final Integer batchSize;

    private List<ClientMetadata> clientsMeta = new ArrayList<>();

    private Map<ClientMetadata, CoresetService.Client> clients = new HashMap<>();

    /**
     * Default random points provider.
     */
    private CoresetPointsProvider provider = new RandomPointsProvider(100_000, 3);

    /**
     * @param config
     */
    public CoresetClient(final ClientConfig config) {
        kValue = config.k();
        coresetAlgName = config.coresetAlgName();

        sampleSize = config.sampleSize();
        batchSize = config.batchSize();

        log.info("Client created with init parameters: k = {}, coreset = {}, sample size = {}, batch = {}",
                new Object[]{kValue, coresetAlgName, sampleSize, batchSize});
        final String serverHosts = config.hosts();
        log.info("Coreset servers = {}", serverHosts);
        for (final String endpoint : serverHosts.split(",")) {
            final String hostName = endpoint.split(":")[0];
            final int portNum = Integer.valueOf(endpoint.split(":")[1]);
            clientsMeta.add(new ClientMetadata(hostName, portNum));
        }
    }

    public CoresetClient(final ClientConfig config, final CoresetPointsProvider provider) {
        this(config);
        this.provider = provider;
    }

    /**
     * @return
     */
    public boolean initialize() {
        for (final ClientMetadata metadata : clientsMeta) {
            try {
                log.info("Opening server socket on port = {}, host name = {}", metadata.getPortNum(), metadata.getHostName());
                TTransport socket = new TFramedTransport(new TSocket(metadata.getHostName(), metadata.getPortNum()));

                final TBinaryProtocol protocol = new TBinaryProtocol(socket);
                final CoresetService.Client client = new CoresetService.Client(protocol);

                socket.open();
                log.info("Initializing coreset service with parameters, " +
                        "k = {}, sample size = {} and algorithm = {}", kValue, sampleSize, coresetAlgName);
                if (!client.initialize(kValue, sampleSize, coresetAlgName)) {
                    log.error("Server wasn't able to accept initialization parameters, please read server logs.");
                    return false;
                }
                metadata.setSocket(socket);
                clients.put(metadata, client);
            } catch (Exception e) {
                e.printStackTrace();
                return false;
            }
        }
        return true;
    }

    /**
     * Sends all points to remote servers using round-robin algorithm.
     */
    public void distributeAllPoints() {
        log.info("Starting to distribute points to remote servers.");
        // Continue until there are points available
        provider.reset();
        long id = 1;
        double totalWeight = 0d;
        while (true) {
            for (Map.Entry<ClientMetadata, CoresetService.Client> each : clients.entrySet()) {
                try {
                    // If not more points to send provider expected to return an empty list
                    final List<CoresetWeightedPoint> nextPoints = provider.nextBatchOfCoresetPoints(batchSize);
                    if (nextPoints.isEmpty()) {
                        log.info("There are no more points to distribute, finishing...");
                        log.debug("#####: Total weight of the points streamed is = {}.", totalWeight);
                        return;
                    }

                    totalWeight += nextPoints.stream().map(x->x.getWeight()).collect(Collectors.summingDouble(x->x));
                    // Wrap point into Thrift container
                    each.getValue().compressPoints(new CoresetPoints(id++, nextPoints));
                } catch (Exception e) {
                    log.error("Cannot sent next batch of points due to ", e);
                }
            }
        }
    }

    public boolean isAllDone() {
        int doneCnt = 0;
        for (Map.Entry<ClientMetadata, CoresetService.Client> each : clients.entrySet()) {
            try {
                if (each.getValue().isDone()) {
                    doneCnt++;
                }
            } catch (TException e) {
                e.printStackTrace();
            }
        }
        return doneCnt == clients.size();
    }

    /**
     * @return
     */
    public double computeTotalEnergy() {
        //TODO: Could be pretty important issue. Need to make sure all servers has finished coreset computation and
        // there are not points compression currently executed in the process. Otherwise it could point to very critical
        // bug.
        final List<CoresetWeightedPoint> totalCoreset = new ArrayList<>();
        double totalWeight = 0d;
        for (Map.Entry<ClientMetadata, CoresetService.Client> each : clients.entrySet()) {
            try {
                final CoresetPoints coresetPoints = each.getValue().getTotalCoreset();
                if (log.isDebugEnabled()) {
                    totalWeight += coresetPoints
                            .getPoints()
                            .stream()
                            .map(CoresetWeightedPoint::getWeight)
                            .collect(Collectors.summingDouble(x -> x));
                }
                totalCoreset.addAll(coresetPoints.getPoints());
            } catch (Exception e) {
                log.error("Unable to retreive final coreset from server host name = {}, port num = {}",
                        each.getKey().getHostName(), each.getKey().getPortNum());
            }
        }

        log.info("Total coreset weight {}", totalWeight);

        final SparseWeightedKMeansPlusPlus kMeansPlusPlus = new SparseWeightedKMeansPlusPlus(kValue);
        log.debug("Clustering total coreset points to get approximated k centers.");

        final List<SparseCentroidCluster> clusters = kMeansPlusPlus.cluster(
                CoresetUtils.toSparseWeightableVectors(totalCoreset));

        final List<CoresetWeightedPoint> centers = clusters.stream().map(x -> {
            Map<Integer, Double> coords = new HashMap();
            int dim = x.getCenter().getVector().getDimension();
            for (int i = 0; i < dim; i++) {
                coords.put(i, x.getCenter().getVector().getEntry(i));
            }

            return new CoresetWeightedPoint(new CoresetPoint(coords, dim), 1d);
        }).collect(Collectors.toList());

        provider.reset();

        double totalEnergy = 0d;
        // Need to use available servers for help to compute total energy
        long id = 1;
        while (true) {
            for (Map.Entry<ClientMetadata, CoresetService.Client> each : clients.entrySet()) {
                try {

                    final List<CoresetWeightedPoint> points = provider.nextBatchOfCoresetPoints(batchSize);
                    if (points.isEmpty()) {
                        return totalEnergy;
                    }

                    totalEnergy += each.getValue().getEnergy(new CoresetPoints(0, centers), new CoresetPoints(id++, points));
                } catch (Exception e) {
                    log.error("Server host name = {}, port num = {}, is not able to compute energy.",
                            each.getKey().getHostName(), each.getKey().getPortNum());
                    log.error(e.getMessage(), e);
                }
            }
        }
    }

    public void closeAll() {
        for (ClientMetadata metadata : clients.keySet()) {
            metadata.closeSocket();
        }
    }

    /**
     * @param args
     */
    public static void main(String[] args) {
        /** First of all let's try to read parameters from command line
         and parse them into {@link CommandLine} instance object.
         */
        CommandLine cli = null;
        CoresetClient client = null;
        try {
            cli = readComandLineArgs(args);

            final ClientConfig clientConfig = ClientConfig.init()
                    .k(Integer.valueOf(cli.getOptionValue(K_ARG)))
                    .coresetAlgName(CoresetAlgorithm.valueOf(cli.getOptionValue(ALGORITHM_ARG, "NON_UNIFORM")))
                    .sampleSize(Integer.valueOf(cli.getOptionValue(SAMPLE_SIZE_ARG, "100")))
                    .batchSize(Integer.valueOf(cli.getOptionValue(BATCH_SIZE_ARG, "1000")))
                    .hosts(cli.getOptionValue(HOSTS_ARG, "localhost:9191"))
                    .create();

            client = new CoresetClient(clientConfig);
        } catch (Exception e) {
            final HelpFormatter help = new HelpFormatter();
            // Once we've got parsing exception, let's print usage help and
            // finish the execution.
            help.printHelp("CoresetClient", getCliOptions());
            System.exit(-1);
        }

        try {
            // Initialize and open clienting connections to all
            // servers provided as command line argument.
            client.initialize();

            // Until there are points to send, distributed them all
            // to available servers.
            client.distributeAllPoints();

            // Wait until all servers will finish computation
            while (!client.isAllDone()) {
                TimeUnit.MILLISECONDS.sleep(100);
            }

            // Once points distributed to all servers need to ask
            // from each server a coreset view it has and use it
            // to calculate over all clusters energy.
            double energy = client.computeTotalEnergy();
            log.info("Total coreset energy is: {}.", energy);
        } catch (Exception e ) {
            log.error(e.getMessage(), e);
        } finally {
            log.debug("Closing all client sockets and exiting...");
            client.closeAll();
        }
    }

    private static CommandLine readComandLineArgs(String[] args) throws ParseException {
        final Options options = getCliOptions();

        final DefaultParser parser = new DefaultParser();
        return parser.parse(options, args);
    }

    private static Options getCliOptions() {
        final Option file = Option.builder()
                .argName(FILE_ARG)
                .longOpt(FILE_ARG)
                .hasArg(true)
                .valueSeparator()
                .desc("Input csv file with dataset points.")
                .build();

        final Option hosts = Option.builder()
                .argName(HOSTS_ARG)
                .longOpt(HOSTS_ARG)
                .hasArgs()
                .valueSeparator()
                .desc("Coresets servers to handle remote computation.")
                .build();

        final Option algorithm = Option.builder()
                .argName(ALGORITHM_ARG)
                .longOpt(ALGORITHM_ARG)
                .hasArg(true)
                .desc("Coreset algorithm to use for remote server initialization.")
                .build();

        final Option k = Option.builder()
                .argName(K_ARG)
                .longOpt(K_ARG)
                .hasArg(true)
                .valueSeparator()
                .desc("Value for <k> parameter of k-means algorithm.")
                .build();

        final Option sampleSize = Option.builder()
                .argName(SAMPLE_SIZE_ARG)
                .longOpt(SAMPLE_SIZE_ARG)
                .hasArg(true)
                .valueSeparator()
                .desc("The coreset size to use while compressing the data.")
                .build();

        final Option batchSize = Option.builder()
                .argName(BATCH_SIZE_ARG)
                .longOpt(BATCH_SIZE_ARG)
                .hasArg(true)
                .valueSeparator()
                .desc("Data batch size to send to the servers.")
                .build();

        final Options options = new Options();

        options.addOption(file);
        options.addOption(hosts);
        options.addOption(algorithm);
        options.addOption(k);
        options.addOption(sampleSize);
        options.addOption(batchSize);
        return options;
    }

    private static CoresetPoints generateRandomPoints(int N, int dim) {
        final Random rnd = new Random();
        final CoresetPoints result = new CoresetPoints();
        for (int i = 0; i < N; i++) {
            Map<Integer, Double> coords = new HashMap<>();
            for (int j = 0; j < dim; j++) {
                coords.put(j, rnd.nextDouble());
            }
            result.addToPoints(new CoresetWeightedPoint(new CoresetPoint(coords, dim), 1d));
        }
        return result;
    }
}
