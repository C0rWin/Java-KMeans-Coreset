package univ.ml.distributed.coreset;

/**
 * Builder component for coreset client, keeps parameters
 * to initiate remote coreset algorithm.
 */
public class ClientConfig {

    public static class ClientConfigBuilder {

        private ClientConfig config;

        private ClientConfigBuilder() {
            config = new ClientConfig();
        }

        public ClientConfigBuilder coresetAlgName(final CoresetAlgorithm coresetAlgName) {
            config.coresetAlgName(coresetAlgName);
            return this;
        }

        public ClientConfigBuilder k(final int kValue) {
            config.k(kValue);
            return this;
        }

        public ClientConfigBuilder sampleSize(final int sampleSize) {
            config.sampleSize(sampleSize);
            return this;
        }

        public ClientConfigBuilder batchSize(final int batchSize) {
            config.batchSize(batchSize);
            return this;
        }

        public ClientConfigBuilder hosts(String hosts) {
            config.hosts(hosts);
            return this;
        }

        public ClientConfig create() {
            return config;
        }
    }

    private int k;

    private CoresetAlgorithm coresetAlgName;

    private Integer sampleSize;

    private Integer batchSize;

    private String hosts;

    private ClientConfig() {

    }

    public static ClientConfigBuilder init() {
        return new ClientConfigBuilder();
    }

    public ClientConfig k(int k) {
        this.k = k;
        return this;
    }

    public int k() {
        return k;
    }

    public ClientConfig coresetAlgName(final CoresetAlgorithm algName) {
        this.coresetAlgName = algName;
        return this;
    }

    public CoresetAlgorithm coresetAlgName() {
        return coresetAlgName;
    }

    public ClientConfig sampleSize(final int sampleSize) {
        this.sampleSize = sampleSize;
        return this;
    }

    public int sampleSize() {
        return sampleSize;
    }

    public ClientConfig batchSize(final int batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    public int batchSize() {
        return batchSize;
    }

    public ClientConfig hosts(String hosts) {
        this.hosts = hosts;
        return this;
    }

    public String hosts() {
        return hosts;
    }


}
