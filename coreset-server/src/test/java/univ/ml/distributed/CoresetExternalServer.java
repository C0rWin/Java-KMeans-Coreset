package univ.ml.distributed;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import org.apache.thrift.server.TNonblockingServer;
import org.apache.thrift.server.TServer;
import org.apache.thrift.transport.TNonblockingServerSocket;
import org.apache.thrift.transport.TNonblockingServerTransport;
import org.apache.thrift.transport.TTransportException;
import org.junit.rules.ExternalResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import univ.ml.distributed.coreset.CoresetService;
import univ.ml.distributed.coreset.StreamingCoresetHandler;

public class CoresetExternalServer extends ExternalResource {

    private Logger log = LoggerFactory.getLogger(CoresetExternalServer.class);

    private List<TServer> coresetServers = new ArrayList<>();

    private ExecutorService executors;

    private CountDownLatch latch;

    public static CoresetExternalServer create(final int... portNums) {
        try {
            return new CoresetExternalServer(portNums);
        } catch (TTransportException e) {
            throw new RuntimeException(e);
        }
    }

    private CoresetExternalServer(final int... portNums) throws TTransportException {
        // Read socket number from command line argument
        latch = new CountDownLatch(portNums.length);
        executors = Executors.newFixedThreadPool(portNums.length);
        for (int portNum : portNums) {
            log.info("Creating server with port number {}", portNum);
            final TNonblockingServerTransport socket = new TNonblockingServerSocket(portNum);

            final CoresetService.Processor<StreamingCoresetHandler> serverProcessor = new CoresetService.Processor<>(new StreamingCoresetHandler());
            coresetServers.add(new TNonblockingServer(new TNonblockingServer.Args(socket)
                    .processor(serverProcessor)));

        }

    }

    @Override
    protected void before() throws Throwable {

        for (TServer coresetServer : coresetServers) {
            executors.submit(() -> {
                try {
                    coresetServer.serve();
                    latch.await();
                    coresetServer.stop();
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            });
        }
    }

    @Override
    protected void after() {
        for (TServer coresetServer : coresetServers) {
            latch.countDown();
        } ;
    }
}
