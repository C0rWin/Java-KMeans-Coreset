k-Means for Streaming and Distributed Big Sparse Data
===

Introduction
---

In this repository we provide Matlab implementation of coreset algorithms, used
for evaluation in following [paper](https://arxiv.org/pdf/1511.08990.pdf):

> **k-Means for Streaming and Distributed Big Sparse Data.**
> *Artem Barger, and Dan Feldman.*
> Proceedings of the 2016 SIAM International Conference on Data
> Mining. Society for Industrial and Applied Mathematics, 2016.

Project Structure
---

Current project consist of three main modules:

1. **kmeans-coreset** - the Java based implementation of coresets algorithms:
    1.1. Uniform coreset
    1.2. Non Uniform coreset (Sensitivity)
    1.3. Deterministic coreset construction suggested in the paper above
2. **coreset-communication** - definition of the protocol for distributed
   computation of the coreset
3. **coreset-server** - implementation of server and client side of the
   distributed protocol which enables coreset computation.


APIs
---

1. ### Coreset algorithms, interfaces and definitions

   All algorithms works with `SparseWeightableVector` which is an
   implementation of weighted vector in `R^d`, extends and implements following
   interfaces:

   * SparseSample
   ```java
   public interface SparseSample extends SparseWeightable {

       void setProbability(final double prob);

       double getProbability();

   }
   ```

   * SparseWeightable
   ```java
   public interface SparseWeightable {

       void setWeight(final double weight);

       double  getWeight();
   }
   ```

   * SparseClusterable
   ```java
   public interface SparseClusterable {

       RealVector getVector();
   }
   ```

   **SparseCoresetAlgorithm** - is the interface which defines a main API for

   coreset algorithm, as follows:

   ```java
   public interface SparseCoresetAlgorithm extends Serializable {

       List<SparseWeightableVector> takeSample(final List<SparseWeightableVector>  pointset);

   }

   ```

   where `takeSample` is the method which actually represents coreset
   computation, it receives a list of weighted poinst and returns new weighted
   list of points reduced from the original list.

   We provide three different implementations of this interface:

   * Uniform coreset - `SparseUniformCoreset`

   ```java
    /**
    * @param sampleSize - coreset size to sample
    */
   public SparseUniformCoreset(final int t) {
       this.t = t;
   }
   ```

   Constructor which receives parameter of the coreset size to sample.

   * Non uniform coreset - `SparseNonUniformCoreset`

   ```java
    /**
    * @param seedingAlgorithm - algorithm used to seed and compute sensitivity
    * @param sampleSize - coreset size to sample
    */
   public SparseNonUniformCoreset(final SparseSeedingAlgorithm seedingAlgorithm, final int sampleSize) {
       this.sampleSize = sampleSize;
       // Default is the k-means++ algorithm which provides (1, log(n)) approximation for k-means problem.
           this.seedingAlgorithm = seedingAlgorithm;
   }
   ```

    Constructor which receives coreset size to sample and intial seeding
    algorithm (`SparseSeedingAlgorithm`) used for sensitivity computation.

    ```java
    public interface SparseSeedingAlgorithm {

        /**
         * Seed initial cluster centers.
         *
         * @param vectors vectors to select initial centers from.
         Method should not change input parameter.
         * @return list of the cluster centers.
         */
        List<SparseCentroidCluster> seed(final List<SparseWeightableVector> vectors);

    }

    ```

   * Our algorithm - `SparseKmeansCoresetAlgorithm`

    ```java
    /**
    * @param sampleSize - coreset size to sample
    */
    public SparseKmeansCoresetAlgorithm(int sampleSize) {
        this.sampleSize = sampleSize;
    }
    ```

    Constructor which receives as a parameter coreset size to compute ("sample").

2. ### Streaming and merge-and-reduce tree

   Following class defines the streaming algorithm - encapsulated
   abstraction of the *merge-and-reduce* tree which allows us to compute
   coreset in streaming seting.

   ```java
   public class StreamingAlgorithm {

       private Map<Integer, List<SparseWeightableVector>> coresetTree = Maps.newHashMap();

       private SparseCoresetAlgorithm coresetAlgorithm;

       public StreamingAlgorithm(SparseCoresetAlgorithm coresetAlgorithm)
       {
           this.coresetAlgorithm = coresetAlgorithm;
       }

       public void addDataset(final List<SparseWeightableVector> dataset) {
           List<SparseWeightableVector> coreset = coresetAlgorithm.takeSample(dataset);
           int treeLevel = 0;
           List<SparseWeightableVector> leaf = coresetTree.get(treeLevel);

           while (leaf != null) {
               coresetTree.remove(treeLevel++);
               coreset.addAll(leaf);
               coreset = coresetAlgorithm.takeSample(coreset);

               leaf = coresetTree.get(treeLevel);
           }

           coresetTree.put(treeLevel, coreset);
       }

       public List<SparseWeightableVector> getTotalCoreset() {
           if (coresetTree.size() == 1) {
               for (List<SparseWeightableVector> coreset : coresetTree.values()) {
                   return
                       coreset;
               }
           }
           final List<SparseWeightableVector> treeCoresetView = Lists.newArrayList();
           for (Map.Entry<Integer, List<SparseWeightableVector>> each : coresetTree.entrySet()) {
               treeCoresetView.addAll(each.getValue());
           }

           return coresetAlgorithm.takeSample(treeCoresetView);
       }

   }
   ```
    It accepts concrete implementation of the `SparseCoresetAlgorithm` interface and proceeds with
    coreset tree construction with streaming batch of points.

3. ### Distributed protocol definition

    Following defines Apache Thrift structure and services for distributed
    coreset computation:

    ```thrift
    namespace java univ.ml.distributed.coreset

    enum CoresetAlgorithm {
        UNIFORM,
        NON_UNIFORM,
        KMEANS_PLUS_PLUS
    }

    struct CoresetPoint {
        1: map<i32, double> coords,
        2: i32 dim
    }

    struct CoresetWeightedPoint {
        1: CoresetPoint point,
        2: double weight,
    }

    struct CoresetPoints {
        1: i64 id,
        2: list<CoresetWeightedPoint> points
        }

    service CoresetService {

        bool initialize(1:i32 k, 2:i32 sampleSize, 3:CoresetAlgorithm algorithm)

        oneway void compressPoints(1:CoresetPoints message)

        CoresetPoints getTotalCoreset()

        double getEnergy(1: CoresetPoints centers, 2: CoresetPoints points)

        bool isDone()
    }
    ```

4. ### Distributed protocol implementation:

   `CoresetServiceHandler` implements the logic of the `CoresetService` define
   with Apache Thrift above, it's capable to serve incomming batches of
   streamed data points and maintain coreset **merge-and-reduce** tree
   according to the initialization parameters.

   Protocol definition allows to initialize service handler with three
   different implementations of the coreset algorithm, the k of kmeans
   algorithm parameters and the actual coreset size.

   Once initialized it will allow to send weithed points from `R^d` to compress
   them into coreset and construct **merge-and-reduce** tree. After all it also
   could be used to compute energy function in the distributed manner,
   leveraging `getEnergy` method, which receives centers and points and
   computes sum of squared distances to the provided centers.


How to run
---

1. ### Building the project.

  In order to build the project please run:
  ```
  mvn clean install
  ```

  Compilation will produce two jar files in coreset-serer/target folder:

  1.1. **client.jar** - the client application to distribute points accross servers
  1.2. **server.jar** - the server application to receive batch of streamed
  points, compute and construct coreset tree.

2. ### Running server

   To execute server code you need to run:

   ```
   java -jar server.jar 9999
   ```

   where 9999 stands for TCP port server will bind to.

   Next to execute client you need to run:

   ```
   java -jar client.jar -k 32 -algorithm KMEANS_PLUS_PLUS \
   -sampleSize 512 -batchSize 1024 \
   -hosts serverIP1:9999,serverIP2:9999,...,serverIPN
   ```

   Where:

   * k - kmeans clustering parameter
   * algorithm - one of the available agorithms options: UNIFORM, NON_UNIFORM, KMEANS_PLUS_PLUS
   * sampleSize - the coreset size to sample
   * batchSize - number of points to send in each batch to the remote server
   * hosts - comma separated list of the remote hosts to distributed points to


Feedback
---

If you'd like to use this implementation, please reference original paper, any
feedback send to Artem Barger (artem@bargr.net)

License
---

The software is released under the MIT License as detailed in kmeans.pyx.

