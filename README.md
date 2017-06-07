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

Feedback
---

If you'd like to use this implementation, please reference original paper, any
feedback send to Artem Barger (artem@bargr.net)

License
---

The software is released under the MIT License as detailed in kmeans.pyx.

