# Distributed run: 8 node processes over gRPC

This run checks the whole distributed path on one machine with real processes: separate `coreset-node` JVMs, one
`coreset-coordinator` that streams the data to them over gRPC, gathers their coresets and solves k-means on the
union. Quality is measured on the full data, not on the coreset.

## Method

- **Data.** 1,000,000 points in d = 20 from 20 Gaussian clusters with strongly unbalanced sizes (25% down to 0.26%
  of the points), so a solver that misses small clusters pays for it.
- **Runs.** For N = 1, 2, 4, 8 nodes and four coreset configurations, the coordinator splits the input into batches
  of 10,000 points, distributes them over the nodes (2 workers each), gathers the node snapshots and runs
  k-means (k = 20, best of 10 restarts, 50 Lloyd iterations) on the union. Each configuration ran with seeds 3, 4
  and 5.
- **Quality.** The centers printed by the coordinator are evaluated on all 1M points (sum of squared distances to
  the nearest center) and divided by a full-data baseline: scikit-learn `KMeans(n_clusters=20, n_init=3,
  random_state=0)` on all points, inertia 7.997e7. The cost of the 20 true generating centers is 7.997e7 as well
  (79,972,433 against 79,970,674), so a ratio of 1.0 is the optimum for practical purposes. `ratio mean` and
  `ratio max` are over the three seeds.
- **Estimate error** is |coreset cost / full-data cost - 1| for the returned centers: how well the coreset
  predicts the cost it reports.
- **Communication** comes from each node's Prometheus counters (`grpc_server_Ingest_bytes_in_total`,
  `grpc_server_Snapshot_bytes_out_total`), summed over the N nodes.

Coreset configurations (per node, inside each node's merge-and-reduce tree):

| label | algorithm | parameters |
|---|---|---|
| uniform | `uniform` | m = 1000 |
| sensitivity α=2 | `sensitivity` | k = 20, m = 1000 (default α = 2) |
| sensitivity α=theory | `sensitivity` | k = 20, m = 1000, `alpha=kmeans++` (provable α) |
| mean m=200 | `mean` | m = 200, 10 Lloyd iterations |

## Setup

Apple M2 Max (12 cores), 96 GiB, Amazon Corretto 17.0.5, everything on 127.0.0.1. Nodes ran with `-Xmx2g`, the
coordinator with `-Xmx4g`. The nodes and the coordinator share the machine, so wall times include contention
between them.

Two builds were compared:

- **before greedy**: integration at 30c0565, where the final k-means solve used plain D² k-means++ seeding;
- **after greedy**: perf/efficiency at 6324e07, where k-means++ is greedy by default (2 + ⌊ln k⌋ candidates per
  step). The coreset constructions are the same in both builds; sensitivity sampling keeps plain D².

## Results

### Before greedy seeding

| nodes | algorithm | coreset points | ratio mean | ratio max | estimate error | wall s | snapshots KB |
|---|---|---|---|---|---|---|---|
| 1 | uniform | 4,197 | 1.0098 | 1.0102 | 2.53% | 3.6 | 848 |
| 1 | sensitivity α=2 | 3,942 | 1.2094 | 1.6103 | 3.38% | 3.6 | 796 |
| 1 | sensitivity α=theory | 3,506 | 1.1711 | 1.4925 | 1.78% | 3.6 | 708 |
| 1 | mean m=200 | 1,200 | 1.0000 | 1.0000 | 1.7e-09 | 21.9 | 242 |
| 2 | uniform | 9,222 | 1.1712 | 1.5048 | 1.65% | 4.1 | 1,863 |
| 2 | sensitivity α=2 | 8,600 | 1.3300 | 1.5013 | 1.95% | 4.0 | 1,737 |
| 2 | sensitivity α=theory | 8,045 | 1.3638 | 1.6056 | 3.00% | 4.0 | 1,625 |
| 2 | mean m=200 | 2,400 | 1.0000 | 1.0000 | 1.7e-09 | 12.1 | 485 |
| 4 | uniform | 14,922 | 1.1698 | 1.5043 | 3.00% | 5.3 | 3,014 |
| 4 | sensitivity α=2 | 13,779 | 1.4886 | 1.5001 | 1.09% | 4.8 | 2,783 |
| 4 | sensitivity α=theory | 12,629 | 1.3213 | 1.4823 | 1.26% | 4.8 | 2,551 |
| 4 | mean m=200 | 4,000 | 1.0000 | 1.0000 | 1.7e-09 | 9.2 | 808 |
| 8 | uniform | 27,478 | 1.1619 | 1.4836 | 1.35% | 6.9 | 5,551 |
| 8 | sensitivity α=2 | 25,395 | 1.1669 | 1.4986 | 0.80% | 6.3 | 5,130 |
| 8 | sensitivity α=theory | 22,049 | 1.3266 | 1.5003 | 0.47% | 5.9 | 4,454 |
| 8 | mean m=200 | 7,200 | 1.0000 | 1.0000 | 1.7e-09 | 7.1 | 1,455 |

### After greedy seeding

| nodes | algorithm | coreset points | ratio mean | ratio max | estimate error | wall s | snapshots KB |
|---|---|---|---|---|---|---|---|
| 1 | uniform | 4,191 | 1.0100 | 1.0118 | 2.41% | 3.8 | 847 |
| 1 | sensitivity α=2 | 3,897 | 1.0087 | 1.0089 | 3.66% | 3.6 | 787 |
| 1 | sensitivity α=theory | 3,845 | 1.0102 | 1.0113 | 3.73% | 3.6 | 777 |
| 1 | mean m=200 | 1,200 | 1.0000 | 1.0000 | 1.7e-09 | 42.8 | 242 |
| 2 | uniform | 9,186 | 1.0048 | 1.0051 | 0.68% | 4.0 | 1,856 |
| 2 | sensitivity α=2 | 8,590 | 1.0042 | 1.0046 | 1.58% | 3.9 | 1,735 |
| 2 | sensitivity α=theory | 7,711 | 1.0049 | 1.0053 | 1.54% | 3.8 | 1,558 |
| 2 | mean m=200 | 2,400 | 1.0000 | 1.0000 | 1.7e-09 | 22.6 | 485 |
| 4 | uniform | 14,925 | 1.0023 | 1.0025 | 0.48% | 4.3 | 3,015 |
| 4 | sensitivity α=2 | 13,830 | 1.0021 | 1.0022 | 0.88% | 4.2 | 2,794 |
| 4 | sensitivity α=theory | 11,898 | 1.0028 | 1.0029 | 0.42% | 4.0 | 2,404 |
| 4 | mean m=200 | 4,000 | 1.0000 | 1.0000 | 1.7e-09 | 14.1 | 808 |
| 8 | uniform | 27,456 | 1.0010 | 1.0011 | 0.12% | 5.5 | 5,546 |
| 8 | sensitivity α=2 | 25,357 | 1.0009 | 1.0010 | 0.13% | 5.4 | 5,122 |
| 8 | sensitivity α=theory | 22,254 | 1.0013 | 1.0014 | 0.50% | 4.7 | 4,496 |
| 8 | mean m=200 | 7,200 | 1.0000 | 1.0000 | 1.7e-09 | 14.6 | 1,455 |

**Reading the tables.**

- Before the change, the coresets were accurate (estimate error 0.5-3.4% for the sampling coresets) but the
  solutions were not: up to 1.61× the optimum, and 1.16-1.49× on average from 2 nodes up. Plain D² seeding leaves
  the small clusters uncovered, and 10 restarts of Lloyd do not recover them. Only the mean coreset, with a
  1200-7200-point union, was solved well in every run.
- With greedy seeding, every configuration is within 1.2% of the optimum (worst seed 1.0118, uniform on one node).
  Quality improves with the number of nodes, because the union grows: at 8 nodes, sensitivity α=2 reaches 1.0009
  with 25,357 coreset points.
- The mean coreset reaches the optimum (1.0000) with the smallest union and an estimate error near 1e-9, but it is
  the slowest to build: 14-43 s of wall time against 3.6-5.5 s for sampling.
- The provable α (`alpha=kmeans++`) gives slightly smaller coresets than α = 2 and no better solutions.

## Communication

| quantity | value |
|---|---|
| ingested by the nodes (all N) | 202 MB for 1M points, 202 B per point |
| raw dense payload | 160 B per point (20 doubles) |
| snapshots returned, 8 nodes | 4.4-5.5 MB for sampling (22-27k points), 1.5 MB for mean (7,200 points) |
| snapshots returned, 1 node | 0.7-0.8 MB for sampling, 0.24 MB for mean |

The coordinator sends dense points in the sparse encoding, so each point carries 20 indices next to its values:
202 B against 160 B of raw doubles, about 26% overhead. What comes back is under 3% of what went in.

## Caveat: the CLI's `cost=`

The coordinator prints `cost=` in its summary line. That value is the cost of the returned centers **on the
coreset**, an estimate of the full-data cost. It is not the full-data cost. The estimate error column above shows
how far apart the two were in this run (up to 3.7%). To measure the full-data cost, evaluate the printed centers on
the input, as done here.

## Reproducing

**Dataset.** Generated with NumPy (`numpy.random.default_rng(42)`):

- n = 1,000,000, d = 20, k = 20;
- cluster weights w_i = 0.5^(i/3) for i = 0..19, normalized; sizes floor(w_i n), with the remainder added to
  cluster 0 (sizes 208,362 down to 2,583);
- centers uniform in [-50, 50]^20; points = center + N(0, 2² I);
- rows shuffled, written as CSV with 4 decimals (166 MB).

**Build.**

```
mvn -B -pl coreset-grpc -am package -DskipTests
```

**Nodes** (i = 1..8; an N-node run uses nodes 1..N):

```
java -Xmx2g -jar coreset-grpc/target/coreset-node.jar --port 710$i --metrics-port 910$i
```

**Coordinator** (seeds 3, 4, 5; the node list has N entries):

```
java -Xmx4g -jar coreset-grpc/target/coreset-coordinator.jar \
  --nodes 127.0.0.1:7101,127.0.0.1:7102,... --input dist1m.csv \
  --k 20 --batch-size 10000 --workers 2 --seed 3 --restarts 10 \
  --algorithm sensitivity --param k=20 --param m=1000
```

Other configurations: `--algorithm uniform --param m=1000`; `--algorithm sensitivity --param k=20 --param m=1000
--param alpha=kmeans++`; `--algorithm mean --param m=200 --param lloydIterations=10`.

The coordinator prints one CSV line per center. Evaluate them on all points (sum over points of the minimum squared
distance to a center) and divide by the baseline cost. Bytes on the wire are the deltas of the node counters
above, read from `http://127.0.0.1:910$i/metrics` before and after the run.
