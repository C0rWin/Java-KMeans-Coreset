# Performance

> **Status: preliminary, noisy host.** Every number below was measured on a 12-core laptop that was shared with
> several concurrent builds and stress runs. The 1-minute load average was 5 when the JMH suite started and 81 when it
> ended, and 167-228 during the harness and JFR runs (recorded per phase in the environment table). Treat absolute
> numbers as pessimistic and the error bars as wide; ratios measured in the same run (protobuf vs Thrift, sparse vs
> dense, 1 vs 12 threads) are more trustworthy than absolutes. The authoritative pass is meant to be rerun with
> `coreset-bench/run-benchmarks.sh` in a quiet window on the merged branch, and these tables replaced with its
> `summary.md`.

## How to run

Everything (build, JMH, ingestion harness, Thrift parity, JFR profiles) is one command. It writes raw JMH JSON, the
harness markdown, JFR recordings with `jfr view` summaries and a `summary.md`, and records host, JVM, git revision and
the load average before and after every phase:

```
JAVA_HOME=<jdk17> JFR_TOOL=<jdk21+>/bin/jfr coreset-bench/run-benchmarks.sh            # full, about 40 min
JAVA_HOME=<jdk17> coreset-bench/run-benchmarks.sh --quick                              # about 10 min
coreset-bench/run-benchmarks.sh --skip-build --no-jfr --jmh 'CodecBenchmark|DistanceBenchmark'
```

Results go to `coreset-bench/results/<UTC timestamp>/` (git-ignored). `JFR_TOOL` is optional; JDK 17's `jfr` has no
`view` command, so without it only `jfr summary` is written. Individual pieces:

```
mvn -B -pl coreset-bench -am package -DskipTests
java -jar coreset-bench/target/benchmarks.jar -prof gc -rf json -rff jmh.json [regex]
java -Xms8g -Xmx8g -XX:+AlwaysPreTouch -cp coreset-bench/target/benchmarks.jar \
     io.github.c0rwin.coreset.bench.perf.NodeIngestHarness [--points N --workers 1,2,4,8 --grpc-nodes 1,2,3,4 ...]
java -Xms8g -Xmx8g -XX:+AlwaysPreTouch -cp coreset-bench/target/benchmarks.jar \
     io.github.c0rwin.coreset.bench.perf.ParityHarness [--points N --repeats R ...]
```

`mvn -B -pl coreset-bench -am verify` runs `BenchSmokeTest`: one JMH benchmark in process and both harnesses on toy
sizes. The harnesses check themselves (every batch applied, gathered snapshot complete), so the smoke test fails on a
broken pipeline, not only on a crash.

## Methodology

**Machine.** Apple M2 Max (8 performance + 4 efficiency cores, 12 logical CPUs), 96 GiB RAM, macOS (Darwin 27.0.0
arm64), Amazon Corretto 17.0.5, G1 (default). No core pinning or frequency control is available on macOS, and the
scheduler may place a fork on efficiency cores; this is another reason for the wide intervals.

**JMH** (`coreset-bench/src/main/java/.../bench/jmh`, JMH 1.37):
- `@Fork(2)`, `@Warmup(3 x 1 s)`, `@Measurement(5 x 1 s)`, average time per operation, 99.9% confidence intervals.
- Each fork runs with a fixed heap (`-Xms` = `-Xmx`, 2 or 4 GiB). Every run uses `-prof gc`; `B/op` below is
  `gc.alloc.rate.norm`, allocated bytes per operation, which is exact and insensitive to load.
- Inputs come from `@State` with seeded generators. Results go to `Blackhole` or are returned. Distance and sampler
  kernels walk 256 distinct pairs or 1024 draws per invocation (`@OperationsPerInvocation`), so one pair cannot be
  memorised by caches or branch predictors.
- Reduced settings: `CompressBenchmark` runs `mean` with 3 Lloyd iterations instead of the default 20 (its cost is
  linear in the iteration count). The JMH tables come from one full run at commit `bd24310`. The three benchmarks
  added after it (`CoresetOpsBenchmark`, `GrpcMetricsBenchmark`, the two probe methods of `MetricsBenchmark`) come
  from a `--quick` run: 1 fork, 2 x 1 s warmup, 3 x 1 s measurement.

**Ingestion harness** (`NodeIngestHarness`, not JMH): producers cycle over 256 distinct pre-generated 1000-point
batches (Gaussian mixture, d = 16, 10 clusters) with unique seqs, so every batch is compressed with its own
seq-derived generator while the resident data stays small. Each configuration runs once as warmup, then
`--repeats` times; tables give the median points/s with the min-max range, and the other columns come from the
median run. Throughput is end to end: first ingest until `awaitApplied` (in process) or until the gathered
snapshot is complete (gRPC). Latency percentiles come from the node's own `RecordingMetrics`. `node.ingest.latency_ns`
is the producer-side `ingest` call, all outcomes included; batch compress is the worker-side `node.batch.compress`
span. A BUSY reply is retried after 50 us, and the BUSY rate is `busy / (busy + batches)`. Allocation is the sum
of `ThreadMXBean.getThreadAllocatedBytes` over all live threads, taken before the workers exit. Peak heap is the sum of
per-pool peaks after `resetPeakUsage`, an upper bound; the baseline in brackets is heap in use after a full GC at
the start, which is mostly the resident workload. gRPC runs put the client and every node in one JVM over Netty
loopback, so they compete for the same cores.

**Parity harness** (`ParityHarness`): the legacy `CoresetServiceHandler` runs in an in-process `TNonblockingServer`
on an ephemeral port and is driven through `TFramedTransport` + `TBinaryProtocol`, as `CoresetClient` does. The
gRPC side is one `CoresetNode` behind `CoresetNodeServer`, driven by `CoresetCoordinator`. Both get the same
distinct points (100k in the full run, 50k in the preliminary one) in 1000-point batches, with k = 10 and m = 200.
`NON_UNIFORM` is paired with `sensitivity` and `UNIFORM` with `uniform`. End-to-end time covers
initialisation, client-side conversion, transport, server-side merge-and-reduce and the final coreset fetch; for
Thrift, an in-order `isDone()` on the same connection precedes `getTotalCoreset()`. Wire bytes are measured in a
separate pass through a counting TCP relay, so they include Thrift framing and gRPC's HTTP/2 frames, headers and acks
alike. Quality: the legacy coreset is converted to a coreset-core `Coreset`, and both are scored by
`validate.Distortion` against one shared candidate set (3 x k-means++ on P, 3 x Lloyd on P, 3 x convex combinations
of bicriteria centers) plus 3 Lloyd solutions computed on each coreset.

**Profiling.** JFR with `settings=profile` on three hot paths: in-process ingestion (W = 4), gRPC ingestion
(2 nodes), and `KMeansBenchmark` seeding and Lloyd at n = 100k, k = 50, d = 16. The recordings were summarised with
JDK 27's `jfr view hot-methods|allocation-by-site`, which reads JDK 17 recordings. With so few samples (200-500 per
recording on the loaded host), the shares are good to a few percent.

## Environment of the preliminary harness and JFR pass

- date (UTC): 2026-11-07T13:54:44Z
- git: bd24310 (bench/perf), dirty
- os: Darwin 27.0.0 arm64
- cpu: Apple M2 Max, 12 logical cpus (8 performance + 4 efficiency), 96 GiB RAM
- java: openjdk version "17.0.5" 2022-10-18 LTS OpenJDK Runtime Environment Corretto-17.0.5.8.1 (build 17.0.5+8-LTS) OpenJDK 64-Bit Server VM Corretto-17.0.5.8.1 (build 17.0.5+8-LTS, mixed mode, sharing) 
- mode: quick
- harness JVM flags: -Xms8g -Xmx8g -XX:+AlwaysPreTouch

| phase | load avg before (1/5/15 min) | load avg after | seconds |
|---|---|---|---|
| build | 167.44 99.71 58.68 | 188.33 110.10 63.62 | 25 |
| jmh | 188.33 110.10 63.62 | 183.61 126.86 74.66 | 90 |
| node-ingest | 183.61 126.86 74.66 | 201.09 132.46 77.26 | 9 |
| parity | 201.09 132.46 77.26 | 215.02 137.71 79.77 | 10 |
| jfr-node | 215.02 137.71 79.77 | 219.82 139.99 80.91 | 3 |
| jfr-grpc | 219.82 139.99 80.91 | 220.96 141.55 81.81 | 8 |
| jfr-kmeans | 220.96 141.55 81.81 | 228.01 148.19 85.57 | 20 |


The JMH tables below come from a separate full run (both forks of every benchmark) whose load average went from
5.07 at the start to 81.6 at the end.

## JMH results

#### AliasSamplerBenchmark

| benchmark | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|
| build | 10000 | 80,625 | 1,393 | ns/op | 200,088 |
| build | 1000000 | 7,805,268 | 84,554 | ns/op | 20,000,092 |
| draw | 10000 | 5.13 | 0.06 | ns/op | 0 |
| draw | 1000000 | 4.95 | 0.59 | ns/op | 0 |

#### CodecBenchmark

| benchmark | nnz | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|
| protoDecode | 16 | 273 | 8.61 | us/op | 1,137,440 |
| protoDecode | 128 | 1,411 | 59.4 | us/op | 6,177,486 |
| protoEncode | 16 | 290 | 4.69 | us/op | 742,240 |
| protoEncode | 128 | 1,945 | 66.6 | us/op | 6,233,146 |
| thriftDecode | 16 | 592 | 17.5 | us/op | 2,195,208 |
| thriftDecode | 128 | 6,182 | 69.1 | us/op | 17,121,083 |
| thriftEncode | 16 | 695 | 23.6 | us/op | 1,683,880 |
| thriftEncode | 128 | 4,782 | 196 | us/op | 11,985,755 |

#### CompressBenchmark

| benchmark | algorithm | m | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|---|
| compress | uniform | 200 | 10000 | 0.07 | 0.00 | ms/op | 327,381 |
| compress | uniform | 200 | 100000 | 0.86 | 0.07 | ms/op | 3,207,435 |
| compress | uniform | 1000 | 10000 | 0.08 | 0.00 | ms/op | 354,697 |
| compress | uniform | 1000 | 100000 | 0.85 | 0.01 | ms/op | 3,236,084 |
| compress | sensitivity | 200 | 10000 | 1.99 | 0.04 | ms/op | 652,724 |
| compress | sensitivity | 200 | 100000 | 24.5 | 0.34 | ms/op | 6,412,835 |
| compress | sensitivity | 1000 | 10000 | 2.02 | 0.05 | ms/op | 697,919 |
| compress | sensitivity | 1000 | 100000 | 30.3 | 0.51 | ms/op | 6,460,541 |
| compress | mean | 200 | 10000 | 113 | 2.31 | ms/op | 811,068 |
| compress | mean | 200 | 100000 | 1,137 | 12.3 | ms/op | 6,211,552 |
| compress | mean | 1000 | 10000 | 545 | 7.91 | ms/op | 1,652,878 |
| compress | mean | 1000 | 100000 | 5,964 | 1,338 | ms/op | 7,053,136 |

#### DistanceBenchmark

| benchmark | dim | nnz | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|
| sparseDense | 4096 | 16 | 43.1 | 7.69 | ns/op | 0 |
| sparseDense | 4096 | 128 | 246 | 29.8 | ns/op | 0 |
| sparseDense | 4096 | 1024 | 1,553 | 159 | ns/op | 0 |
| sparseDense | 1048576 | 16 | 53.8 | 8.77 | ns/op | 0 |
| sparseDense | 1048576 | 128 | 429 | 91.9 | ns/op | 0 |
| sparseDense | 1048576 | 1024 | 7,994 | 1,335 | ns/op | 0 |
| sparseSparse | 4096 | 16 | 133 | 17.9 | ns/op | 0 |
| sparseSparse | 4096 | 128 | 1,708 | 162 | ns/op | 0 |
| sparseSparse | 4096 | 1024 | 12,451 | 4,196 | ns/op | 0 |
| sparseSparse | 1048576 | 16 | 73.2 | 12.8 | ns/op | 0 |
| sparseSparse | 1048576 | 128 | 1,088 | 8.97 | ns/op | 0 |
| sparseSparse | 1048576 | 1024 | 9,240 | 209 | ns/op | 0 |

#### KMeansBenchmark

| benchmark | d | k | n | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|---|
| lloydIteration | 16 | 10 | 20000 | 6.48 | 0.23 | ms/op | 566,203 |
| lloydIteration | 16 | 10 | 100000 | 32.8 | 1.32 | ms/op | 2,806,217 |
| lloydIteration | 16 | 50 | 20000 | 23.6 | 1.01 | ms/op | 589,733 |
| lloydIteration | 16 | 50 | 100000 | 120 | 5.96 | ms/op | 2,829,780 |
| lloydIteration | 128 | 10 | 20000 | 43.7 | 2.02 | ms/op | 597,583 |
| lloydIteration | 128 | 10 | 100000 | 246 | 7.16 | ms/op | 2,837,671 |
| lloydIteration | 128 | 50 | 20000 | 191 | 1.41 | ms/op | 746,611 |
| lloydIteration | 128 | 50 | 100000 | 974 | 25.8 | ms/op | 2,986,792 |
| seed | 16 | 10 | 20000 | 5.49 | 3.88 | ms/op | 560,147 |
| seed | 16 | 10 | 100000 | 21.9 | 2.10 | ms/op | 2,800,156 |
| seed | 16 | 50 | 20000 | 16.6 | 0.65 | ms/op | 560,313 |
| seed | 16 | 50 | 100000 | 95.5 | 16.1 | ms/op | 2,800,353 |
| seed | 128 | 10 | 20000 | 21.6 | 2.76 | ms/op | 560,156 |
| seed | 128 | 10 | 100000 | 203 | 14.3 | ms/op | 2,800,244 |
| seed | 128 | 50 | 20000 | 108 | 8.39 | ms/op | 560,359 |
| seed | 128 | 50 | 100000 | 1,573 | 1,405 | ms/op | 2,800,654 |

#### MergeReduceTreeBenchmark

| benchmark | algorithm | m | metrics | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|---|---|---|
| add | uniform | 200 | noop | 16.0 | 3.51 | us/op | 22,666 |
| add | uniform | 200 | recording | 26.3 | 4.11 | us/op | 23,557 |
| add | uniform | 1000 | noop | 136 | 54.3 | us/op | 111,585 |
| add | uniform | 1000 | recording | 116 | 25.0 | us/op | 112,595 |
| add | sensitivity | 200 | noop | 160 | 19.6 | us/op | 37,587 |
| add | sensitivity | 200 | recording | 178 | 28.6 | us/op | 38,970 |
| add | sensitivity | 1000 | noop | 604 | 154 | us/op | 183,887 |
| add | sensitivity | 1000 | recording | 496 | 41.7 | us/op | 185,824 |

#### MetricsBenchmark

| benchmark | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|
| count1 | 5.37 | 0.35 | ns/op | 0 |
| count12 | 12.2 | 2.47 | ns/op | 0 |
| count4 | 6.44 | 1.50 | ns/op | 0 |
| longAdder1 | 6.90 | 1.96 | ns/op | 0 |
| longAdder12 | 9.93 | 1.43 | ns/op | 0 |
| longAdder4 | 5.69 | 0.80 | ns/op | 0 |
| record1 | 18.9 | 10.00 | ns/op | 0 |
| record12 | 203 | 13.4 | ns/op | 0 |
| record4 | 90.5 | 16.8 | ns/op | 0 |
| span | 2,390 | 278 | ns/op | 240 |

#### Added after the full run (quick mode: 1 fork, 3 x 1 s)

| benchmark | score | ± 99.9% CI | unit | B/op |
|---|---|---|---|---|
| CoresetOpsBenchmark.protoDecodeCoreset (m=1000) | 167 | 215 | us/op | 524,152 |
| CoresetOpsBenchmark.union2 (m=1000) | 15.9 | 64.5 | us/op | 32,152 |
| CoresetOpsBenchmark.unionFold8 (m=1000) | 234 | 1,324 | us/op | 577,240 |
| GrpcMetricsBenchmark.sendAndAck | 344 | 619 | ns/op | 288 |
| MetricsBenchmark.allocatedBytesProbe | 93.3 | 46.8 | ns/op | 0 |
| MetricsBenchmark.threadCpuTimeProbe | 1,364 | 2,863 | ns/op | 0 |

## Ingestion throughput (preliminary)

## In-process CoresetNode: 400000 points in batches of 1000, sensitivity {k=10, m=200}, 2 producers, queue 64

| mode | W | points/s (median) | min-max | ingest p50 / p99 (us) | batch compress p50 / p99 (us) | BUSY rate | alloc MB/s | GC n / ms | peak heap MiB (baseline) | snapshot pts | weight err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| single-writer | 1 | 2942618 | 2873205-3012031 | 12.1 / 246.8 | 210 / 440 | 0.7947 | 336 | 0 / 0 | 127 (79) | 407 | 5.9e-02 |
| single-writer | 2 | 4529548 | 3533465-5525631 | 19.3 / 228.4 | 204 / 514 | 0.5238 | 616 | 0 / 0 | 127 (79) | 803 | 3.8e-03 |
| single-writer | 4 | 10317974 | 10264196-10371752 | 12.0 / 47.9 | 208 / 440 | 0.2870 | 1139 | 0 / 0 | 131 (79) | 1592 | 1.2e-03 |
| single-writer | 8 | 10526874 | 9874227-11179521 | 12.2 / 193.5 | 212 / 3883 | 0.0000 | 1091 | 0 / 0 | 131 (79) | 3321 | 1.6e-02 |
| cas | 1 | 2610307 | 2132573-3088040 | 12.0 / 53.0 | 206 / 465 | 0.7994 | 340 | 0 / 0 | 127 (79) | 391 | 2.2e-03 |
| cas | 2 | 5506462 | 5355410-5657515 | 14.3 / 67.1 | 208 / 528 | 0.6396 | 623 | 0 / 0 | 127 (79) | 377 | 2.0e-02 |
| cas | 4 | 8850674 | 6230445-11470904 | 14.8 / 53.0 | 206 / 403 | 0.1786 | 1268 | 0 / 0 | 131 (79) | 396 | 1.7e-02 |
| cas | 8 | 12866606 | 11752047-13981166 | 12.2 / 39.7 | 208 / 1942 | 0.0000 | 1550 | 0 / 0 | 131 (79) | 412 | 2.2e-02 |

## gRPC over Netty loopback: 400000 points, 2 workers per node (single-writer), 2 producers, client and nodes in one JVM

| nodes | points/s (median) | min-max | node ingest p50 / p99 (us) | BUSY resends | alloc MB/s | GC n / ms | peak heap MiB (baseline) | proto bytes out / point | snapshot pts | weight err |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 784260 | 678348-890172 | 5.7 / 52.5 | 0 | 1085 | 1 / 2 | 497 (83) | 166.0 | 773 | 1.1e-02 |
| 2 | 2402638 | 1904556-2900720 | 5.7 / 37.6 | 0 | 2337 | 0 / 0 | 816 (84) | 166.0 | 1604 | 3.5e-03 |
| 3 | 2520814 | 1476313-3565315 | 6.7 / 493.6 | 0 | 4403 | 0 / 0 | 820 (84) | 166.0 | 2435 | 2.9e-03 |
| 4 | 3008980 | 2800011-3217949 | 6.8 / 236.5 | 0 | 3996 | 0 / 0 | 821 (85) | 166.0 | 3258 | 7.9e-03 |

**Reading the table.** BUSY is backpressure working as designed: two producers outpace one or two workers, so most
offers bounce until W = 8. In-process throughput reaches about 10-13 M points/s at W = 4-8, while one gRPC node manages
0.8 M points/s. The codec alone explains most of that gap: protobuf encode plus decode costs 0.56 ms per 1000-point
batch at nnz 16 (CodecBenchmark), about 1.8 M points/s per core, and here the client, Netty and the nodes all share
the overloaded cores. Single-writer snapshots are W times larger than CAS snapshots (W private trees against one
shared tree), as documented in `CoresetNode`. The weight error column is |sum of coreset weights - input weight| /
input weight. Sensitivity sampling is unbiased, not weight-preserving, so errors of 1e-3 to 6e-2 after hundreds of
merges are sampling noise, not loss.

## Legacy Thrift vs gRPC parity (preliminary)

## Serialized size of one 1000-point batch (dim 131072)

| nnz | protobuf bytes | Thrift binary bytes | Thrift / protobuf |
|---|---|---|---|
| 16 | 198025 | 224020 | 1.13 |
| 128 | 1416858 | 1568020 | 1.11 |

## Legacy Thrift vs gRPC node: 50000 points (d=16, 10 clusters) in 50 batches, k=10, m=200

| stack | algorithm | workers | end-to-end s (median) | min-max | points/s | wire bytes client->server | server->client | coreset pts | weight err | max distortion | mean distortion |
|---|---|---|---|---|---|---|---|---|---|---|---|
| thrift | NON_UNIFORM | 1 | 0.879 | 0.777-0.981 | 56877 | 11202803 | 94193 | 412 | 2.9e-16 | 1.1512 | 1.0603 |
| grpc | sensitivity | 1 | 0.142 | 0.122-0.161 | 352926 | 8307063 | 66162 | 388 | 7.4e-03 | 1.1165 | 1.0580 |
| grpc | sensitivity | 4 | 0.066 | 0.056-0.076 | 756826 | 8306756 | 248427 | 1485 | 1.4e-03 | 1.0545 | 1.0166 |
| thrift | UNIFORM | 1 | 0.107 | 0.102-0.112 | 468764 | 11202803 | 94193 | 414 | 0.0e+00 | 1.1127 | 1.0380 |
| grpc | uniform | 1 | 0.062 | 0.054-0.069 | 807805 | 8307155 | 68646 | 402 | 0.0e+00 | 1.1190 | 1.0453 |
| grpc | uniform | 4 | 0.046 | 0.040-0.052 | 1088650 | 8307121 | 244739 | 1463 | 0.0e+00 | 1.0296 | 1.0125 |

**Recommendation: delete the Thrift stack.** On every measured axis gRPC is ahead or level:

- End to end, sensitivity on one worker takes 0.14 s against NON_UNIFORM's 0.88 s (6.2x). Uniform takes 0.06 s against
  0.11 s. With 4 workers the gRPC node goes further still, which the single-threaded `TNonblockingServer` handler
  cannot do.
- Bytes on the wire client to server drop by 26% (8.31 MB vs 11.20 MB for 50k points), and payload size by 11-13%
  per batch.
- Encode is 2.4x faster at nnz 16 and decode 2.2x. At nnz 128 encode is 2.5x and decode 4.4x faster, with 1.9-2.8x less
  allocation (CodecBenchmark).
- Distortion is equivalent: max 1.12 vs 1.15 (sensitivity vs NON_UNIFORM) and 1.12 vs 1.11 (uniform) on the shared
  candidate set.
- The legacy stack's one apparent edge, exact weight conservation, comes from `NormalizeWeightsDecorator`
  rescaling the sample to the input weight, a deliberate bias rather than a better estimator. The legacy protocol
  also has no acknowledgements or deduplication (`compressPoints` is oneway), so it cannot retry safely.

The deletion should wait only for the quiet-window rerun to confirm these ratios.

## Hotspots (JFR)

Hot path 1, in-process ingestion (W = 4, sensitivity k = 10, m = 200, 1000-point batches):

| CPU samples | method | allocation pressure | site |
|---|---|---|---|
| 36.6% | `SparseVector.squaredDistance(SparseVector)` | 30.3% | `Seeding$D2.<init>` (5 arrays of n) |
| 19.6% | `Seeding$D2.add` | 19.9% | `AliasSampler.of` (4 arrays of n) |
| 15.5% | `SensitivityCoreset.compressWithProvenance` | 17.3% | `SensitivityCoreset.compressWithProvenance` |
| 8.3% | `Session.ingest` (producer side) | 3.5% | `ImmutableCollections.listFromArray` (`List.copyOf`) |
| 4.1% | `AliasSampler.of` | 2.9% | `Arrays.copyOf(Object[])` |
| 3.1% | `Coreset.<init>` | 2.4% | `AliasSampler.drawCounts` |

Hot path 2, gRPC ingestion (2 nodes, 1 worker each, client in the same JVM):

| CPU samples | method | allocation pressure | site |
|---|---|---|---|
| 22.0% | `SparseVector.squaredDistance(SparseVector)` | 11.8% | `SparseVector.of` (clones the arrays below) |
| 11.0% | `Seeding$D2.add` | 11.3% | `ProtoCodec.fromProto(SparseVector)` (int[]/double[] copies) |
| 8.4% | protobuf `SparseVector$Builder.mergeFrom` | 8.8% | `Arrays.copyOf(double[])` (protobuf list growth) |
| 7.8% | protobuf `DoubleArrayList.addDouble` | 8.0% | `DoubleArrayList.addDouble` |
| 2.6% | `Session.ingest` | 6.3% + 5.6% | `IntArrayList` copy and growth |
| 2.6% | protobuf `SparseVector.writeTo` | 4.3% | `WeightedPoint.toBuilder` |

Hot path 3, k-means (n = 100k, k = 50, d = 16):

| benchmark | CPU samples | method | allocation |
|---|---|---|---|
| seed | 67.3% | `SparseVector.squaredDistance(SparseVector)` | 55% in `KMeansPlusPlus.seed` (4 arrays of n) |
| seed | 29.0% | `KMeansPlusPlus.seed` (loop around it) | |
| lloydIteration | 78.1% | `SparseVector.squaredDistance(double[], double)` | 59% in `WeightedLloyd.run` (per-call arrays) |
| lloydIteration | 14.3% | `WeightedLloyd$Lloyd.assign` | 21% `SparseVector.dense` (output centers) |

## Findings and recommended fixes

None of these were applied in this change. The files involved are owned by the in-flight correctness fixes
(`SparseVector`, `KMeansPlusPlus`, `WeightedLloyd`, `MeanCoreset`, `SensitivityCoreset`, `MergeReduceTree`,
`Session`, `CoresetNode`), and the plan is to apply efficiency work after those land and re-measure on the merged
branch. Numbers are from the tables above.

1. **The sparse-sparse merge distance dominates seeding.** It takes 37% of node CPU, 22% on the gRPC path and 67% of
   k-means++. Merging two random supports costs 2-7 ns per stored entry (1.1-1.7 us at nnz 128), while gathering
   against a dense array costs 1.5-2.7 ns per entry in cache (246 ns at nnz 128, dim 4096). Seeding
   (`KMeansPlusPlus.seed`, `Seeding.D2.add`) computes n distances to each new center. Scattering that center into
   a reusable dense buffer once (O(nnz_c) to fill and to clear) turns every distance into a gather. Exactness must be
   kept: accumulate (x_i - c_i)^2 over the point's support and add the center's uncovered entries, instead of using
   the norm expansion that the correctness review flagged.
2. **Lloyd is compute-bound in sparse-dense distances (78%).** Nothing is wasted per distance. The next step is
   algorithmic: Hamerly/Elkan bounds skip most distance evaluations after the first iterations. Each call also
   allocates 28 bytes per point (`vectors`, `weights`, `assignment`, `dist`).
3. **Per-batch scratch arrays.** A sensitivity compress allocates about 64 bytes per input point: 32 for the five D2
   arrays, 8 for the sensitivities, 20 for the alias table plus its two worklists, and 4 for the counts. That is 6.4 MB
   per 100k-point call. A uniform compress allocates 32 bytes per point for a sample of m. Fixes: per-worker
   reusable scratch; one worklist for Vose's small and large stacks (small grows from the front, large from the back,
   same LIFO order, so identical tables; -4 bytes per element); and, for uniform with equal weights, skipping the
   alias table entirely. With equal weights every alias probability is exactly 1.0, so a draw is `nextInt(n)` plus a
   discarded `nextDouble()`. Replaying that and sorting the m indices gives bit-identical output with no n-sized
   arrays.
4. **Coreset list copies.** `Coreset.union` copies point references four times: into an `ArrayList`, the two
   `addAll` `toArray` copies, then `List.copyOf`'s `toArray` and `listFromArray`. That is 32 KB for a 2000-point
   union. The canonical constructor adds a dimension-check pass on every construction. `CoresetCoordinator.gather`
   folds unions pairwise, which is quadratic: 577 KB to gather 8 x 1000 points. A `Coreset.Builder` that fills one
   exactly sized array and hands it to an immutable list view (dimensions checked as points arrive, union of k parts
   in one allocation and one pass) removes all but one copy. It also gives `MergeReduceTree.snapshot`,
   `Session.snapshot` and `ProtoCodec.fromProto(Coreset)` a zero-copy way to build their output.
5. **gRPC decode copies every vector twice.** `ProtoCodec.fromProto` copies the packed fields into fresh
   `int[]`/`double[]`, then `SparseVector.of` validates and clones them again. Together that is 23% of gRPC-path
   allocation, and 1.14 MB per 1000-point batch at nnz 16. An `of` variant that adopts arrays it has validated removes
   one copy. Protobuf's primitive-list growth (per-element `addIndices`/`addValues` on encode, varint index lists on
   decode) is another 29% of gRPC-path allocation; writing the packed fields directly with `CodedOutputStream` avoids
   the encode half.
6. **`Session.ingest` copies every batch** (`List.copyOf` plus a dimension check per point), costing 8% of in-process
   node CPU on the producer threads. On the gRPC path the list is freshly decoded and owned already, so the copy is
   pure overhead; the transport adapter could hand over ownership.
7. **Instrumentation cost.**
   - A `Span` costs about 2.4 us and 240 B. Most of the time is the thread CPU clock, about 1.4 us per read on macOS
     and read twice per span. The 240 B are three `name + suffix` strings, built and re-hashed on every close.
     With about 5 spans per batch that is about 12 us of a 200 us batch compress, roughly 5%. Fixes: cache the
     suffixed names per span name, and make `cpu_ns` opt-in or sampled.
   - `RecordingMetrics.record` scales poorly under contention: 19, 91 and 203 ns/op at 1, 4 and 12 threads, against
     7, 6 and 10 ns for a bare `LongAdder`, because the histogram buckets are one shared `AtomicLongArray`. Lazily
     striped bucket arrays would bring this near the `LongAdder` floor.
   - `count` stays at 5-12 ns.
   - `MetricNames.mrtreeLevelPoints/Merges` and `GrpcMetrics` build metric names per call or per message (288 B per
     send+ack pair); precomputing them is free.

## Scaling summary (preliminary)

| axis | observation |
|---|---|
| n (compress) | 10x more points: uniform 11-13x, sensitivity 12-15x, mean 10-11x time; linear, as expected |
| k (Lloyd iteration, n = 100k, d = 16) | k 10 to 50: 3.7x time (n*k distances, plus a fixed update cost) |
| d (Lloyd iteration, n = 100k, k = 10) | d 16 to 128: 7.5x time (O(nnz) distances) |
| k (seeding, n = 100k, d = 16) | k 10 to 50: 4.4x time |
| workers (in process, single-writer) | 1, 2, 4, 8 workers: 2.9, 4.5, 10.3, 10.5 M points/s (producer-bound at 8) |
| workers (in process, CAS) | 2.6, 5.5, 8.9, 12.9 M points/s |
| nodes (gRPC loopback, same JVM) | 1, 2, 3, 4 nodes: 0.78, 2.4, 2.5, 3.0 M points/s |
| threads (metrics record) | 1, 4, 12 threads: 19, 91, 203 ns/op (contended buckets) |
