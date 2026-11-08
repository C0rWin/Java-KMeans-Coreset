# Coreset benchmarks

This directory holds benchmark results produced by `coreset-testbed`, one
dated directory per run. Each run directory contains:

- `summary.md`: the aggregated tables, one per dataset.
- `summary.csv`: the same data with every column (CI coverage, weight error,
  distance evaluations, serialized bytes, load average, ...).
- `reports.jsonl.gz`: every per-run validity report, one JSON object per line
  (`coreset-testbed aggregate --in <file>` rebuilds the summaries from it).
- `shared.json`: what the shared full-data candidates cost per dataset and k.
- `config.json`, `environment.json`, `command.txt`: what ran, on what, and how.

## 2026-11-08: standard suite (current)

Results are in [`2026-11-08/summary.md`](2026-11-08/summary.md). The suite ran
740 runs (148 configurations × 5 seeds) in 17.8 minutes on an Apple M2 Max
(12 cores) with JDK 17.0.5, at commit 847492f. The one-minute load average
stayed between 2.8 and 18.5 (mean 7.2). The exact command is in
[`2026-11-08/command.txt`](2026-11-08/command.txt).

The grid and its exclusions are the same as in the 2026-11-07 run below:
datasets × k ∈ {10, 50} × m ∈ {200, 1000, 4000} × four algorithms ×
{offline, stream with 64 blocks} × 5 seeds, without the mean coreset at
m ≥ 1000 and without bicriteria sensitivity sampling at k = 50 on the
1e6-point and sparse inputs.

Distortion by coreset size, averaged over the four datasets and both k (mean
over 5 seeds of each run's maximum; offline / stream):

| m | uniform | sensitivity k-means++ | sensitivity bicriteria | mean |
|---|---|---|---|---|
| 200 | 1.97 / 4.20 | 2.06 / 3.15 | 2.69 / 6.21 | 1.10 / 1.15 |
| 1000 | 1.12 / 1.24 | 1.15 / 1.25 | 1.26 / 1.48 | – |
| 4000 | 1.04 / 1.06 | 1.05 / 1.07 | 1.06 / 1.07 | – |

At k = 10 and m ≥ 1000, averaged over the four datasets:

| m | algorithm | distortion offline / stream | downstream rel. cost offline / stream |
|---|---|---|---|
| 1000 | uniform | 1.062 / 1.129 | +1.7% / +3.2% |
| 1000 | sensitivity k-means++ | 1.053 / 1.093 | +0.7% / +3.1% |
| 1000 | sensitivity bicriteria | 1.128 / 1.114 | +3.7% / +4.4% |
| 4000 | uniform | 1.027 / 1.039 | -0.5% / -0.1% |
| 4000 | sensitivity k-means++ | 1.025 / 1.034 | -0.3% / -0.1% |
| 4000 | sensitivity bicriteria | 1.042 / 1.049 | +0.0% / +0.4% |

What the numbers say:

- **Size matters more than the algorithm.** At k = 10, m ≥ 1000 every
  sampling method lands within a few percent of the full data, and the best
  of 10 solutions on the coreset costs at most 4.4% more than the full-data
  reference on average (individual cells range from -1.8% to +8.2%). At
  m = 4000 the coreset solution is as good as or slightly better than the
  reference, which is itself not optimal. At m = 200 with k = 50 every sampling
  method fails (2.7-5.0 offline, 5.0-14.4 in stream mode); ESA'22 uses coreset
  sizes of at least 50k for this reason.
- **Uniform sampling is a strong baseline on well-spread data.** On balanced
  clusters, the ESA'22 instance and the sparse documents it matches or beats
  k-means++ sensitivity sampling. Sensitivity sampling wins on skewed cluster
  sizes: on `gauss-unbalanced-1m` at k = 10, m = 1000 it scores 1.067 against
  1.150 offline and 1.128 against 1.320 in stream mode.
- **Bicriteria seeding loses to k-means++ seeding almost everywhere** and
  keeps fewer distinct points (723 of m = 1000 on sparse data at k = 10,
  against 992 with k-means++).
- **The mean coreset (Barger-Feldman) is the most accurate per point at small
  m.** With 200 points it scores 1.008-1.057 on the Gaussian and sparse data,
  except 1.19 (offline) and 1.31 (stream) at k = 50 on the unbalanced mixture.
  In all 80 runs it never underestimated a cost (0 one-sided violations), and
  weight and second moment were exact up to rounding (≤ 3e-14). It loses on
  the ESA'22 instance (1.25 offline, 1.49 in stream mode at k = 10). It is
  also by far the most expensive to build: 1.2-7.6 s CPU offline at m = 200,
  against tens of milliseconds for sensitivity sampling.
- **Stream mode degrades with composition, mostly through size.** At k = 10,
  m = 1000, the level-0 summary and the top level (64 blocks, 7 reductions)
  differ by at most 0.074 for the sampling methods. The final stream coreset
  is worse than the offline one (for example 1.320 against 1.150 for uniform
  on `gauss-unbalanced-1m`), partly because a merged sample keeps fewer
  distinct points: about 655 for uniform at m = 1000, against 994-1000
  offline. The mean coreset degrades with depth on the ESA'22 instance (1.20
  to 1.49 at k = 10) and on the unbalanced mixture at k = 50 (1.18 to 1.31).
- **Candidates computed on the coreset find the worst case.** For the sampling
  methods, 87% of runs reach their maximum distortion on a Lloyd solution of
  the coreset (`lloyd@C`, `bestOf@C`). For the mean coreset the maximum is
  always on a full-data candidate.
- **Confidence intervals hold on average, not per run.** Across the 210
  offline sensitivity runs, 93.6% of the 95% intervals contain the true cost.
  Within one run all candidates share one sample, so coverage is often all or
  nothing: 9 runs fall below 50%.
- **Validation costs more than construction for the fast algorithms.** Over
  the suite, construction took 688 s CPU, evaluation 184 s plus 287 s for the
  downstream solves on the coresets, and the shared full-data candidates 167 s
  (`2026-11-08/shared.json`). Uniform builds a coreset of 1e6 points in about
  2 ms but takes about 100 ms to validate. In distance evaluations, validation
  (2.9e10) costs more than all constructions together (2.5e10). Each run
  reads the full data 1.04 times on average.

### Why distortion differs from 2026-11-07

The 2026-11-07 run used the core before the efficiency work. Two things
changed:

- `CandidateSolutions` and the Lloyd solves on the coreset now seed with
  greedy k-means++, which finds stronger solutions. Stronger candidates expose
  more of a coreset's error, so measured distortion rose slightly, mostly at
  small m. Uniform compression output is bit-identical to the previous run;
  its distortion changed only because the candidates changed.
- Construction got cheaper: median build CPU per configuration is 1.9× lower
  for the mean coreset and 2.3× lower for uniform, and the mean coreset
  allocates 187 MB per configuration instead of 1527 MB.

## 2026-11-07: standard suite (previous)

Results are in [`2026-11-07/summary.md`](2026-11-07/summary.md), command and
commit in [`2026-11-07/command.txt`](2026-11-07/command.txt): 740 runs in 22.9
minutes at commit 6dd0059, before the efficiency work. An earlier run with
fewer exclusions, including mean at m = 1000 on the 1e5-point inputs and
bicriteria at k = 50 on sparse data, is kept in
[`2026-11-07/extended/`](2026-11-07/extended/summary.md).

Datasets (both runs):

- `gauss-unbalanced-1m`: 1e6 × 10, 20 Gaussian clusters (σ = 3), Zipf(1.5)
  cluster sizes.
- `gauss-balanced-200k`: 2e5 × 20, 50 equal clusters (σ = 5).
- `esa22-k10-a5`: the ESA'22 benchmark instance with k = 10, α = 5
  (1e5 × 50).
- `sparse-bow-100k`: 1e5 documents over a 20,000-term vocabulary, 20 topics,
  about 38 nonzeros per document.

## What is measured

A coreset C of a weighted point set P is good when, for every set of k centers
Q, `cost(C, Q)` is close to `cost(P, Q)`. The *distortion* of C on Q is
`max(r, 1/r)` with `r = cost(C, Q) / cost(P, Q)`. The true distortion is a
supremum over all Q, and deciding it is co-NP hard (Schwiegelshohn and
Sheikh-Omar, ESA 2022), so every number here is a **lower bound** measured on
a finite set of candidate solutions:

- `kmeans++@P#i`, `convex@B#i`: the ESA'22 candidates on the full data, five
  k-means++ seedings and five random convex combinations of bicriteria
  centers.
- `bestOf@P`: the best of 3 k-means++ and Lloyd runs on the full data. It
  doubles as the reference solution for the downstream comparison.
- `lloyd@C#i`, `bestOf@C`: solutions computed on the coreset itself, which are
  the ones a user would deploy. They are evaluated on P as well.

The full-data candidates are generated once per (dataset, k) and shared by all
runs. Algorithms are therefore compared on identical queries, and the
expensive full-data passes are paid once.

Each report also carries:

- **Integrity**: relative error of total weight, first moment and second
  moment against the input. The mean coreset preserves all three exactly, and
  the sampling coresets preserve them in expectation. For the mean coreset,
  `oneSidedViolations` counts candidates whose cost it underestimates. That
  count must be 0 (Barger and Feldman, Lemma 1).
- **Confidence intervals**: for offline sensitivity samples, a 95% CLT interval
  of `cost(P, Q)` computed from the sample alone, for each full-data candidate.
  `ci_coverage` is the fraction of intervals that contain the true cost.
  Candidates optimized on the coreset get no interval, because they are not
  independent of the sample.
- **ESA'22 δ-sweep** (benchmark instance only): the paper's lower bound. For
  each deficiency threshold δ it takes the clustering with the most deficient
  clusters, puts centers only at non-deficient cluster means, and compares the
  exact full cost with the coreset cost.
- **Stream levels**: in stream mode the input arrives in b = 64 blocks. Each
  block is compressed and added to a merge-and-reduce tree. Right after
  `2^l` blocks the tree holds a single summary that has been reduced
  `l + 1` times. Its distortion against exactly those blocks is reported per
  level (`level 0` and `top level` in the tables), so degradation under
  composition is measured, not assumed.
- **Downstream**: `rel. cost = cost(P, bestOf@C) / cost(P, bestOf@P) - 1`,
  where bestOf@C is the best of 10 restarts on the coreset. This is
  optimization quality, not coreset quality. ESA'22 explains why the two
  differ.
- **Construction cost**: wall and CPU time of the constructing thread,
  allocated bytes (`ThreadMXBean`), heap peak above the pre-run baseline (pool
  peaks plus a 1 ms heap sampler, after a GC), GC count and time,
  distance evaluations counted by the algorithms, output size and protobuf
  size.
- **Validation cost**: wall time and distance evaluations spent evaluating the
  run, plus how many full-data passes that took (normally 1). The shared
  candidates are accounted separately in `shared.json`.

Aggregation follows ESA'22. Per (dataset, mode, algorithm, k, m) group the
headline `distortion` is the **mean over the 5 seeds of each run's maximum
distortion**. `sd` and `max` (the worst seed) are reported next to it.
The `pass` column is the share of runs whose distortion stayed at or below 1.2
with clean integrity. The 1.2 threshold is a configuration input, not a
theoretical constant. Distortions above 5 and above 10 are flagged as failure
and catastrophic failure, following Draganov et al. (SIGMOD'24).

## Reproducing

```
mvn -B -pl coreset-testbed -am package -DskipTests
java -Xms16g -Xmx16g -XX:+UseG1GC -XX:+AlwaysPreTouch \
  -jar coreset-testbed/target/coreset-testbed.jar suite standard --out results/
```

All randomness derives from the suite seed (42), so the same commit, flags and
suite produce identical coresets, costs and distortions. Only timings, memory
and environment fields differ. The test suite checks this
(`ReproducibilityTest`). `suite smoke` runs every dataset kind, algorithm and
mode, including distributed over in-process nodes and over gRPC, in about a
minute. To run your own grid:

```
java -jar coreset-testbed/target/coreset-testbed.jar run --config my.yaml --out results/
```

The built-in suites in `coreset-testbed/src/main/resources/suites/` document
the config format. Public datasets (`datasets list`) are never downloaded
implicitly. `datasets fetch <name>` or `--allow-download` stores them under
`~/.cache/coreset-testbed`, and the file's SHA-256 is checked against the
descriptor. On first use each one is converted to a binary point file next to
the download (`datasets convert <name>` does it ahead of time); see
[../datasets.md](../datasets.md) for the datasets, preprocessing variants and
the format.

## Caveats that apply to every run

- Timings come from one JVM run per suite with no warmup discarded. Barrett et
  al. (OOPSLA'17) show steady state is not guaranteed, so all iterations are
  reported. For per-kernel performance use JMH.
- Wall time is only meaningful on an idle machine. Each report records the
  one-minute load average. When it exceeds the core count, compare CPU times,
  allocation and distance evaluations instead.
- The first-moment error is normalized by the data scale,
  |Δ| / sqrt(W · E) with E the second moment, so it stays meaningful on
  centered data such as the ESA'22 instance (0.014-0.19 there for the
  sampling coresets).
