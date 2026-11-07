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

## 2026-11-07: standard suite

Results are in [`2026-11-07/summary.md`](2026-11-07/summary.md). The suite
ran 740 runs in 22.9 minutes on an Apple M2 Max (12 cores) with JDK 17.0.5.
The exact command and commit are in
[`2026-11-07/command.txt`](2026-11-07/command.txt).

**Scaled down.** The full grid is datasets × k ∈ {10, 50} ×
m ∈ {200, 1000, 4000} × four algorithms × {offline, stream with 64 blocks} ×
5 seeds. Three kinds of cells are left out to stay near 30 minutes:

- The mean coreset at m ≥ 1000. It runs Lloyd with m centers over the whole
  input, about 30 s CPU per build already at n = 1e5.
- The mean coreset at m = 4000 everywhere.
- Bicriteria sensitivity sampling at k = 50 on the 1e6-point and sparse inputs,
  10-20 s per build.

An earlier run with the original exclusions took 51 minutes on a loaded
machine. It also covers mean at m = 1000 on the 1e5-point inputs and
bicriteria at k = 50 on sparse data, and is kept in
[`2026-11-07/extended/`](2026-11-07/extended/summary.md). Its 740 runs that
overlap this one produced bit-identical coresets and distortions.

Datasets:

- `gauss-unbalanced-1m`: 1e6 × 10, 20 Gaussian clusters (σ = 3), Zipf(1.5)
  cluster sizes.
- `gauss-balanced-200k`: 2e5 × 20, 50 equal clusters (σ = 5).
- `esa22-k10-a5`: the ESA'22 benchmark instance with k = 10, α = 5
  (1e5 × 50).
- `sparse-bow-100k`: 1e5 documents over a 20,000-term vocabulary, 20 topics,
  about 38 nonzeros per document.

Headline distortion, offline (mean over 5 seeds of each run's maximum, lower is
better; "–" marks an excluded cell):

| dataset | k | m | uniform | sensitivity k-means++ | sensitivity bicriteria | mean |
|---|---|---|---|---|---|---|
| gauss-unbalanced-1m | 10 | 200 | 1.578 | 1.279 | 1.640 | 1.005 |
| gauss-unbalanced-1m | 10 | 1000 | 1.145 | 1.086 | 1.145 | – |
| gauss-unbalanced-1m | 10 | 4000 | 1.068 | 1.064 | 1.060 | – |
| gauss-unbalanced-1m | 50 | 200 | 4.867 | 4.164 | – | 1.178 |
| gauss-unbalanced-1m | 50 | 1000 | 1.196 | 1.281 | – | – |
| gauss-unbalanced-1m | 50 | 4000 | 1.069 | 1.089 | – | – |
| gauss-balanced-200k | 10 | 200 | 1.146 | 1.219 | 1.398 | 1.008 |
| gauss-balanced-200k | 10 | 1000 | 1.032 | 1.057 | 1.110 | – |
| gauss-balanced-200k | 10 | 4000 | 1.015 | 1.019 | 1.041 | – |
| gauss-balanced-200k | 50 | 200 | 2.178 | 3.019 | 7.611 | 1.015 |
| gauss-balanced-200k | 50 | 1000 | 1.188 | 1.196 | 1.667 | – |
| gauss-balanced-200k | 50 | 4000 | 1.050 | 1.054 | 1.151 | – |
| esa22-k10-a5 | 10 | 200 | 1.139 | 1.238 | 1.232 | 1.164 |
| esa22-k10-a5 | 10 | 1000 | 1.026 | 1.047 | 1.049 | – |
| esa22-k10-a5 | 10 | 4000 | 1.007 | 1.016 | 1.015 | – |
| esa22-k10-a5 | 50 | 200 | 1.712 | 2.215 | 2.409 | 1.250 |
| esa22-k10-a5 | 50 | 1000 | 1.159 | 1.245 | 1.290 | – |
| esa22-k10-a5 | 50 | 4000 | 1.042 | 1.070 | 1.094 | – |
| sparse-bow-100k | 10 | 200 | 1.104 | 1.184 | 2.032 | 1.013 |
| sparse-bow-100k | 10 | 1000 | 1.027 | 1.046 | 1.258 | – |
| sparse-bow-100k | 10 | 4000 | 1.010 | 1.023 | 1.075 | – |
| sparse-bow-100k | 50 | 200 | 1.645 | 2.766 | – | 1.026 |
| sparse-bow-100k | 50 | 1000 | 1.115 | 1.233 | – | – |
| sparse-bow-100k | 50 | 4000 | 1.033 | 1.067 | – | – |

What the numbers say:

- **Size matters more than the algorithm.** At m = 80k (k = 50, m = 4000),
  uniform and k-means++ sensitivity sampling stay at or below 1.09 on every
  dataset. At m = 4k (k = 50, m = 200) every sampling method fails, with
  distortions from 1.6 to 7.6. This is why ESA'22 uses coreset sizes of at
  least 50k.
- **Uniform sampling is a strong baseline on well-spread data.** On balanced
  clusters, the ESA'22 instance and the sparse documents, uniform sampling
  matches or beats k-means++ sensitivity sampling at every size. The likely
  reason is that importance weights add variance the data does not repay.
  Sensitivity sampling wins where cluster sizes are skewed. On
  `gauss-unbalanced-1m` at k = 10 it scores 1.28 against 1.58 at m = 200 and
  1.09 against 1.15 at m = 1000. At k = 50 it wins only at m = 200.
- **Bicriteria seeding loses to k-means++ seeding almost everywhere**, up to
  7.6 against 3.0, and it takes 4-14× more CPU to build. Its O(k log n) small
  clusters get large `8 w / |C|` sensitivities, so the sample concentrates and
  keeps fewer distinct points: 2,172 of m = 4000 on sparse data with k = 10,
  against 3,848 with k-means++.
- **The mean coreset (Barger-Feldman) is the most accurate per point on
  Gaussian and sparse data.** With 200 points it scores 1.005-1.026, except
  1.18 at k = 50 on the unbalanced mixture. In all 80 runs it never
  underestimated a cost (0 one-sided violations), and weight and second moment
  were exact up to rounding (≤ 3e-14). On the ESA'22 instance it loses
  clearly: its δ-sweep lower bound is 1.30 offline and 1.49 in stream mode,
  while the sampling methods stay at or below 1.13 at the same size. A likely
  cause is that averaging merges points that the instance's other clusterings
  keep apart. It also costs 40× (against sensitivity sampling) to 10⁴×
  (against uniform) more CPU to build: 8.2 s at n = 1e6, m = 200.
- **Candidates computed on the coreset find the worst case.** For the sampling
  methods, 78-89% of runs reach their maximum distortion on a Lloyd solution
  of the coreset (`lloyd@C`, `bestOf@C`), not on the ESA'22 full-data
  candidates. Without them the reported distortion would be 7% lower at the
  median and up to 13× lower in the worst case. For the mean coreset the
  maximum is always on a full-data candidate.
- **Stream mode degrades gently per level.** At k = 10, m = 1000, the level-0
  summary (one block) and the top level (all 64 blocks after 7 reductions)
  differ by at most 0.06 for the sampling methods. Per-level numbers use only
  the full-data candidates. The final stream coreset is worse than the offline
  one, for example 1.33 against 1.15 for uniform on `gauss-unbalanced-1m`.
  Part of the gap is size: a merged sample keeps fewer distinct points (about
  655 for uniform at m = 1000, against 994-1000 offline). The mean coreset
  degrades with depth only on the ESA'22 instance, from 1.12 to 1.33.
- **Confidence intervals hold on average, not per run.** Across the 210
  offline sensitivity runs, 94.5% of the 95% intervals contain the true cost.
  Within one run all candidates share one sample, so coverage is often all or
  nothing: 8 runs fall below 50%.
- **Validation costs more than construction for the fast algorithms.** Over
  the suite, construction took 857 s CPU, evaluation 167 s plus 177 s for the
  downstream solves, and the shared full-data candidates 147 s
  (`2026-11-07/shared.json`). Uniform builds a coreset of 1e6 points in 15 ms
  but takes about 50 ms to validate. In distance evaluations, validation
  (2.9e10) costs as much as all constructions together (2.7e10, most of it the
  mean coreset's Lloyd). Each run reads the full data 1.04 times on average.
  The first stream run per (dataset, k) pays a second pass for the per-level
  prefix costs.
- **Downstream cost tracks distortion.** At m = 4000, the best-of-10 solution
  on the coreset is within 4% of the best-of-3 solution on the full data
  (bicriteria on sparse data aside, at 13%), and is often better (negative
  `rel. cost`). The full-data reference is itself not optimal: at k = 50 on 50
  balanced clusters, solving on 200 means beats it by 40%.

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
descriptor.

## Caveats that apply to every run

- Timings come from one JVM run per suite with no warmup discarded. Barrett et
  al. (OOPSLA'17) show steady state is not guaranteed, so all iterations are
  reported. For per-kernel performance use JMH.
- Wall time is only meaningful on an idle machine. Each report records the
  one-minute load average. When it exceeds the core count, compare CPU times,
  allocation and distance evaluations instead.
- The first-moment relative error is ill-conditioned when the data mean is
  near the origin. The ESA'22 instance is centered, so its value there is
  meaningless.
