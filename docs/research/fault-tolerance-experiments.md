# Fault-injection experiments: coresets under node loss, packet erasure, stragglers and deep trees

This note reports a first empirical test of the fault-tolerance theory for k-means coresets: random and adversarial
node loss (Theorems A0, A1, A1', A2, A3), packet erasure of coreset points (B1, B2, B4), coded redundancy against
stragglers (C1-C3), error composition in merge-and-reduce (E1) and aggregation trees (F1, F2). The theory, the fault
models and the numeric predictions come from the research note's sections 3 and 5 (an internal working document);
every prediction of its section 5.5 for the models run here gets a verdict below. A falsified prediction is reported
as a result.

Experiments F1-F6 ran; F7 (Byzantine nodes) did not (see [Deviations](#deviations-from-the-specification)).

## Verdicts

35 checks: 24 PASS, 8 FAIL, 3 split verdicts (one part passes, one fails or is inconclusive). The measured column
quotes the run; full numbers are in [`data/fault-injection/summary.md`](data/fault-injection/summary.md).

| id | prediction (research section 5.5) | measured | verdict |
|---|---|---|---|
| F1/A1 shape | random partition: mean \|r(Q)\| = c sqrt(s_max phi)/(1-phi), linear in sqrt(phi)/(1-phi) | mean \|r\| ratio phi=1/2 : 1/8 = 2.59; linear-in-sqrt(phi)/(1-phi) predicts 3.50, the exact without-replacement law sqrt(phi kappa_pt/(1-phi)) predicts 2.65; per-candidate RMS / sqrt(phi kappa_pt/(1-phi)) = 0.99-1.02 | **FAIL** (the shape; the exact law fits) |
| F1/A1 N-independence | c independent of N | c = 0.248, 0.326, 0.316 at N = 100k, 200k, 400k: N s_max/S = 17.3, 8.6, 8.5, because s_max is the max of a bound and is set by single far points; with kappa_pt the calibration is 0.991, 1.001, 1.017 | **FAIL** (with s_max) |
| F1/A0 equal parts | none / ht-prior / ht-realized give identical downstream ratios on equal parts | max \|difference\| 8.4e-14 | **PASS** |
| F1/A0 zipf-sizes | the decoders differ on zipf-sizes parts, where `none` is biased | downstream identical (max diff 1.0e-13: one global factor cannot move the argmin); the cost estimate differs: mean \|r\| none 0.096, ht-prior 0.093, ht-size 0.001 at f=0.1 | **FAIL** (downstream) / **PASS** (cost estimate) |
| F1/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (zipf, random partition) | sensitivity m=500/node: 1.0022 (p90 1.0025, max 1.0027) vs 1.0019 without loss | **PASS** |
| F1/A1' cluster-sorted | downstream >> 2 whenever a lost node held a whole cluster, with that frequency | P[downstream > 2] = 0.73 vs P[a lost node held >= 90% of a cluster] = 0.76; P[> 2 \| whole cluster lost] = 0.96, P[> 2 \| not] = 0.00 (450 runs) | **PASS** |
| F1/A2 ESA'22 | tau = 1e-3 brings ESA'22 distortion down to random-partition levels | input-order partition, phi=1/8: median distortion 1.0091 without, 1.0088 with tau=1e-3, vs 1.0004 on a random partition | **FAIL** |
| F1/A2 storage | extra stored points <= (S/tau) log_{1/f}(1/tau), flat in N (tau=1e-3, f=0.1) | 25,295 / 24,288 / 16,029 extra copies at N = 100k / 200k / 400k; bound 276,000 | **PASS** |
| F2/A3 loss rate | coreset loss rate = f^(r+1) | 20/20 cells within 20% (e.g. f=0.3: 0.306, 0.091, 0.026 for r = 0, 1, 2) | **PASS** |
| F2/A3 error | error = A1 at phi_eff | measured / predicted mean-square perturbation = 1.00 | **PASS** |
| F3/B1 slope | split + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1, x1.41 at rho=0.5 | zipf m=1000: 0.563 (95% CI 0.463-0.655), x1.458 | **PASS** |
| F3/C3 rateless link | rateless with one ACK: distortion independent of rho, ~m/(1-rho) transmissions | slope 0.114 (CI -0.022-0.251); 2000 packets for m=1000 at rho=0.5 | **PASS** |
| F3/B1 bursts | GE bursts at the same mean rho, random packing: same mean, wider spread | mean excess GE/Bernoulli = 1.022; sd(m') 79 vs 16, but sd(distortion) 0.049 vs 0.054 | **PASS** (mean, m' spread) / **FAIL** (distortion spread) |
| F3/B1 sorted+bursts | sorted packing + bursts: large outliers | p99 excess 6.99 sorted vs 0.25 random packing (x27.9), same packets and loss pattern | **PASS** |
| F3/B2 plateau | merged packets: the excess plateaus at ~sqrt(rho/(1-rho) kappa_pt) | zipf-outliers, rho=0.3: m 2000->8000 shrinks merged x0.76, split x0.55; merged 0.084 vs floor 0.077 (full B2 formula 0.086) | **PASS** |
| F3/B2 balanced | balanced Zipf at large N: merged ~ split | merged/split = 0.979 | **PASS** |
| F3/B4 global offset | mean coreset, global offset: heavy tail, P[distortion > 2] = 1-(1-rho)^{#far cells} | no single cell is "far" (each loss alone keeps distortion < 2), so the formula predicts 0; measured 0-0.01 for rho <= 0.3 but 0.88 (zipf) and 0.58 (zipf-heavy) at rho=0.5, from aggregate mass loss | **FAIL** |
| F3/B4 CF triples | CF triples fix the offset bias but not the missing mass | mean excess cf / global without rescaling = 0.30; cf / global with realized-mass rescaling = 1.001; p99 excess cf / split = 4.3 | **PASS** (missing mass remains) / **INCONCLUSIVE** (offset bias) |
| F3/B4 split | split centroids: distortion ~ 1 + O(sqrt(rho eta)) | worst p99 excess / sqrt(rho eta/(1-rho)) = 0.41 (eta = 0.1) | **PASS** |
| F3/B4 replicate | replicated centroids remove the heavy tail | worst P[distortion > 2] = 0 | **PASS** |
| F3/B4 FEC | FEC restores exactness at overhead ~ rho | >= 98.0% of runs equal to no loss (delta = 1%); overhead 0.18 / 0.56 / 1.25 at rho = 0.1 / 0.3 / 0.5 | **PASS** |
| F4/C2 FR bytes | FR received bytes ~ (n/(s+1)) m, below uncoded n m | received/uncoded 0.514, 0.325, 0.261 vs g/n 0.500, 0.313, 0.250 (s = 1, 2, 3) | **PASS** |
| F4/C2 FR distortion | FR distortion equals offline at the group's data size | FR vs a fresh g-part partition: 1.0537/1.0444, 1.0597/1.0590, 1.0678/1.0766 (all within 2.5 se); no group lost under any <= s stragglers | **PASS** |
| F4/C3 rateless bytes | per-node bytes proportional to 1/mu | 99,110 / 50,130 / 25,202 bytes (x0.51, x0.25) | **PASS** |
| F4/C3 degradation | beyond s stragglers, rateless degrades following B1 | attacked-group mean-square error x1.13 (B1: x1.17), x2.22 (x2.00), x1.26 (x1.06), ...; cluster-sorted, s+1 stragglers in one group: downstream FR 9.69 vs rateless mu=2 1.011 | **PASS** |
| F4/A1 uncoded random | uncoded + HT under random stragglers follows A1 | measured / predicted mean-square = 1.05 | **PASS** |
| F4/A1' adversarial | uncoded + HT fails under the adversarial straggler | cluster-sorted s=1: uncoded downstream 4.68 (max 7.56); FR identical to no straggler (max diff 0.0) | **PASS** |
| F5/E1 random order | Var r(Q) flat in L beyond ~3 levels | slope of log Var vs log(L+1), L >= 3: -0.08 +/- 0.21 (m=500), -0.08 +/- 0.13, -0.07 +/- 0.17 | **PASS** |
| F5/E1 cluster-sorted | Var grows linearly in L+1 | slope 0.18 +/- 0.09 (m=500); the E1 bound itself grows with slope 0.15 on this order | **FAIL** |
| F5/E1 coefficient | fitted coefficient of Var on sum_l kappa_l = S/m | Var = 0.022 (S/m) sum_l kappa_l, R^2 0.995 | **FAIL** (the bound holds with 45x slack) |
| F5/E1(iv) per-level erasure | rho_l on every transfer multiplies level terms by 1/(1-rho_l) | Var ratio 1.23 vs 1/(1-0.1) = 1.11 (within the 25% tolerance) | **PASS** |
| F6/F1 lost fraction | lost fraction = 1-(1-f)^L | 11/12 cells within 3 se (e.g. 0.471 vs 0.469 at L=6, f=0.1); flat tree f=0.01: 0.0085 vs 0.0100, z = -4.3 | **FAIL** (one cell) |
| F6/F1 sqrt(L) | error proportional to sqrt(L) | slope 0.73 +/- 0.21; the F1 formula sqrt(L f/(1-f) kappa) matches a = 4, 8, 64 (ratio 0.92-1.08) but the binary tree is 1.39x above it | **FAIL** |
| F6/F2 checkpoint | checkpoint r=1: loss = L f^2 | 0.0593 / 0.0304 / 0.0182 / 0.0102 vs 0.060 / 0.030 / 0.020 / 0.010 | **PASS** |
| F6/F1 projection-sorted | projection-sorted trees without checkpointing fail catastrophically | p99 distortion 6.13 vs 1.027 on a random partition; with r=1 checkpointing 1.62 | **PASS** |

## Main findings

1. **Random node loss costs the exact without-replacement variance, not the A1 shape.** For a uniformly random
   partition the relative error of every candidate is sqrt(phi kappa_pt(Q)/(1-phi)) to within 1-3% (calibration 0.95-1.03
   over N in {100k, 200k, 400k}, n in {8, 16, 64}, phi in {1/8, 1/4, 1/2}). A1's bound sqrt(s_max phi)/(1-phi) is an
   upper bound (c ~ 0.3), but its phi-dependence is wrong by a factor 1/sqrt(1-phi), and s_max, a maximum over a
   sensitivity bound, is a noisy proxy that breaks the N-scaling. Replace s_max by kappa_pt(Q) <= s_max in A1.
2. **HT decoding never changes the clustering, only the cost estimate (A0 is exact).** All decoders apply one global
   factor, so the downstream solution is the same to 1e-13. The prior factor 1/(1-f) adds scale noise exactly as
   predicted (random parts, f=0.3: mean |r| 0.127 vs sqrt(f/((1-f)n)) sqrt(2/pi) = 0.131), and only the size-aware
   realized decoder (N / surviving size) is unbiased on unequal parts.
3. **Adversarial placement is the only catastrophic node fault.** Cluster-sorted loss gives downstream ratios of 3-17
   whenever a lost node held >= 90% of a cluster (96% of such runs exceed 2, none of the others); an adversarial
   straggler under uncoded HT gives 4.7. Coreset replication (A3) and fractional repetition remove it: loss rates
   follow f^(r+1) cell by cell and FR results are bit-identical with and without s stragglers.
4. **Sensitivity-aware replication (A2) does not detect partition adversaries.** Its storage claim holds (extra
   copies far below the bound and decreasing in N) and it halves the loss error on random partitions, but on
   cluster-sorted or input-order partitions only tau = 1e-4, which replicates every point, helps: the k-means++
   sensitivity bound flags far points and tiny clusters, not "lives on one node". On ESA'22 all sensitivities sit
   near S/N, so tau = 1e-3 cannot help there.
5. **Erasure invariance (B1) holds quantitatively.** The excess distortion grows as (1-rho)^{-0.5} (slopes 0.48-0.58
   over three instances, two sample sizes and two packetizations), independent of burstiness when packets are
   randomly ordered; rateless sending with one ACK makes the error rho-free at m/(1-rho) transmissions.
   Content-dependent packing is the failure mode: sorted packets under bursts reach distortion 8.5.
6. **The merged-duplicates floor (B2) is real but needs heavy points to show.** With 50 far outliers in 200k points
   the merged-packet erasure error stops shrinking at the predicted sqrt(rho/(1-rho) kappa_pt) and the full B2
   formula matches every m within 25% (within 12% at rho = 0.3); on the Zipf and ESA'22 instances kappa_pt <= 7.5e-5 puts the floor below the
   sampling term for all m <= 8000, so merged ~ split there.
7. **The mean coreset's fragility is lost mass, not the offset.** Rescaling the surviving cell counts by N / received
   mass cuts the excess distortion by 70% and CF triples add nothing beyond it; what remains (p99 1.38 at rho=0.3) is
   missing support, removed by splitting cells (eta = 0.1, 5.1x messages) or replicating them (4x at rho = 0.3), or exactly by FEC
   at 0.56x overhead.
8. **Merge-and-reduce error is depth-free under random order (E1), and tracks sum_l kappa_l on any order.** Variance
   doubles from offline to 3 levels, then stays flat to 8 levels; across all orders, m and depths it equals
   0.022 (S/m) sum_l kappa_l (R^2 0.995). The predicted sqrt(L) growth for cluster-sorted streams did not appear
   because no candidate's cost sits in one subtree at every level on this instance (sum_l kappa_l itself grows
   only with slope 0.15), and the S/m coefficient is 45x loose.

## Method

### Fault layer (package `io.github.c0rwin.coreset.testbed.faults`)

- **Instances** are prepared once: the ESA'22 candidate pool on the full data (10 k-means++ seedings, 10 random
  convex combinations of bicriteria centers) plus `bestOf@P` (best of 5 Lloyd runs, 30 iterations), which is both the
  downstream reference and the cluster labelling used by the cluster-sorted partitions; a per-row cost table
  (cost of every point on every candidate, one fused parallel pass); kappa_pt(Q) = sum_p (c_p/cost(P,Q))^2 per
  candidate; and the k-means++ sensitivity bounds s_p (standard bound, alpha = 2) for s_max/S, lost sensitivity
  mass and A2 placement.
- **Scoring without distance evaluations.** Sensitivity and uniform samples remember the global row of every point
  (`Sample` replays `SensitivityCoreset` / `UniformCoreset` call for call; a unit test checks identical points and
  weights for the same generator), so cost(C, Q) of any decoded sample under any fault realization is a sum over the
  cost table. Mean coresets and merge-reduce outputs are costed directly (their points are not input rows).
- **Partitions:** random (shuffle, equal parts), cluster-sorted (by `bestOf@P` cluster, cluster order shuffled),
  projection-sorted (random Gaussian direction), zipf-sizes (part sizes proportional to 1/(i+1)), input order.
- **Node loss:** fixed lost fraction phi (lost nodes chosen independently of the partition) or i.i.d. f; pre-compute
  (raw data lost) and post-compute (coreset lost unless one of r ring replicas survives).
- **Channels:** Bernoulli(rho) and Gilbert (two-state, all packets lost in the bad state) with mean burst 1, 5, 20
  at the same stationary rho. One uniform per packet, so Bernoulli patterns at different rho from the same stream
  are nested.
- **Packetizations** of a sample: per draw (split, random order), merged (point + count), 8 or 64 draws per packet
  after a random permutation, 8 draws per packet sorted by cluster; rateless (fresh i.i.d. draws until m arrive).
- **Decoders:** none, ht-prior 1/(1-p), ht-realized n/(n-l) or m/m', ht-size N / surviving size.
- **Protections:** split packets, coreset replication r, fractional-repetition groups (shared-seed coreset of the
  group union, first arrival used), rateless groups (shared proposal, m/mu independent draws per member, first mu
  answers pooled), CF triples / split cells / replicated cells / ideal systematic MDS FEC for the mean coreset,
  sensitivity-aware replication with threshold tau (copies 1 + ceil(log_{1/f}(s_p/tau)), realized HT per class).
- **Metrics per run** (JSONL): distortion (ESA'22 maximum of max(r+1, 1/(r+1)) over the 21 candidates), the full r(Q)
  vector for the cells used in fits, downstream ratio cost(P, Q^)/cost(P, bestOf@P) with Q^ the best of 10 Lloyd runs
  (20 iterations) on the decoded coreset, s_max/S, realized m', lost data fraction, lost sensitivity mass,
  bytes (protobuf size of each coreset message as the gRPC transport sends it; for F3 packets, 28 header bytes per
  packet plus 12 bytes per stored entry and 12 per point) and messages, wall time (F5).
- **Seeds:** data, partition, sampling, fault and solver streams are separate (`MergeRandomness.derive` of the base
  seed 42 and a purpose string). Defenses within a seed share the partition, the sample and the fault pattern; the
  level-1 (exact data) and level-2 (coreset) runs of F1 see the same lost nodes. Results do not depend on the thread
  count (a test compares 1 and 3 threads).
- **Statistics:** slopes by least squares with Student-t intervals, or percentile bootstrap over seeds (1000
  resamples) where the points share seeds (B1). Variance comparisons use paired differences r - r_no-fault, which
  remove the sampling noise common to both.

### Datasets and scale

| id | n | d | k | s_max/S | smallest cluster | role |
|---|---|---|---|---|---|---|
| zipf | 200,000 | 20 | 20 | 4.3e-5 | 1,030 | main instance (Gaussian mixture, 20 clusters, sigma 3, box 100, Zipf 1.5 sizes) |
| zipf-100k / zipf-400k | 100k / 400k | 20 | 20 | 1.7e-4 / 2.1e-5 | 515 / 2,060 | N-scaling of A1 and A2 storage |
| zipf-heavy | 200,000 | 20 | 20 | 4.9e-4 | 125 | Zipf 2.5 sizes: tiny clusters |
| zipf-outliers | 200,050 | 20 | 20 | 4.4e-2 | 1 | zipf plus 50 far points: the B2 stress case |
| esa22 | 100,000 | 50 | 10 | 2.0e-5 | 10,000 | ESA'22 benchmark instance (k=10, alpha=5) |

### Experiment grid (as run)

| id | grid | seeds |
|---|---|---|
| F1 raw | zipf: n in {8,16,64} x {random, cluster-sorted, projection-sorted, zipf-sizes} x phi in {1/8,1/4,1/2}, f in {0.05,0.1,0.2,0.3} x 4 decoders; zipf-100k/400k random n=16; zipf-heavy, esa22 (with input order) n=16 | 100 |
| F1 coreset | zipf, n=16, {random, cluster-sorted, zipf-sizes} x {sensitivity m=500, uniform m=500, mean m=50} per node x {none, phi 1/8,1/4,1/2, f 0.1,0.3} | 30 |
| F1 A2 | zipf-heavy, zipf, esa22 x {random, cluster-sorted, input} x tau in {none, 1e-2, 1e-3, 1e-4} x 5 losses | 100 |
| F2 | zipf, sensitivity m=500/node, n in {16, 64}, random (and cluster-sorted at 16), f in {0.05,0.1,0.2,0.3}, r in {0,1,2} | 30 x 20 fault draws |
| F3 sample | zipf (100 seeds), zipf-heavy, esa22 (60): m in {1000, 4000} x rho in {0,.01,.05,.1,.2,.3,.5} x 4 channels x 6 packetizations x decoders | 60-100 |
| F3 B2 | zipf, zipf-heavy, esa22, zipf-outliers: m in {250,...,8000} x rho in {0, .1, .3} x {per-draw, merged} | 100 |
| F3 B4 | zipf, zipf-heavy: 200 means (3 Lloyd iterations) x 6 rho x 6 variants | 10 x 20 |
| F4 | zipf, n=16, m=500, s in {1,2,3}, {random, cluster-sorted}, 6 straggler patterns, uncoded / replicate / FR / rateless mu in {1,2,4} | 30 |
| F5 | zipf, blocks in {1,...,256}, m in {500,1000,4000}, {random, cluster-sorted}, sensitivity and uniform; rho_l = 0.1 for sensitivity m=1000 | 200 |
| F6 | zipf, 64 leaves, arity in {2,4,8,64}, f in {0.01,0.05,0.1}, checkpoint r in {0,1}, internal HT on/off, {random, projection-sorted} | 1000 |

The full run took 18.8 minutes wall (74 CPU-minutes) on the 12-core M2 Max with 8 worker threads, peak RSS 4.3 GB;
the regeneration from the shaded jar below took 12.9 minutes on a less loaded machine.

## Selected results

### F1: random node loss (exact data, `ht-realized`, n = 16)

| N | phi | mean \|r\| | sqrt(s_max phi)/(1-phi) | ratio | RMS / sqrt(phi kappa_pt/(1-phi)) |
|---|---|---|---|---|---|
| 100k | 1/8, 1/4, 1/2 | 0.00174, 0.00250, 0.00434 | 0.0053, 0.0088, 0.0186 | 0.33, 0.28, 0.23 | 1.03, 0.97, 0.98 |
| 200k | 1/8, 1/4, 1/2 | 0.00110, 0.00165, 0.00284 | 0.0026, 0.0044, 0.0093 | 0.42, 0.38, 0.31 | 1.03, 0.99, 0.99 |
| 400k | 1/8, 1/4, 1/2 | 0.00073, 0.00109, 0.00197 | 0.0019, 0.0031, 0.0065 | 0.39, 0.35, 0.30 | 1.01, 1.01, 1.03 |

The ratio to the A1 shape falls with phi on every N; the calibration against the exact law does not move with N, n
or phi.

### F1: coreset pipeline (zipf, n = 16, sensitivity m = 500 per node, `ht-size`)

| partition | loss | distortion mean | downstream mean | p90 | max | P[downstream > 2] |
|---|---|---|---|---|---|---|
| random | none | 1.030 | 1.0019 | 1.0023 | 1.002 | 0 |
| random | phi = 1/8 | 1.033 | 1.0022 | 1.0025 | 1.003 | 0 |
| random | phi = 1/2 | 1.047 | 1.0039 | 1.0046 | 1.005 | 0 |
| cluster-sorted | phi = 1/8 | 1.253 | 3.09 | 7.00 | 11.0 | 0.50 |
| cluster-sorted | phi = 1/2 | 1.946 | 16.96 | 31.2 | 36.3 | 1.00 |
| zipf-sizes | phi = 1/2 | 1.069 | 1.0075 | 1.0100 | 1.012 | 0 |

### F1/A2: sensitivity-aware replication (exact data, n = 16, median / p90 distortion)

| instance | partition | loss | tau = none | 1e-3 | 1e-4 | extra copies at 1e-3 / 1e-4 |
|---|---|---|---|---|---|---|
| zipf-heavy | random | phi = 1/2 | 1.0119 / 1.0170 | 1.0061 / 1.0107 | 1.0011 / 1.0017 | 19,583 / 403,680 |
| zipf-heavy | cluster-sorted | phi = 1/2 | 1.670 / 2.680 | 1.546 / 2.252 | 1.014 / 1.029 | 19,583 / 403,680 |
| esa22 | input order | phi = 1/8 | 1.0091 / 1.0126 | 1.0088 / 1.0120 | 1.0005 / 1.0006 | 4,355 / 119,192 |
| esa22 | random | phi = 1/8 | 1.0004 / 1.0006 | 1.0004 / 1.0006 | 1.0001 / 1.0001 | 4,355 / 119,192 |

### F2: coreset replication (zipf, n = 16, random partition)

| f | r | coreset loss rate | f^(r+1) | mean-square perturbation | A1 at phi_eff | bytes/node |
|---|---|---|---|---|---|---|
| 0.1 | 0 / 1 / 2 | 0.098 / 0.0094 / 0.0010 | 0.1 / 0.01 / 0.001 | 3.3e-5 / 2.8e-6 / 1.8e-7 | 3.2e-5 / 2.7e-6 / 2.8e-7 | 86k / 190k / 286k |
| 0.3 | 0 / 1 / 2 | 0.306 / 0.091 / 0.026 | 0.3 / 0.09 / 0.027 | 1.4e-4 / 3.3e-5 / 8.1e-6 | 1.4e-4 / 3.1e-5 / 8.1e-6 | 66k / 182k / 283k |

### F3/B1: mean distortion by loss rate (zipf, m = 1000, Bernoulli, `ht-realized`)

| rho | per-draw | merged | random-b64 | sorted-b8 | rateless |
|---|---|---|---|---|---|
| 0 | 1.0950 | 1.0950 | 1.0950 | 1.0950 | 1.0950 |
| 0.1 | 1.1015 | 1.0998 | 1.1004 | 1.1345 | 1.0923 |
| 0.3 | 1.1195 | 1.1131 | 1.1167 | 1.2061 | 1.1012 |
| 0.5 | 1.1385 | 1.1428 | 1.1339 | 1.3177 | 1.1016 |

At rho = 0.3: decoder none 1.562, ht-prior 1.130, ht-realized 1.1195. Sorted packing hurts even under Bernoulli loss
(1.206 vs 1.120), since one packet then carries 8 draws of one cluster. With bursts of mean length 20, random packing
of 64 draws per packet left a few runs with nothing received (infinite distortion); per-draw and merged packets did
not.

### F3/B2: erasure error of merged versus split packets (rho = 0.3, `ht-prior`, candidate with the largest kappa_pt)

| instance | kappa_pt | m | split RMS (predicted) | merged RMS (predicted) | floor |
|---|---|---|---|---|---|
| zipf-outliers | 0.0139 | 1000 | 0.099 (0.104) | 0.145 (0.129) | 0.077 |
| zipf-outliers | 0.0139 | 4000 | 0.049 (0.052) | 0.095 (0.093) | 0.077 |
| zipf-outliers | 0.0139 | 8000 | 0.037 (0.037) | 0.084 (0.086) | 0.077 |
| zipf-heavy | 6.7e-5 | 8000 | 0.012 (0.015) | 0.012 (0.016) | 0.005 |
| esa22 | 1.0e-5 | 8000 | 0.008 (0.008) | 0.009 (0.008) | 0.002 |

### F3/B4: mean coreset, zipf, 200 cells (no-loss distortion 1.025)

| rho | global, no rescale | global, realized mass | CF triples | split (eta 0.1) | replicate | FEC |
|---|---|---|---|---|---|---|
| 0.1 | 1.182 | 1.099 | 1.099 | 1.036 | 1.031 | 1.026 |
| 0.3 | 1.544 | 1.183 | 1.183 | 1.059 | 1.030 | 1.029 |
| 0.5 | 2.221 | 1.286 | 1.286 | 1.097 | 1.031 | 1.027 |
| messages / cells - 1 | 0 | 0 | 0 | 4.1 | 1 / 3 / 6 | 0.18 / 0.56 / 1.25 |

The mean coreset's downstream ratio is below 1 on zipf-heavy (0.94-0.96): Lloyd with 10 restarts on 200 weighted means
finds better solutions than the 5-restart full-data reference.

### F4: stragglers (n = 16, m = 500, s = 1; downstream mean)

| partition | stragglers | uncoded | replicate | FR | rateless mu=2 |
|---|---|---|---|---|---|
| random | none | 1.0019 | 1.0019 | 1.0038 | 1.0061 |
| random | adversarial s | 1.0020 | 1.0019 | 1.0038 | 1.0061 |
| cluster-sorted | random s | 1.877 | 1.0033 | 1.0200 | 1.0087 |
| cluster-sorted | adversarial s | 4.678 | 1.0033 | 1.0200 | 1.0087 |
| cluster-sorted | adversarial s+1 | 8.297 | 3.328 | 9.685 | 1.0106 |

Received points: uncoded 7,553, replicate 7,553 (from 2x the bytes), FR 3,886, rateless mu=2 2,454.

### F5: merge-reduce variance (zipf, sensitivity, m = 500)

| blocks | L | MS r (random) | E1 bound | MS r (cluster-sorted) | E1 bound |
|---|---|---|---|---|---|
| 1 | 0 | 0.00387 | 0.184 | 0.00421 | 0.184 |
| 8 | 3 | 0.00816 | 0.345 | 0.00791 | 0.384 |
| 32 | 5 | 0.00890 | 0.362 | 0.00832 | 0.417 |
| 256 | 8 | 0.00774 | 0.367 | 0.00892 | 0.432 |

### F6: aggregation trees (f = 0.05, internal HT, no checkpoint)

| partition | arity | L | lost fraction (1-(1-f)^L) | RMS r (runs with data) | p99 distortion |
|---|---|---|---|---|---|
| random | 2 | 6 | 0.269 (0.265) | 0.0038 | 1.027 |
| random | 64 | 1 | 0.051 (0.050) | 0.0010 | 1.006 |
| projection-sorted | 2 | 6 | 0.264 (0.265) | | 6.13 (1.62 with r = 1) |
| projection-sorted | 64 | 1 | 0.051 (0.050) | | 1.24 |

## Deviations from the specification

- **Scale.** N = 200k (100k-400k for N-scaling) instead of 1M; k fixed by instance (20, or 10 on ESA'22) instead of
  {10, 20, 50}; m in {500, 1000, 4000} on the main grids. Variance fits use 100-1000 seeds, the coreset pipelines 30.
- **Public datasets:** none was cached locally and downloads were not allowed, so covertype/census/pendigits are
  absent. The run never touches the network.
- **F7 (Byzantine) was not run**, and FEC is an ideal MDS code (any K of n received packets decode), not RaptorQ.
- **B2 stress instance.** The spec named ESA'22 and high k for the merged-packet floor; on ESA'22 kappa_pt ~ 1/N, so
  the floor is invisible at any feasible m. `zipf-outliers` (the zipf mixture plus 50 far points) was added for
  this check only.
- **A2 placement** uses the full-data k-means++ sensitivity bound as an oracle upper bound, as section 3.1 allows;
  replicas go to random other nodes; f = phi is used for the copy count under fixed-fraction loss. Survival is
  scored on exact data (level "raw"), not through node coresets.
- **A1 is tested on exact survivors** (epsilon_0 = 0), which isolates epsilon_1; the coreset pipeline is a separate
  level with its own verdicts. F6 is scored on exact data only.
- **Mean-coreset cells** use 200 means and 3 Lloyd iterations (the testbed's standard setting), and the F3/B4 "far
  cell" count is the number of cells whose loss alone pushes the global-offset distortion above 2.
- **Downstream solver:** best of 10 Lloyd runs. A first run with 3 restarts left local optima (downstream 1.07 with no
  loss at all) that masked the loss effect.
- **Verdict thresholds** were fixed in code before the final run: slopes within the stated +/- 0.1 (B1) or 0.3 (E1)
  or 0.15 (F6), ratios within 15-35% depending on the check, z <= 3 for loss fractions. Split verdicts are reported
  when a prediction has two parts.

## Caveats

- The F6 flat-tree outlier (z = -4.3) is a single seed stream: 20 other streams for the same cell give 582-691 lost
  leaves for 640 expected, against 541 here. It is still reported as a FAIL under the fixed z <= 3 rule.
- E1's S/m coefficient is a worst-case bound (S_v <= S = 92 for k = 20); the measured 0.022 S/m means the effective
  sum cost^2/q per reduce is about 2 cost^2. The structural part, variance proportional to sum_l kappa_l, is what
  the data supports.
- The bytes model of F3 packets is analytic; node coreset messages (F1, F2, F4) use the real protobuf size.

## Reproduce

```bash
JAVA_HOME=/path/to/jdk17 mvn -B -pl coreset-testbed -am package -DskipTests
java -Xmx24g -jar coreset-testbed/target/coreset-testbed.jar faults --out results/faults --scale full --threads 8
# quick check (tiny instances, about 5 seconds):
java -jar coreset-testbed/target/coreset-testbed.jar faults --out results/faults-quick --scale quick
```

`--experiments F1,F3` runs a subset and `--seed` changes the base seed (default 42). Each run writes
`<experiment>.jsonl.gz` (one line per run), `<experiment>-cells.csv` (mean, sd, p50, p90, max of every metric per
cell), `summary.md` (all tables and the verdicts) and `verdicts.csv`.

### Data

Committed in [`data/fault-injection/`](data/fault-injection/) (0.5 MB): the per-cell aggregates
`F1-cells.csv.gz` ... `F6-cells.csv.gz` (gzipped; `raw-SHA256SUMS` hashes the uncompressed CSVs),
`summary.md` (every table and the verdicts), `verdicts.csv`, `run.log`, the
SHA-256 list of the full output `raw-SHA256SUMS`, and a deterministic sample of the raw runs, `F<k>-seed0.jsonl.gz`:
every JSONL line with `"seed":0`, cut with
`gzcat F1.jsonl.gz | grep -E '"seed":0[,}]' | gzip -9n > F1-seed0.jsonl.gz`.

The full raw JSONL (21 MB gzipped) is kept outside git, in `~/.cache/coreset-testbed/faults-raw/full-seed42/`. It is
regenerated bit for bit from the head of branch `research/fault-injection` (checked with Corretto 17.0.5 on
macOS arm64; `Math.log`/`Math.pow` intrinsics on other JVMs or CPUs may change the last bits) by

```bash
JAVA_HOME=/path/to/jdk17 mvn -B -pl coreset-testbed -am package -DskipTests
java -Xmx24g -jar coreset-testbed/target/coreset-testbed.jar faults \
    --out ~/.cache/coreset-testbed/faults-raw/full-seed42 --scale full --seed 42 --threads 8
cd ~/.cache/coreset-testbed/faults-raw/full-seed42 && shasum -a 256 -c path/to/docs/research/data/fault-injection/raw-SHA256SUMS
```

`shasum -c` reports `F5.jsonl.gz` as failed by design (see below). All seeds derive from `--seed 42`, and the thread
count does not change any result. Two independent full runs
(8 threads each) gave byte-identical files for everything except `F5.jsonl.gz`, whose `wallMs` field is a wall-clock
measurement. Its content without that field has SHA-256
`85e58def3d2f01950c79279c213029384eb0ead0e7af06c7c445fedab04f7fa9`, computed over the lines re-serialized as compact
JSON with sorted keys:

```bash
gzcat F5.jsonl.gz | python3 -c "import sys,json,hashlib; h=hashlib.sha256()
for l in sys.stdin:
    d=json.loads(l); d.pop('wallMs',None); h.update((json.dumps(d,sort_keys=True,separators=(',',':'))+'\n').encode())
print(h.hexdigest())"
```

| file | SHA-256 |
|---|---|
| F1.jsonl.gz | `6d27cba48c4be133f572ab1f7c55b63076061d17f123390aacd6161a818fb802` |
| F2.jsonl.gz | `2b7e8daaa8f356e293008802ea883d3aee3c003090ed2252a096be0228525d66` |
| F3.jsonl.gz | `cd49e2b609324825e5de360c27bba1945ec25b3ad9c4537f4ffc2dae3696121f` |
| F4.jsonl.gz | `7d8b7115cb008b3ed8b034dfa85b4b17305187cf750661cf6637725429f89eba` |
| F5.jsonl.gz | `12a15c6ab6458c83ebb29b40bc2ab1f5c66fb5903fec8bf74e87133c548fa65d` (this run; `wallMs` differs per run) |
| F6.jsonl.gz | `47a8d9f5661f9bd3945aa1c50a597631b36c8b61daec9e40689eb67cf4080f60` |

## Real datasets

The same predictions, checked on four public datasets (z-scored variants from the testbed's binary cache, see
`docs/datasets.md`), for k in {10, 25} and m in {1000, 4000}. 90 checks: 67 PASS, 20 FAIL, 3 INCONCLUSIVE. Full
per-instance numbers are in [`data/fault-injection-real/summary.md`](data/fault-injection-real/summary.md).

### Instances

| instance | n | d | k | s_max/S | max kappa_pt | smallest cluster | used by |
|---|---|---|---|---|---|---|---|
| covertype | 581,012 | 54 | 10 / 25 | 4.8e-4 / 1.2e-2 | 2.2e-4 / 5.7e-4 | 6,575 / 474 | F1, F2, F3, F5 |
| census1990 | 2,458,285 | 68 | 10 / 25 | 1.2e-5 / 7.3e-6 | 1.0e-6 / 9.0e-7 | 105,494 / 3,711 | F1, F2, F3 |
| kddcup99 | 4,898,431 | 38 | 10 / 25 | 7.4e-3 / 3.6e-2 | 1.6e-2 / 5.1e-3 | 1 / 1 | F1, F2, F3 |
| higgs, 1M subsample | 1,000,000 | 28 | 10 / 25 | 9.0e-6 / 7.8e-6 | 1.6e-6 / 1.7e-6 | 21,955 / 5,150 | F1, F2, F3 |
| census1990 / kddcup99 / higgs, 500k subsamples | 500,000 | | 10 / 25 | | | | F5 |
| higgs, all rows | 11,000,000 | 28 | 10 | 1.8e-6 | 1.6e-7 | 240,020 | F3 (B1/B2 at rho = 0.3) |

Clusters and the cluster-sorted partition come from the reference solution (best of 10 Lloyd restarts, 30
iterations, each restart k-means++ seeded) on the data actually used. kddcup99 has clusters of a single point (its
duplicate-heavy records), which makes it the extreme-imbalance case.

### Grid

| part | setting | seeds |
|---|---|---|
| F1 exact data | n = 16, random and cluster-sorted partitions, phi in {1/8, 1/4, 1/2}, f in {0.1, 0.3}, decoders ht-realized and ht-size | 50 |
| F1 coreset pipeline | n = 16, sensitivity m = 500 per node, random and cluster-sorted, losses none, phi 1/8, phi 1/2, f 0.3 | 20 |
| F2 | n = 16, random, f in {0.05, 0.1, 0.2, 0.3}, r in {0, 1, 2} | 20 x 10 fault draws |
| F3 | m in {1000, 4000}, 7 loss rates, Bernoulli and GE (burst 20), per-draw and merged, rateless; B2 at rho in {0, 0.3} | 100 (B2: 30) |
| F5 | blocks in {1, 8, 16, 32, 64, 128}, m in {1000, 4000}, random and cluster-sorted order, sensitivity | 60 |

The run took 45.3 minutes wall (4.3 CPU-hours, 10 threads), peak RSS 25.4 GB.

### Verdicts

| check | covertype k10 | covertype k25 | census k10 | census k25 | kddcup k10 | kddcup k25 | higgs-1M k10 | higgs-1M k25 |
|---|---|---|---|---|---|---|---|---|
| A1 kappa law: calibration in 0.85-1.15 at phi = 1/8, 1/4, 1/2 | PASS 0.94-0.99 | PASS 0.88-1.12 | PASS 0.95-1.03 | PASS 0.90-0.99 | PASS 0.96-1.01 | PASS 0.88-1.05 | PASS 0.93-1.00 | PASS 0.94-1.04 |
| A1 s_max shape: ratio phi 1/2 : 1/8 = 3.50 +/- 10% | FAIL 2.70 | FAIL 1.90 | FAIL 2.76 | FAIL 2.76 | FAIL 2.63 | PASS 3.22 | FAIL 2.57 | FAIL 2.76 |
| A1 downstream < 1.02 at phi = 1/8 | PASS 1.006 | PASS 1.009 | PASS 0.992 | PASS 1.015 | PASS 1.017 | PASS 1.004 | PASS 1.005 | PASS 1.016 |
| A1' cluster-sorted: downstream > 2 when a whole cluster is lost | FAIL max 1.15 | FAIL max 1.38 | FAIL max 1.15 | FAIL max 1.41 | FAIL max 1.86 | FAIL 0.73 of whole-loss runs > 2, max 6.95 | FAIL max 1.12 | FAIL max 1.14 |
| F2/A3: loss rate f^(r+1), error = A1 at phi_eff | PASS 0.71 | PASS 0.85 | PASS 0.85 | PASS 0.97 | PASS 1.03 | PASS 0.97 | PASS 0.86 | PASS 1.11 |
| F3/B1 slope 0.5 +/- 0.1 (m 1000 / 4000) | PASS 0.43 / 0.50 | PASS 0.54 / 0.57 | PASS 0.43 / 0.49 | PASS 0.43 / 0.55 | PASS 0.59 / 0.57 | PASS 0.52 / 0.54 | PASS 0.42 / 0.43 | PASS 0.40 / 0.47 |
| F3 rateless: no dependence on rho | FAIL -0.19 | PASS -0.01 | PASS 0.04 | FAIL -0.16 | PASS 0.12 | PASS 0.03 | PASS 0.02 | PASS -0.02 |
| F3 bursts: GE mean excess within 15% of Bernoulli | PASS 1.05 | PASS 1.02 | PASS 1.05 | PASS 0.98 | PASS 0.96 | PASS 1.02 | PASS 1.04 | PASS 1.04 |
| F3/B2 formula: split and merged within 35% | PASS | PASS | PASS | PASS | PASS (merged 0.095 vs B2 0.088, floor 0.083) | PASS (merged 0.060 vs 0.056, floor 0.047) | PASS | PASS |
| F5/E1: random-order Var flat for L >= 3 (m 1000 / 4000) | INCONCL. 0.46 / -0.29 | PASS 0.05 / 0.14 | INCONCL. 0.34 / -0.22 | INCONCL. 0.30 / -0.42 | FAIL 0.54 [0.08, 1.04] / 0.14 | PASS 0.24 / 0.12 | FAIL 0.05 / 0.58 [0.05, 1.15] | PASS 0.12 / 0.03 |
| F5/E1: Var proportional to (S/m) sum_l kappa_l (R^2 >= 0.9) | PASS 0.034, R^2 0.96 | PASS 0.024, 0.98 | PASS 0.014, 0.96 | PASS 0.009, 0.97 | PASS 0.051, 0.96 | PASS 0.036, 0.95 | PASS 0.008, 0.98 | PASS 0.005, 0.99 |

All 11M HIGGS rows (k = 10, 30 seeds): the B2 formula holds (split 0.011 vs 0.012, merged 0.013 vs 0.012, **PASS**);
the excess growth from rho = 0 to 0.3 is x1.27-1.49 against the predicted x1.195 (**FAIL** at the +/- 0.1 tolerance).
A seed bootstrap gives standard errors of 0.07-0.11 for these ratios; three of the four 95% intervals contain 1.195,
per-draw at m = 1000 (1.29-1.70) does not.

The F3 slope and F5 flatness rules use a seed bootstrap: PASS when both point estimates are within tolerance, FAIL
when a 95% interval excludes the predicted value, INCONCLUSIVE otherwise. The other thresholds are those of the
synthetic run.

### What the real data adds

1. **The kappa_pt law is exact on real data too.** Per-candidate calibration is 0.88-1.12 on all eight instances,
   including kddcup99 with single-point clusters; the s_max shape fails on 7 of 8 (ratio 1.9-2.8, exact law 2.65),
   and its constant c ranges over 0.09-0.51 between datasets.
2. **Adversarial partitions are not catastrophic on real data.** Losing a node that holds >= 90% of a reference
   cluster leaves the downstream ratio below 1.9 on every instance except kddcup99 at k = 25, where it exceeds 2 in
   73% of such runs (max 6.95). The Delta^2/k lower bound of A1' needs well-separated clusters, which real
   clusterings rarely have; kddcup99's far, tiny clusters are the exception.
3. **Erasure invariance and the B2 formula transfer unchanged.** Slopes 0.40-0.59 on every instance, bursty loss with
   random packing costs nothing extra, and the merged-packet floor appears exactly where Theorem B2 puts it: on
   kddcup99 (kappa_pt 5e-3 to 1.6e-2) merged packets lose 2-3x more than split ones, elsewhere merged ~ split.
4. **Merge-and-reduce variance tracks sum_l kappa_l (R^2 0.95-0.99) at 0.005-0.051 (S/m).** Depth flatness under
   random order is supported on 3 instances, inconclusive on 3 (bootstrap intervals about +/- 0.5 even with 60
   seeds) and rejected on 2, where one of the two m values shows a slope of about 0.55.
5. **Two rateless cells show a small negative slope** (-0.19 and -0.16 with intervals excluding 0): distortion
   slightly lower at higher loss. All rho share the first draws of one sequence, so this reads as a fluctuation of
   the rho = 0 sample in those seeds rather than a mechanism; it is reported as FAIL.

### Deviations (real data)

- HIGGS is subsampled to 1M rows (seeded, file order kept) for F1, F2 and F3; one F3 B2 run uses all 11M rows.
- F5 runs on seeded 500k-row subsamples of census1990, kddcup99 and HIGGS (each a separate instance with its own
  reference solution); covertype runs F5 on all rows.
- F1 uses one coreset algorithm (sensitivity) and four loss settings in the pipeline; no downstream solves in F3.
- The reference and the k-means++ candidates run as parallel tasks with derived seeds (synthetic instances keep the
  sequential single-generator preparation, whose outputs are unchanged).
- The cluster-sorted partition sorts by the reference solution's labels (k-means++ seeding refined by Lloyd), not by
  raw k-means++ labels.

### Reproduce (real data)

The four z-scored binaries must be in the cache (`coreset-testbed datasets convert <name> --variant zscore`).

```bash
java -Xmx48g -jar coreset-testbed/target/coreset-testbed.jar faults --experiments REAL \
    --out ~/.cache/coreset-testbed/faults-raw/real-seed42 --scale full --seed 42 --threads 10
```

Committed in [`data/fault-injection-real/`](data/fault-injection-real/) (0.35 MB): `REAL-F*-cells.csv.gz`,
`summary.md`, `verdicts.csv`, `run.log`, `raw-SHA256SUMS` and the seed-0 samples `REAL-F*-seed0.jsonl.gz`. The raw
JSONL (11 MB) is in `~/.cache/coreset-testbed/faults-raw/real-seed42/`. Its SHA-256 values are in `raw-SHA256SUMS`;
`REAL-F5.jsonl.gz` carries `wallMs`, and its content without that field hashes to
`b9b641214a99d8fbf88a11f12f90e16b6eb476b7413e608aa9dfa9dbbee2fea0` (script above). The real pass ran once, so these
hashes are not cross-checked by a second run; the code paths are the seeded ones whose synthetic outputs reproduced
byte for byte.
