# Fault-injection results (scale full, seed 42)

## Verdicts

| id | prediction | measured | verdict |
|---|---|---|---|
| F1/A1 shape | random partition: mean\|r\| = c sqrt(s_max phi)/(1-phi), linear in sqrt(phi)/(1-phi) | mean\|r\| ratio phi 1/2 : 1/8 = 2.59 vs 3.50 (linear in sqrt(phi)/(1-phi)) and 2.65 (sqrt(phi/(1-phi))); per-candidate RMS / sqrt(phi kappa_pt/(1-phi)) = 0.99-1.02 | **FAIL** |
| F1/A1 N-independence | c independent of N (N in {1e5, 4e5}; scaled: 1e5, 2e5, 4e5) | c = 0.248, 0.326, 0.316 at N = 100k, 200k, 400k (max/min 1.316; s_max is the max of the k-means++ bound, N s_max/S = 17.3, 8.6, 8.5); with kappa_pt in place of s_max the calibration is 0.991, 1.001, 1.017 | **FAIL** |
| F1/A0 equal parts | none / ht-prior / ht-realized give identical downstream ratios on equal parts | max \|downstream(ht-size) - downstream(none)\| = 8.4e-14 (random parts) | **PASS** |
| F1/A0 zipf-sizes | decoders differ on zipf-sizes parts, where `none` is biased | downstream identical across decoders (max \|diff\| 1.0e-13: one global factor cannot move the argmin); cost estimate differs: mean\|r\| none 0.096, ht-prior 0.093, ht-size 0.001 at f=0.1 | **FAIL (downstream) / PASS (cost estimate)** |
| F1/A1 downstream | downstream ratio barely moves: < 1.02 at phi = 1/8 (zipf, random partition) | sensitivity m=500/node: mean 1.0022 (p90 1.0025, max 1.0027) vs 1.0019 without loss (loss adds +0.0003) | **PASS** |
| F1/A1' cluster-sorted | downstream >> 2 whenever a lost node held a whole cluster, frequency = fraction of such nodes | P[downstream > 2] = 0.73 overall vs P[a lost node held >= 90% of a cluster] = 0.76; P[> 2 \| whole cluster lost] = 0.96, P[> 2 \| not] = 0.00 (450 runs) | **PASS** |
| F1/A2 ESA'22 | tau = 1e-3 brings the ESA'22 distortion down to random-partition levels | input-order partition, phi=1/8: median distortion 1.0091 without replication, 1.0088 with tau=1e-3 (4355 of 100000 points above tau; mean s_p = S/N = 0.00052, so the bound flags no structure), vs 1.0004 on a random partition | **FAIL** |
| F1/A2 storage | extra stored points <= (S/tau) log_{1/f}(1/tau), flat in N (tau=1e-3, f=0.1) | extra copies N=100000: 25295; N=200000: 24288; N=400000: 16029;  bound 276000 | **PASS** |
| F2/A3 loss rate | observed coreset loss rate = f^(r+1) | 20/20 cells (random partition, >= 30 expected losses) within 20% of f^(r+1) | **PASS** |
| F2/A3 error | error = A1 at phi_eff (r = 0, random partition) | mean-square loss perturbation r(Q) - r_no-loss(Q) / phi/(1-phi) (kappa_pt + r0^2) = 1.00 (pooled over f and n) | **PASS** |
| F3/B1 slope | split + realized: log-log slope of (distortion-1) vs 1/(1-rho) = 0.5 +/- 0.1; x1.41 at rho=0.5 | zipf m=1000 per-draw: slope 0.563 (95% CI 0.463..0.655), excess ratio rho=0.5:0 = 1.458 | **PASS** |
| F3/C3 rateless link | rateless with one ACK: distortion independent of rho (m draws always arrive), at ~m/(1-rho) transmissions | slope 0.114 (CI -0.022..0.251); packets sent at rho=0.5: 2000 for m=1000 | **PASS** |
| F3/B1 bursts | GE at the same mean rho with random packing: same mean as Bernoulli, wider spread | rho=0.3, burst 20: mean excess GE/Bernoulli = 1.022; sd(distortion) 0.0486 vs 0.0538; sd(m') 79 vs 16 | **PASS (m' spread) / FAIL (distortion spread)** |
| F3/B1 sorted+bursts | sorted packing + bursts: large outliers (failure) | rho=0.3, burst 20, 8 draws/packet: p99 excess sorted 6.993 vs random packing 0.251 (x27.9); max 8.544 vs 1.429; total-loss runs (no packet arrived) 0 of 100 for both | **PASS** |
| F3/B2 plateau | merged packets: excess plateaus as m grows at ~sqrt(rho/(1-rho) kappa_pt) (high-s_max instance; ESA'22 has kappa_pt ~ 1/N and no plateau in range) | zipf-outliers rho=0.3, Q* = max kappa_pt: erasure RMS m=2000->8000 shrinks x0.76 merged vs x0.55 split (sqrt(m ratio) = 0.50); merged RMS at m=8000 0.0837 vs floor 0.0773 (full B2 formula 0.0855) | **PASS** |
| F3/B2 balanced | balanced Zipf instance at large N: merged ~ split | zipf rho=0.3 m=1000: merged/split erasure RMS for Q* = 0.979 | **PASS** |
| F3/B4 global offset | mean coreset, global offset: heavy-tailed; P[distortion > 2] = 1 - (1-rho)^{#far cells} | largest \|measured - predicted\| = 0.88 (zipf rho=0.01: 0.00 vs 0.00; zipf rho=0.05: 0.00 vs 0.00; zipf rho=0.1: 0.00 vs 0.00; zipf rho=0.2: 0.00 vs 0.00; zipf rho=0.3: 0.01 vs 0.00; zipf rho=0.5: 0.88 vs 0.00; zipf-heavy rho=0.01: 0.00 vs 0.00; zipf-heavy rho=0.05: 0.00 vs 0.00; zipf-heavy rho=0.1: 0.00 vs 0.00; zipf-heavy rho=0.2: 0.00 vs 0.00; zipf-heavy rho=0.3: 0.01 vs 0.00; zipf-heavy rho=0.5: 0.58 vs 0.00; ) | **FAIL** |
| F3/B4 CF triples | CF triples fix the offset bias but not the missing mass | zipf rho=0.3, mean excess distortion: cf / global offset without rescaling = 0.30, cf / global offset with realized-mass rescaling = 1.001; p99 excess cf / split = 4.3 | **PASS (missing mass remains) / INCONCLUSIVE (offset bias: realized-mass rescaling alone already matches CF)** |
| F3/B4 split | split centroids restore distortion ~ 1 + O(sqrt(rho eta)) | rho <= 0.3, eta = 0.1: worst (p99 distortion - no-loss) / sqrt(rho eta/(1-rho)) = 0.41 | **PASS** |
| F3/B4 replicate | replicated centroids (r = ceil(log delta / log rho), delta = 0.01) remove the heavy tail | worst P[distortion > 2] over rho = 0.000 | **PASS** |
| F3/B4 FEC | FEC restores exactness at overhead ~ rho | fraction of runs equal to no loss (to 1e-6) >= 0.980 (target 1 - delta = 0.99); overhead in the table | **PASS** |
| F4/C2 FR bytes | FR received bytes ~ (n/(s+1)) m, less than uncoded n m | s=1: FR 3886 pts vs uncoded 7553 (ratio 0.514, g/n = 0.500); s=2: FR 2453 pts vs uncoded 7553 (ratio 0.325, g/n = 0.313); s=3: FR 1971 pts vs uncoded 7553 (ratio 0.261, g/n = 0.250);  | **PASS** |
| F4/C2 FR distortion | FR distortion equals offline at the group's data size (s stragglers) | s=1: FR 1.0537 vs offline-g 1.0444 (diff 0.0093, 2se 0.0079); s=2: FR 1.0597 vs offline-g 1.0590 (diff 0.0007, 2se 0.0096); s=3: FR 1.0678 vs offline-g 1.0766 (diff -0.0088, 2se 0.0143); no FR group lost under any <= s straggler set | **PASS** |
| F4/C3 rateless bytes | rateless per-node bytes proportional to 1/mu | s=2, no stragglers: per-node bytes mu=1 99110, mu=2 50130 (x0.51), mu=4 25202 (x0.25) | **PASS** |
| F4/C3 degradation | rateless beyond s stragglers degrades gracefully following B1 (instead of losing the group) | s=1 mu=2: group MS x1.13 at m'/m=0.86 (B1: x1.17); s=1 mu=4: group MS x1.26 at m'/m=0.94 (B1: x1.06); s=2 mu=2: group MS x2.22 at m'/m=0.50 (B1: x2.00); s=2 mu=4: group MS x1.12 at m'/m=1.00 (B1: x1.00); s=3 mu=2: group MS x1.01 at m'/m=0.88 (B1: x1.13); s=3 mu=4: group MS x1.03 at m'/m=1.00 (B1: x1.00); cluster-sorted s=1, s+1 stragglers in the attacked group: downstream FR 9.685 vs rateless mu=2 1.0106 | **PASS** |
| F4/A1 uncoded random | uncoded + HT under random stragglers follows A1 | mean-square straggler perturbation of r(Q) / A1 prediction phi/(1-phi)(kappa_pt + r0^2) = 1.05 (pooled s, patterns) | **PASS** |
| F4/A1' adversarial straggler | uncoded + HT fails under the adversarial straggler (holder of the smallest cluster); coded schemes do not | cluster-sorted, s=1: uncoded downstream mean 4.678 (max 7.562); FR 1.0200, max \|FR(adversarial) - FR(no straggler)\| = 0.0 over paired seeds | **PASS** |
| F5/E1 random order | random order: Var r(Q) flat in L beyond ~3 levels (depth-free) | sensitivity m=500: slope of log Var vs log(L+1) = -0.08 (all: random m=500: -0.08 +/- 0.21 (sum kappa: 0.07); random m=1000: -0.08 +/- 0.13 (sum kappa: 0.07); random m=4000: -0.07 +/- 0.17 (sum kappa: 0.06); cluster-sorted m=500: 0.18 +/- 0.09 (sum kappa: 0.15); cluster-sorted m=1000: -0.02 +/- 0.31 (sum kappa: 0.14); cluster-sorted m=4000: 0.10 +/- 0.10 (sum kappa: 0.10); ) | **PASS** |
| F5/E1 cluster-sorted | cluster-sorted: Var grows linearly in L+1 (sqrt L in RMS) | sensitivity m=500: slope 0.18 (1 = linear in L+1); the E1 bound (S/m) sum_l kappa_l on this order grows with slope 0.15 | **FAIL** |
| F5/E1 coefficient | fitted coefficient of Var on sum_l kappa_l = S/m (c = 1 in units of S/m) | c = 0.022 (R^2 0.995): Var follows sum_l kappa_l but at 0.022 S/m; E1 is an upper bound with slack S_v <= S and sum cost^2/q <= S cost^2 | **FAIL (bound holds, not tight)** |
| F5/E1(iv) per-level erasure | per-level erasures multiply level terms by 1/(1-rho_l) (rho_l = 0.1 on every transfer, realized-count decoding) | random order, m=1000, 256 blocks: Var ratio lossy/clean = 1.228 (1/(1-rho) = 1.111) | **PASS** |
| F6/F1 lost fraction | lost fraction = 1-(1-f)^L | a=2 L=6: 0.4711 vs 0.4686; a=4 L=3: 0.2735 vs 0.2710; a=8 L=2: 0.1881 vs 0.1900; [a=64 f=0.01: 0.0085 vs 0.0100, z = -4.3] a=64 L=1: 0.0988 vs 0.1000; (all f within 3 run-level se: false) | **FAIL** |
| F6/F1 sqrt(L) | error proportional to sqrt(L) at fixed f | f=0.05, random, internal HT, runs with surviving data: slope of log RMS r vs log L = 0.73 +/- 0.21; a=2 L=6: RMS 0.003760 (F1 formula sqrt(L f/(1-f) kappa) 0.002698), total loss 0.0020; a=4 L=3: RMS 0.002069 (F1 formula sqrt(L f/(1-f) kappa) 0.001908), total loss 0.0000; a=8 L=2: RMS 0.001532 (F1 formula sqrt(L f/(1-f) kappa) 0.001558), total loss 0.0000; a=64 L=1: RMS 0.001011 (F1 formula sqrt(L f/(1-f) kappa) 0.001101), total loss 0.0000;  | **FAIL** |
| F6/F2 checkpoint | checkpoint r=1: loss = L f^2 | a=2: 0.0593 vs L f^2 = 0.0600; a=4: 0.0304 vs L f^2 = 0.0300; a=8: 0.0182 vs L f^2 = 0.0200; a=64: 0.0102 vs L f^2 = 0.0100;  | **PASS** |
| F6/F1 projection-sorted | projection-sorted trees without checkpointing fail catastrophically | a=2, f=0.05: p99 distortion 6.135 (projection) vs 1.0268 (random); max Infinity; P[> 2] = 0.092; with checkpoint r=1 p99 1.6249 | **PASS** |

## F1: node crash before compute (raw data lost)

### A1: random partition, fixed lost fraction, `ht-realized`

Exact survivors (epsilon_0 = 0). `mean|r|` averages |r(Q)| over seeds and the candidate pool; `calib` is the mean over candidates of RMS r_j / sqrt(phi kappa_pt(Q_j)/(1-phi)) (1 = the exact without-replacement variance).

| instance | nodes | N | s_max/S | phi | mean abs r | x1 = sqrt(s_max phi)/(1-phi) | mean abs r / x1 | calib |
|---|---|---|---|---|---|---|---|---|
| zipf-100k | 16 | 100000 | 0.000173 | 0.125 | 0.001740 | 0.005321 | 0.327 | 1.026 |
| zipf-100k | 16 | 100000 | 0.000173 | 0.25 | 0.002497 | 0.008779 | 0.284 | 0.972 |
| zipf-100k | 16 | 100000 | 0.000173 | 0.5 | 0.004337 | 0.01862 | 0.233 | 0.975 |
| zipf | 16 | 200000 | 4.28e-05 | 0.125 | 0.001101 | 0.002644 | 0.417 | 1.026 |
| zipf | 16 | 200000 | 4.28e-05 | 0.25 | 0.001650 | 0.004362 | 0.378 | 0.989 |
| zipf | 16 | 200000 | 4.28e-05 | 0.5 | 0.002840 | 0.009253 | 0.307 | 0.987 |
| zipf-400k | 16 | 400000 | 2.13e-05 | 0.125 | 0.0007297 | 0.001865 | 0.391 | 1.006 |
| zipf-400k | 16 | 400000 | 2.13e-05 | 0.25 | 0.001085 | 0.003077 | 0.353 | 1.012 |
| zipf-400k | 16 | 400000 | 2.13e-05 | 0.5 | 0.001972 | 0.006526 | 0.302 | 1.033 |
| zipf | 8 | 200000 | 4.28e-05 | 0.125 | 0.001089 | 0.002644 | 0.412 | 1.000 |
| zipf | 8 | 200000 | 4.28e-05 | 0.25 | 0.001715 | 0.004362 | 0.393 | 1.016 |
| zipf | 8 | 200000 | 4.28e-05 | 0.5 | 0.002981 | 0.009253 | 0.322 | 1.004 |
| zipf | 64 | 200000 | 4.28e-05 | 0.125 | 0.001032 | 0.002644 | 0.391 | 0.952 |
| zipf | 64 | 200000 | 4.28e-05 | 0.25 | 0.001667 | 0.004362 | 0.382 | 0.994 |
| zipf | 64 | 200000 | 4.28e-05 | 0.5 | 0.002873 | 0.009253 | 0.310 | 1.021 |

Shape: mean|r|(phi=1/2) / mean|r|(phi=1/8) = 2.592 (mean over N); linear in sqrt(phi)/(1-phi) predicts 3.500, the exact without-replacement law sqrt(phi/(1-phi)) predicts 2.646.

### A0: decoders (zipf, n = 16, raw level: cost-estimate error)

| partition | loss | none mean abs r | ht-prior | ht-realized | ht-size |
|---|---|---|---|---|---|
| random | phi=0.25 | 0.2501 | 0.0017 | 0.0017 | 0.0017 |
| random | f=0.1 | 0.0958 | 0.0800 | 0.0008 | 0.0008 |
| random | f=0.3 | 0.3018 | 0.1270 | 0.0018 | 0.0018 |
| zipf-sizes | phi=0.25 | 0.2610 | 0.1448 | 0.1448 | 0.0017 |
| zipf-sizes | f=0.1 | 0.0962 | 0.0930 | 0.0656 | 0.0008 |
| zipf-sizes | f=0.3 | 0.3149 | 0.2110 | 0.1669 | 0.0020 |

Downstream ratio with the survivors rescaled by `ht-size` minus unscaled (same solver seed): max |diff| = 8.4e-14 on random parts, 1.0e-13 on zipf-sizes parts.

### Coreset pipeline (zipf, n = 16, decoder ht-size)

| partition | algorithm | loss | distortion mean | p90 | downstream mean | p90 | max | P[downstream > 2] | P[whole cluster lost] |
|---|---|---|---|---|---|---|---|---|---|
| random | sensitivity | none | 1.0302 | 1.0399 | 1.0019 | 1.0023 | 1.002 | 0.00 | 0.00 |
| random | sensitivity | phi=0.125 | 1.0330 | 1.0478 | 1.0022 | 1.0025 | 1.003 | 0.00 | 0.00 |
| random | sensitivity | phi=0.25 | 1.0377 | 1.0526 | 1.0026 | 1.0030 | 1.003 | 0.00 | 0.00 |
| random | sensitivity | phi=0.5 | 1.0466 | 1.0608 | 1.0039 | 1.0046 | 1.005 | 0.00 | 0.00 |
| random | sensitivity | f=0.1 | 1.0310 | 1.0408 | 1.0020 | 1.0024 | 1.003 | 0.00 | 0.00 |
| random | sensitivity | f=0.3 | 1.0414 | 1.0491 | 1.0029 | 1.0037 | 1.005 | 0.00 | 0.00 |
| random | uniform | none | 1.0533 | 1.0854 | 1.0025 | 1.0027 | 1.003 | 0.00 | 0.00 |
| random | uniform | phi=0.125 | 1.0579 | 1.0939 | 1.0166 | 1.0032 | 1.416 | 0.00 | 0.00 |
| random | uniform | phi=0.25 | 1.0614 | 1.0970 | 1.0034 | 1.0037 | 1.004 | 0.00 | 0.00 |
| random | uniform | phi=0.5 | 1.0835 | 1.1425 | 1.0052 | 1.0056 | 1.006 | 0.00 | 0.00 |
| random | uniform | f=0.1 | 1.0549 | 1.0887 | 1.0027 | 1.0030 | 1.003 | 0.00 | 0.00 |
| random | uniform | f=0.3 | 1.0688 | 1.1061 | 1.0038 | 1.0045 | 1.007 | 0.00 | 0.00 |
| random | mean | none | 1.0310 | 1.0320 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| random | mean | phi=0.125 | 1.0307 | 1.0327 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| random | mean | phi=0.25 | 1.0319 | 1.0348 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| random | mean | phi=0.5 | 1.0342 | 1.0406 | 1.0001 | 1.0001 | 1.000 | 0.00 | 0.00 |
| random | mean | f=0.1 | 1.0309 | 1.0326 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| random | mean | f=0.3 | 1.0320 | 1.0360 | 1.0001 | 1.0001 | 1.000 | 0.00 | 0.00 |
| cluster-sorted | sensitivity | none | 1.0538 | 1.0899 | 1.0034 | 1.0038 | 1.004 | 0.00 | 0.00 |
| cluster-sorted | sensitivity | phi=0.125 | 1.2528 | 1.3476 | 3.0942 | 6.9952 | 11.039 | 0.50 | 0.57 |
| cluster-sorted | sensitivity | phi=0.25 | 1.4665 | 1.9275 | 6.9643 | 14.7443 | 17.577 | 0.87 | 0.90 |
| cluster-sorted | sensitivity | phi=0.5 | 1.9455 | 3.0594 | 16.9614 | 31.1828 | 36.251 | 1.00 | 1.00 |
| cluster-sorted | sensitivity | f=0.1 | 1.2247 | 1.4115 | 3.5974 | 7.8885 | 16.021 | 0.50 | 0.50 |
| cluster-sorted | sensitivity | f=0.3 | 1.4878 | 2.0150 | 8.1263 | 16.0055 | 26.717 | 0.80 | 0.83 |
| cluster-sorted | uniform | none | 1.0442 | 1.0624 | 1.0026 | 1.0028 | 1.003 | 0.00 | 0.00 |
| cluster-sorted | uniform | phi=0.125 | 1.2489 | 1.3444 | 3.0951 | 6.9817 | 11.044 | 0.50 | 0.57 |
| cluster-sorted | uniform | phi=0.25 | 1.4597 | 1.8185 | 6.7534 | 14.8149 | 17.502 | 0.83 | 0.90 |
| cluster-sorted | uniform | phi=0.5 | 1.9548 | 3.4380 | 16.9699 | 31.2258 | 36.255 | 1.00 | 1.00 |
| cluster-sorted | uniform | f=0.1 | 1.2170 | 1.3877 | 3.5669 | 7.9270 | 16.207 | 0.50 | 0.50 |
| cluster-sorted | uniform | f=0.3 | 1.4859 | 1.9528 | 8.1147 | 15.9975 | 26.560 | 0.80 | 0.83 |
| cluster-sorted | mean | none | 1.0259 | 1.0267 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| cluster-sorted | mean | phi=0.125 | 1.2395 | 1.3464 | 3.0984 | 7.0098 | 11.098 | 0.50 | 0.57 |
| cluster-sorted | mean | phi=0.25 | 1.4469 | 1.7878 | 6.7699 | 14.7867 | 17.534 | 0.83 | 0.90 |
| cluster-sorted | mean | phi=0.5 | 1.9372 | 3.2200 | 17.0083 | 31.2889 | 36.309 | 1.00 | 1.00 |
| cluster-sorted | mean | f=0.1 | 1.2021 | 1.3854 | 3.5672 | 7.9016 | 16.126 | 0.50 | 0.50 |
| cluster-sorted | mean | f=0.3 | 1.4718 | 1.9612 | 8.1330 | 16.0379 | 26.697 | 0.80 | 0.83 |
| zipf-sizes | sensitivity | none | 1.0480 | 1.0655 | 1.0039 | 1.0044 | 1.005 | 0.00 | 0.00 |
| zipf-sizes | sensitivity | phi=0.125 | 1.0530 | 1.0710 | 1.0045 | 1.0051 | 1.007 | 0.00 | 0.00 |
| zipf-sizes | sensitivity | phi=0.25 | 1.0523 | 1.0764 | 1.0048 | 1.0059 | 1.007 | 0.00 | 0.00 |
| zipf-sizes | sensitivity | phi=0.5 | 1.0689 | 1.0943 | 1.0075 | 1.0100 | 1.012 | 0.00 | 0.00 |
| zipf-sizes | sensitivity | f=0.1 | 1.0518 | 1.0690 | 1.0042 | 1.0054 | 1.006 | 0.00 | 0.00 |
| zipf-sizes | sensitivity | f=0.3 | 1.0541 | 1.0788 | 1.0054 | 1.0070 | 1.011 | 0.00 | 0.00 |
| zipf-sizes | uniform | none | 1.0727 | 1.1155 | 1.0055 | 1.0058 | 1.006 | 0.00 | 0.00 |
| zipf-sizes | uniform | phi=0.125 | 1.0785 | 1.1205 | 1.0062 | 1.0072 | 1.008 | 0.00 | 0.00 |
| zipf-sizes | uniform | phi=0.25 | 1.0802 | 1.1160 | 1.0068 | 1.0081 | 1.010 | 0.00 | 0.00 |
| zipf-sizes | uniform | phi=0.5 | 1.1048 | 1.1360 | 1.0105 | 1.0130 | 1.015 | 0.00 | 0.00 |
| zipf-sizes | uniform | f=0.1 | 1.0731 | 1.0969 | 1.0060 | 1.0071 | 1.008 | 0.00 | 0.00 |
| zipf-sizes | uniform | f=0.3 | 1.0766 | 1.1194 | 1.0209 | 1.0108 | 1.419 | 0.00 | 0.00 |
| zipf-sizes | mean | none | 1.0310 | 1.0329 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| zipf-sizes | mean | phi=0.125 | 1.0315 | 1.0342 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| zipf-sizes | mean | phi=0.25 | 1.0326 | 1.0373 | 1.0000 | 1.0001 | 1.000 | 0.00 | 0.00 |
| zipf-sizes | mean | phi=0.5 | 1.0333 | 1.0372 | 1.0001 | 1.0002 | 1.000 | 0.00 | 0.00 |
| zipf-sizes | mean | f=0.1 | 1.0315 | 1.0354 | 1.0000 | 1.0000 | 1.000 | 0.00 | 0.00 |
| zipf-sizes | mean | f=0.3 | 1.0343 | 1.0384 | 1.0001 | 1.0001 | 1.000 | 0.00 | 0.00 |

Cluster-sorted (A1'): P[downstream > 2] = 0.73 overall vs P[a lost node held >= 90% of a cluster] = 0.76; P[> 2 | whole cluster lost] = 0.96, P[> 2 | not] = 0.00 (450 runs)

### A2: sensitivity-aware replication (raw level, realized per-class HT)

| instance | partition | loss | tau | distortion p50 | p90 | max | heavy points | extra copies | bound (S/tau) log_{1/f}(1/tau) | lost fraction |
|---|---|---|---|---|---|---|---|---|---|---|
| zipf-heavy | random | phi=0.125 | none | 1.0045 | 1.0067 | 1.01 | 0 | 0 | 0.00 | 0.1250 |
| zipf-heavy | random | phi=0.25 | none | 1.0067 | 1.0111 | 1.02 | 0 | 0 | 0.00 | 0.2500 |
| zipf-heavy | random | phi=0.5 | none | 1.0119 | 1.0170 | 1.02 | 0 | 0 | 0.00 | 0.5000 |
| zipf-heavy | random | f=0.1 | none | 1.0030 | 1.0056 | 1.01 | 0 | 0 | 0.00 | 0.0825 |
| zipf-heavy | random | f=0.3 | none | 1.0076 | 1.0124 | 1.02 | 0 | 0 | 0.00 | 0.3125 |
| zipf-heavy | random | phi=0.125 | 1e-02 | 1.0041 | 1.0061 | 1.01 | 709 | 709 | 2.04e+04 | 0.1246 |
| zipf-heavy | random | phi=0.25 | 1e-02 | 1.0061 | 1.0103 | 1.02 | 709 | 798 | 3.06e+04 | 0.2493 |
| zipf-heavy | random | phi=0.5 | 1e-02 | 1.0109 | 1.0159 | 1.02 | 709 | 1012 | 6.11e+04 | 0.4989 |
| zipf-heavy | random | f=0.1 | 1e-02 | 1.0030 | 1.0048 | 1.01 | 709 | 709 | 1.84e+04 | 0.0822 |
| zipf-heavy | random | f=0.3 | 1e-02 | 1.0066 | 1.0119 | 1.02 | 709 | 798 | 3.52e+04 | 0.3117 |
| zipf-heavy | random | phi=0.125 | 1e-03 | 1.0020 | 1.0033 | 1.00 | 9083 | 10863 | 3.06e+05 | 0.1196 |
| zipf-heavy | random | phi=0.25 | 1e-03 | 1.0035 | 1.0055 | 1.01 | 9083 | 11995 | 4.58e+05 | 0.2404 |
| zipf-heavy | random | phi=0.5 | 1e-03 | 1.0061 | 1.0107 | 1.01 | 9083 | 19583 | 9.17e+05 | 0.4831 |
| zipf-heavy | random | f=0.1 | 1e-03 | 1.0014 | 1.0027 | 1.00 | 9083 | 9792 | 2.76e+05 | 0.0790 |
| zipf-heavy | random | f=0.3 | 1e-03 | 1.0040 | 1.0075 | 1.01 | 9083 | 13525 | 5.28e+05 | 0.3016 |
| zipf-heavy | random | phi=0.125 | 1e-04 | 1.0003 | 1.0006 | 1.00 | 200000 | 215849 | 4.07e+06 | 0.0078 |
| zipf-heavy | random | phi=0.25 | 1e-04 | 1.0007 | 1.0011 | 1.00 | 200000 | 238665 | 6.11e+06 | 0.0432 |
| zipf-heavy | random | phi=0.5 | 1e-04 | 1.0011 | 1.0017 | 1.00 | 200000 | 403680 | 1.22e+07 | 0.1238 |
| zipf-heavy | random | f=0.1 | 1e-04 | 1.0000 | 1.0006 | 1.00 | 200000 | 209792 | 3.68e+06 | 0.0049 |
| zipf-heavy | random | f=0.3 | 1e-04 | 1.0008 | 1.0025 | 1.01 | 200000 | 242938 | 7.04e+06 | 0.0891 |
| zipf-heavy | cluster-sorted | phi=0.125 | none | 1.1163 | 1.7599 | 2.59 | 0 | 0 | 0.00 | 0.1250 |
| zipf-heavy | cluster-sorted | phi=0.25 | none | 1.2702 | 1.8303 | 2.58 | 0 | 0 | 0.00 | 0.2500 |
| zipf-heavy | cluster-sorted | phi=0.5 | none | 1.6695 | 2.6802 | 4.59 | 0 | 0 | 0.00 | 0.5000 |
| zipf-heavy | cluster-sorted | f=0.1 | none | 1.1152 | 1.5779 | 2.04 | 0 | 0 | 0.00 | 0.1050 |
| zipf-heavy | cluster-sorted | f=0.3 | none | 1.4004 | 2.2235 | 4.58 | 0 | 0 | 0.00 | 0.2875 |
| zipf-heavy | cluster-sorted | phi=0.125 | 1e-02 | 1.1151 | 1.6958 | 2.47 | 709 | 709 | 2.04e+04 | 0.1246 |
| zipf-heavy | cluster-sorted | phi=0.25 | 1e-02 | 1.2687 | 1.8170 | 2.37 | 709 | 798 | 3.06e+04 | 0.2492 |
| zipf-heavy | cluster-sorted | phi=0.5 | 1e-02 | 1.6376 | 2.7495 | 4.10 | 709 | 1012 | 6.11e+04 | 0.4989 |
| zipf-heavy | cluster-sorted | f=0.1 | 1e-02 | 1.1102 | 1.5458 | 2.02 | 709 | 709 | 1.84e+04 | 0.1047 |
| zipf-heavy | cluster-sorted | f=0.3 | 1e-02 | 1.4152 | 2.2328 | 4.13 | 709 | 798 | 3.52e+04 | 0.2867 |
| zipf-heavy | cluster-sorted | phi=0.125 | 1e-03 | 1.1052 | 1.7002 | 2.26 | 9083 | 10863 | 3.06e+05 | 0.1210 |
| zipf-heavy | cluster-sorted | phi=0.25 | 1e-03 | 1.2409 | 1.6486 | 2.27 | 9083 | 11995 | 4.58e+05 | 0.2409 |
| zipf-heavy | cluster-sorted | phi=0.5 | 1e-03 | 1.5461 | 2.2521 | 3.20 | 9083 | 19583 | 9.17e+05 | 0.4830 |
| zipf-heavy | cluster-sorted | f=0.1 | 1e-03 | 1.0987 | 1.4771 | 1.81 | 9083 | 9792 | 2.76e+05 | 0.1016 |
| zipf-heavy | cluster-sorted | f=0.3 | 1e-03 | 1.4225 | 2.2274 | 2.57 | 9083 | 13525 | 5.28e+05 | 0.2759 |
| zipf-heavy | cluster-sorted | phi=0.125 | 1e-04 | 1.0056 | 1.0245 | 1.03 | 200000 | 215849 | 4.07e+06 | 0.0079 |
| zipf-heavy | cluster-sorted | phi=0.25 | 1e-04 | 1.0134 | 1.0182 | 1.05 | 200000 | 238665 | 6.11e+06 | 0.0438 |
| zipf-heavy | cluster-sorted | phi=0.5 | 1e-04 | 1.0140 | 1.0294 | 1.04 | 200000 | 403680 | 1.22e+07 | 0.1244 |
| zipf-heavy | cluster-sorted | f=0.1 | 1e-04 | 1.0000 | 1.0435 | 1.07 | 200000 | 209792 | 3.68e+06 | 0.0135 |
| zipf-heavy | cluster-sorted | f=0.3 | 1e-04 | 1.0149 | 1.0497 | 1.11 | 200000 | 242938 | 7.04e+06 | 0.0707 |
| zipf | random | phi=0.125 | none | 1.0034 | 1.0061 | 1.01 | 0 | 0 | 0.00 | 0.1250 |
| zipf | random | phi=0.25 | none | 1.0055 | 1.0077 | 1.01 | 0 | 0 | 0.00 | 0.2500 |
| zipf | random | phi=0.5 | none | 1.0103 | 1.0154 | 1.03 | 0 | 0 | 0.00 | 0.5000 |
| zipf | random | f=0.1 | none | 1.0027 | 1.0057 | 1.01 | 0 | 0 | 0.00 | 0.0956 |
| zipf | random | f=0.3 | none | 1.0061 | 1.0117 | 1.02 | 0 | 0 | 0.00 | 0.3019 |
| zipf | random | phi=0.125 | 1e-02 | 1.0034 | 1.0061 | 1.01 | 0 | 0 | 2.04e+04 | 0.1250 |
| zipf | random | phi=0.25 | 1e-02 | 1.0055 | 1.0077 | 1.01 | 0 | 0 | 3.06e+04 | 0.2500 |
| zipf | random | phi=0.5 | 1e-02 | 1.0103 | 1.0154 | 1.03 | 0 | 0 | 6.11e+04 | 0.5000 |
| zipf | random | f=0.1 | 1e-02 | 1.0027 | 1.0057 | 1.01 | 0 | 0 | 1.84e+04 | 0.0956 |
| zipf | random | f=0.3 | 1e-02 | 1.0061 | 1.0117 | 1.02 | 0 | 0 | 3.52e+04 | 0.3019 |
| zipf | random | phi=0.125 | 1e-03 | 1.0021 | 1.0034 | 1.01 | 24288 | 24288 | 3.06e+05 | 0.1109 |
| zipf | random | phi=0.25 | 1e-03 | 1.0033 | 1.0058 | 1.01 | 24288 | 24288 | 4.58e+05 | 0.2257 |
| zipf | random | phi=0.5 | 1e-03 | 1.0062 | 1.0097 | 1.02 | 24288 | 32086 | 9.17e+05 | 0.4624 |
| zipf | random | f=0.1 | 1e-03 | 1.0015 | 1.0037 | 1.01 | 24288 | 24288 | 2.76e+05 | 0.0854 |
| zipf | random | f=0.3 | 1e-03 | 1.0040 | 1.0072 | 1.01 | 24288 | 25318 | 5.28e+05 | 0.2758 |
| zipf | random | phi=0.125 | 1e-04 | 1.0004 | 1.0006 | 1.00 | 199998 | 240452 | 4.07e+06 | 0.0067 |
| zipf | random | phi=0.25 | 1e-04 | 1.0008 | 1.0012 | 1.00 | 199998 | 268737 | 6.11e+06 | 0.0375 |
| zipf | random | phi=0.5 | 1e-04 | 1.0010 | 1.0017 | 1.00 | 199998 | 418084 | 1.22e+07 | 0.1382 |
| zipf | random | f=0.1 | 1e-04 | 1.0000 | 1.0015 | 1.00 | 199998 | 224286 | 3.68e+06 | 0.0106 |
| zipf | random | f=0.3 | 1e-04 | 1.0009 | 1.0023 | 1.01 | 199998 | 288884 | 7.04e+06 | 0.0660 |
| zipf | cluster-sorted | phi=0.125 | none | 1.2401 | 1.3872 | 2.42 | 0 | 0 | 0.00 | 0.1250 |
| zipf | cluster-sorted | phi=0.25 | none | 1.2787 | 1.7842 | 2.86 | 0 | 0 | 0.00 | 0.2500 |
| zipf | cluster-sorted | phi=0.5 | none | 1.6506 | 2.5932 | 4.13 | 0 | 0 | 0.00 | 0.5000 |
| zipf | cluster-sorted | f=0.1 | none | 1.2271 | 1.4242 | 1.82 | 0 | 0 | 0.00 | 0.1081 |
| zipf | cluster-sorted | f=0.3 | none | 1.3502 | 1.9862 | 2.85 | 0 | 0 | 0.00 | 0.2906 |
| zipf | cluster-sorted | phi=0.125 | 1e-02 | 1.2401 | 1.3872 | 2.42 | 0 | 0 | 2.04e+04 | 0.1250 |
| zipf | cluster-sorted | phi=0.25 | 1e-02 | 1.2787 | 1.7842 | 2.86 | 0 | 0 | 3.06e+04 | 0.2500 |
| zipf | cluster-sorted | phi=0.5 | 1e-02 | 1.6506 | 2.5932 | 4.13 | 0 | 0 | 6.11e+04 | 0.5000 |
| zipf | cluster-sorted | f=0.1 | 1e-02 | 1.2271 | 1.4242 | 1.82 | 0 | 0 | 1.84e+04 | 0.1081 |
| zipf | cluster-sorted | f=0.3 | 1e-02 | 1.3502 | 1.9862 | 2.85 | 0 | 0 | 3.52e+04 | 0.2906 |
| zipf | cluster-sorted | phi=0.125 | 1e-03 | 1.1865 | 1.3835 | 1.83 | 24288 | 24288 | 3.06e+05 | 0.1112 |
| zipf | cluster-sorted | phi=0.25 | 1e-03 | 1.2541 | 1.4840 | 1.83 | 24288 | 24288 | 4.58e+05 | 0.2246 |
| zipf | cluster-sorted | phi=0.5 | 1e-03 | 1.4709 | 1.8140 | 2.69 | 24288 | 32086 | 9.17e+05 | 0.4610 |
| zipf | cluster-sorted | f=0.1 | 1e-03 | 1.1428 | 1.3694 | 1.76 | 24288 | 24288 | 2.76e+05 | 0.0956 |
| zipf | cluster-sorted | f=0.3 | 1e-03 | 1.2885 | 1.7674 | 2.04 | 24288 | 25318 | 5.28e+05 | 0.2664 |
| zipf | cluster-sorted | phi=0.125 | 1e-04 | 1.0076 | 1.0160 | 1.03 | 199998 | 240452 | 4.07e+06 | 0.0067 |
| zipf | cluster-sorted | phi=0.25 | 1e-04 | 1.0276 | 1.0430 | 1.08 | 199998 | 268737 | 6.11e+06 | 0.0371 |
| zipf | cluster-sorted | phi=0.5 | 1e-04 | 1.0251 | 1.0464 | 1.06 | 199998 | 418084 | 1.22e+07 | 0.1388 |
| zipf | cluster-sorted | f=0.1 | 1e-04 | 1.0054 | 1.0273 | 1.06 | 199998 | 224286 | 3.68e+06 | 0.0090 |
| zipf | cluster-sorted | f=0.3 | 1e-04 | 1.0299 | 1.1083 | 1.15 | 199998 | 288884 | 7.04e+06 | 0.0606 |
| esa22 | random | phi=0.125 | none | 1.0004 | 1.0006 | 1.00 | 0 | 0 | 0.00 | 0.1250 |
| esa22 | random | phi=0.25 | none | 1.0007 | 1.0010 | 1.00 | 0 | 0 | 0.00 | 0.2500 |
| esa22 | random | phi=0.5 | none | 1.0012 | 1.0016 | 1.00 | 0 | 0 | 0.00 | 0.5000 |
| esa22 | random | f=0.1 | none | 1.0004 | 1.0006 | 1.00 | 0 | 0 | 0.00 | 0.1031 |
| esa22 | random | f=0.3 | none | 1.0008 | 1.0011 | 1.00 | 0 | 0 | 0.00 | 0.2919 |
| esa22 | random | phi=0.125 | 1e-02 | 1.0004 | 1.0006 | 1.00 | 0 | 0 | 1.15e+04 | 0.1250 |
| esa22 | random | phi=0.25 | 1e-02 | 1.0007 | 1.0010 | 1.00 | 0 | 0 | 1.73e+04 | 0.2500 |
| esa22 | random | phi=0.5 | 1e-02 | 1.0012 | 1.0016 | 1.00 | 0 | 0 | 3.45e+04 | 0.5000 |
| esa22 | random | f=0.1 | 1e-02 | 1.0004 | 1.0006 | 1.00 | 0 | 0 | 1.04e+04 | 0.1031 |
| esa22 | random | f=0.3 | 1e-02 | 1.0008 | 1.0011 | 1.00 | 0 | 0 | 1.99e+04 | 0.2919 |
| esa22 | random | phi=0.125 | 1e-03 | 1.0004 | 1.0006 | 1.00 | 4355 | 4355 | 1.73e+05 | 0.1200 |
| esa22 | random | phi=0.25 | 1e-03 | 1.0007 | 1.0009 | 1.00 | 4355 | 4355 | 2.59e+05 | 0.2413 |
| esa22 | random | phi=0.5 | 1e-03 | 1.0012 | 1.0016 | 1.00 | 4355 | 4355 | 5.18e+05 | 0.4884 |
| esa22 | random | f=0.1 | 1e-03 | 1.0004 | 1.0006 | 1.00 | 4355 | 4355 | 1.56e+05 | 0.0991 |
| esa22 | random | f=0.3 | 1e-03 | 1.0008 | 1.0011 | 1.00 | 4355 | 4355 | 2.98e+05 | 0.2829 |
| esa22 | random | phi=0.125 | 1e-04 | 1.0001 | 1.0001 | 1.00 | 100000 | 119192 | 2.30e+06 | 0.0067 |
| esa22 | random | phi=0.25 | 1e-04 | 1.0002 | 1.0003 | 1.00 | 100000 | 158383 | 3.45e+06 | 0.0249 |
| esa22 | random | phi=0.5 | 1e-04 | 1.0003 | 1.0004 | 1.00 | 100000 | 277575 | 6.91e+06 | 0.0592 |
| esa22 | random | f=0.1 | 1e-04 | 1.0000 | 1.0002 | 1.00 | 100000 | 104355 | 2.08e+06 | 0.0100 |
| esa22 | random | f=0.3 | 1e-04 | 1.0002 | 1.0004 | 1.00 | 100000 | 175402 | 3.98e+06 | 0.0378 |
| esa22 | cluster-sorted | phi=0.125 | none | 1.0082 | 1.0099 | 1.01 | 0 | 0 | 0.00 | 0.1250 |
| esa22 | cluster-sorted | phi=0.25 | none | 1.0123 | 1.0156 | 1.02 | 0 | 0 | 0.00 | 0.2500 |
| esa22 | cluster-sorted | phi=0.5 | none | 1.0211 | 1.0276 | 1.03 | 0 | 0 | 0.00 | 0.5000 |
| esa22 | cluster-sorted | f=0.1 | none | 1.0062 | 1.0104 | 1.02 | 0 | 0 | 0.00 | 0.0938 |
| esa22 | cluster-sorted | f=0.3 | none | 1.0135 | 1.0217 | 1.03 | 0 | 0 | 0.00 | 0.3069 |
| esa22 | cluster-sorted | phi=0.125 | 1e-02 | 1.0082 | 1.0099 | 1.01 | 0 | 0 | 1.15e+04 | 0.1250 |
| esa22 | cluster-sorted | phi=0.25 | 1e-02 | 1.0123 | 1.0156 | 1.02 | 0 | 0 | 1.73e+04 | 0.2500 |
| esa22 | cluster-sorted | phi=0.5 | 1e-02 | 1.0211 | 1.0276 | 1.03 | 0 | 0 | 3.45e+04 | 0.5000 |
| esa22 | cluster-sorted | f=0.1 | 1e-02 | 1.0062 | 1.0104 | 1.02 | 0 | 0 | 1.04e+04 | 0.0938 |
| esa22 | cluster-sorted | f=0.3 | 1e-02 | 1.0135 | 1.0217 | 1.03 | 0 | 0 | 1.99e+04 | 0.3069 |
| esa22 | cluster-sorted | phi=0.125 | 1e-03 | 1.0078 | 1.0096 | 1.01 | 4355 | 4355 | 1.73e+05 | 0.1203 |
| esa22 | cluster-sorted | phi=0.25 | 1e-03 | 1.0115 | 1.0146 | 1.02 | 4355 | 4355 | 2.59e+05 | 0.2404 |
| esa22 | cluster-sorted | phi=0.5 | 1e-03 | 1.0204 | 1.0266 | 1.03 | 4355 | 4355 | 5.18e+05 | 0.4883 |
| esa22 | cluster-sorted | f=0.1 | 1e-03 | 1.0060 | 1.0100 | 1.02 | 4355 | 4355 | 1.56e+05 | 0.0899 |
| esa22 | cluster-sorted | f=0.3 | 1e-03 | 1.0130 | 1.0206 | 1.03 | 4355 | 4355 | 2.98e+05 | 0.2972 |
| esa22 | cluster-sorted | phi=0.125 | 1e-04 | 1.0004 | 1.0005 | 1.00 | 100000 | 119192 | 2.30e+06 | 0.0067 |
| esa22 | cluster-sorted | phi=0.25 | 1e-04 | 1.0010 | 1.0013 | 1.00 | 100000 | 158383 | 3.45e+06 | 0.0248 |
| esa22 | cluster-sorted | phi=0.5 | 1e-04 | 1.0014 | 1.0019 | 1.00 | 100000 | 277575 | 6.91e+06 | 0.0591 |
| esa22 | cluster-sorted | f=0.1 | 1e-04 | 1.0000 | 1.0010 | 1.00 | 100000 | 104355 | 2.08e+06 | 0.0084 |
| esa22 | cluster-sorted | f=0.3 | 1e-04 | 1.0011 | 1.0029 | 1.01 | 100000 | 175402 | 3.98e+06 | 0.0430 |
| esa22 | input | phi=0.125 | none | 1.0091 | 1.0126 | 1.01 | 0 | 0 | 0.00 | 0.1250 |
| esa22 | input | phi=0.25 | none | 1.0130 | 1.0178 | 1.02 | 0 | 0 | 0.00 | 0.2500 |
| esa22 | input | phi=0.5 | none | 1.0225 | 1.0318 | 1.05 | 0 | 0 | 0.00 | 0.5000 |
| esa22 | input | f=0.1 | none | 1.0071 | 1.0119 | 1.01 | 0 | 0 | 0.00 | 0.0963 |
| esa22 | input | f=0.3 | none | 1.0154 | 1.0241 | 1.04 | 0 | 0 | 0.00 | 0.2925 |
| esa22 | input | phi=0.125 | 1e-02 | 1.0091 | 1.0126 | 1.01 | 0 | 0 | 1.15e+04 | 0.1250 |
| esa22 | input | phi=0.25 | 1e-02 | 1.0130 | 1.0178 | 1.02 | 0 | 0 | 1.73e+04 | 0.2500 |
| esa22 | input | phi=0.5 | 1e-02 | 1.0225 | 1.0318 | 1.05 | 0 | 0 | 3.45e+04 | 0.5000 |
| esa22 | input | f=0.1 | 1e-02 | 1.0071 | 1.0119 | 1.01 | 0 | 0 | 1.04e+04 | 0.0963 |
| esa22 | input | f=0.3 | 1e-02 | 1.0154 | 1.0241 | 1.04 | 0 | 0 | 1.99e+04 | 0.2925 |
| esa22 | input | phi=0.125 | 1e-03 | 1.0088 | 1.0120 | 1.01 | 4355 | 4355 | 1.73e+05 | 0.1193 |
| esa22 | input | phi=0.25 | 1e-03 | 1.0125 | 1.0171 | 1.02 | 4355 | 4355 | 2.59e+05 | 0.2420 |
| esa22 | input | phi=0.5 | 1e-03 | 1.0219 | 1.0312 | 1.05 | 4355 | 4355 | 5.18e+05 | 0.4887 |
| esa22 | input | f=0.1 | 1e-03 | 1.0069 | 1.0115 | 1.01 | 4355 | 4355 | 1.56e+05 | 0.0921 |
| esa22 | input | f=0.3 | 1e-03 | 1.0148 | 1.0225 | 1.03 | 4355 | 4355 | 2.98e+05 | 0.2831 |
| esa22 | input | phi=0.125 | 1e-04 | 1.0005 | 1.0006 | 1.00 | 100000 | 119192 | 2.30e+06 | 0.0067 |
| esa22 | input | phi=0.25 | 1e-04 | 1.0010 | 1.0014 | 1.00 | 100000 | 158383 | 3.45e+06 | 0.0250 |
| esa22 | input | phi=0.5 | 1e-04 | 1.0015 | 1.0021 | 1.00 | 100000 | 277575 | 6.91e+06 | 0.0596 |
| esa22 | input | f=0.1 | 1e-04 | 1.0004 | 1.0012 | 1.00 | 100000 | 104355 | 2.08e+06 | 0.0069 |
| esa22 | input | f=0.3 | 1e-04 | 1.0011 | 1.0028 | 1.01 | 100000 | 175402 | 3.98e+06 | 0.0373 |

zipf-heavy, random partition, phi = 1/2: p90 distortion 1.0170 without replication, 1.0107 with tau = 1e-3 (19583 extra copies for 200000 points).



## F2: node crash after compute (coreset message lost), ring replication of coresets

| nodes | partition | f | r | coreset loss rate | f^(r+1) | excess MS r(Q) | predicted (A1 at phi_eff) | distortion p50 | p99 | downstream mean | max | bytes/node |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 16 | random | f=0.05 | 0 | 0.0525 | 0.0500 | 1.70e-05 | 1.66e-05 | 1.0336 | 1.0635 | 1.0020 | 1.003 | 90293 |
| 16 | random | f=0.05 | 1 | 0.0027 | 0.0025 | 8.94e-07 | 7.94e-07 | 1.0335 | 1.0635 | 1.0019 | 1.002 | 190334 |
| 16 | random | f=0.05 | 2 | 0.0000 | 0.0001 | 0.00 | 0.00 | 1.0334 | 1.0635 | 1.0154 | 1.407 | 285889 |
| 16 | random | f=0.1 | 0 | 0.0975 | 0.1000 | 3.34e-05 | 3.16e-05 | 1.0342 | 1.0658 | 1.0021 | 1.003 | 86006 |
| 16 | random | f=0.1 | 1 | 0.0094 | 0.0100 | 2.83e-06 | 2.65e-06 | 1.0335 | 1.0635 | 1.0019 | 1.002 | 189699 |
| 16 | random | f=0.1 | 2 | 0.0010 | 0.0010 | 1.82e-07 | 2.79e-07 | 1.0335 | 1.0635 | 1.0019 | 1.002 | 285790 |
| 16 | random | f=0.2 | 0 | 0.2007 | 0.2000 | 7.97e-05 | 7.64e-05 | 1.0366 | 1.0701 | 1.0025 | 1.004 | 76163 |
| 16 | random | f=0.2 | 1 | 0.0413 | 0.0400 | 1.27e-05 | 1.35e-05 | 1.0335 | 1.0635 | 1.0020 | 1.002 | 186658 |
| 16 | random | f=0.2 | 2 | 0.0085 | 0.0080 | 2.63e-06 | 2.75e-06 | 1.0334 | 1.0635 | 1.0019 | 1.002 | 285074 |
| 16 | random | f=0.3 | 0 | 0.3058 | 0.3000 | 0.000141 | 0.000135 | 1.0408 | 1.0793 | 1.0030 | 1.005 | 66150 |
| 16 | random | f=0.3 | 1 | 0.0910 | 0.0900 | 3.27e-05 | 3.07e-05 | 1.0350 | 1.0643 | 1.0021 | 1.003 | 181916 |
| 16 | random | f=0.3 | 2 | 0.0256 | 0.0270 | 8.13e-06 | 8.05e-06 | 1.0335 | 1.0635 | 1.0019 | 1.003 | 283449 |
| 16 | cluster-sorted | f=0.05 | 0 | 0.0516 | 0.0500 | 0.00322 | 2.85e-05 | 1.0842 | 1.7104 | 1.6949 | 8.975 | 91382 |
| 16 | cluster-sorted | f=0.05 | 1 | 0.0027 | 0.0025 | 0.000158 | 1.84e-06 | 1.0456 | 1.1679 | 1.0570 | 2.203 | 192455 |
| 16 | cluster-sorted | f=0.05 | 2 | 0.0003 | 0.0001 | 1.43e-05 | 1.10e-07 | 1.0456 | 1.1006 | 1.0033 | 1.004 | 289045 |
| 16 | cluster-sorted | f=0.1 | 0 | 0.1024 | 0.1000 | 0.00651 | 6.00e-05 | 1.1674 | 1.9821 | 3.2205 | 23.054 | 86497 |
| 16 | cluster-sorted | f=0.1 | 1 | 0.0099 | 0.0100 | 0.000561 | 5.07e-06 | 1.0464 | 1.3785 | 1.2145 | 5.475 | 191766 |
| 16 | cluster-sorted | f=0.1 | 2 | 0.0009 | 0.0010 | 6.21e-05 | 5.91e-07 | 1.0456 | 1.1006 | 1.0033 | 1.004 | 288984 |
| 16 | cluster-sorted | f=0.2 | 0 | 0.2007 | 0.2000 | 0.0145 | 0.000135 | 1.2744 | 2.2903 | 5.1054 | 19.036 | 77013 |
| 16 | cluster-sorted | f=0.2 | 1 | 0.0375 | 0.0400 | 0.00235 | 2.12e-05 | 1.0505 | 1.7904 | 1.8506 | 7.571 | 189099 |
| 16 | cluster-sorted | f=0.2 | 2 | 0.0085 | 0.0080 | 0.000483 | 4.89e-06 | 1.0457 | 1.3323 | 1.2991 | 6.785 | 288250 |
| 16 | cluster-sorted | f=0.3 | 0 | 0.3077 | 0.3000 | 0.0261 | 0.000235 | 1.3770 | 3.1189 | 9.9573 | 39.046 | 66701 |
| 16 | cluster-sorted | f=0.3 | 1 | 0.0974 | 0.0900 | 0.00618 | 5.74e-05 | 1.1450 | 1.8920 | 2.5107 | 7.277 | 183331 |
| 16 | cluster-sorted | f=0.3 | 2 | 0.0316 | 0.0270 | 0.00191 | 1.73e-05 | 1.0477 | 1.6542 | 1.0727 | 3.082 | 286033 |
| 64 | random | f=0.05 | 0 | 0.0521 | 0.0500 | 4.66e-06 | 5.26e-06 | 1.0171 | 1.0293 |  |  | 77786 |
| 64 | random | f=0.05 | 1 | 0.0024 | 0.0025 | 2.05e-07 | 2.37e-07 | 1.0161 | 1.0273 |  |  | 163926 |
| 64 | random | f=0.05 | 2 | 0.0001 | 0.0001 | 1.27e-08 | 1.04e-08 | 1.0161 | 1.0273 |  |  | 246179 |
| 64 | random | f=0.1 | 0 | 0.1001 | 0.1000 | 9.44e-06 | 1.07e-05 | 1.0178 | 1.0321 |  |  | 73849 |
| 64 | random | f=0.1 | 1 | 0.0104 | 0.0100 | 9.00e-07 | 9.83e-07 | 1.0163 | 1.0273 |  |  | 163275 |
| 64 | random | f=0.1 | 2 | 0.0009 | 0.0010 | 9.03e-08 | 8.24e-08 | 1.0161 | 1.0273 |  |  | 246113 |
| 64 | random | f=0.2 | 0 | 0.2000 | 0.2000 | 1.98e-05 | 2.42e-05 | 1.0191 | 1.0353 |  |  | 65650 |
| 64 | random | f=0.2 | 1 | 0.0396 | 0.0400 | 3.56e-06 | 4.04e-06 | 1.0169 | 1.0310 |  |  | 160876 |
| 64 | random | f=0.2 | 2 | 0.0074 | 0.0080 | 6.01e-07 | 7.40e-07 | 1.0163 | 1.0274 |  |  | 245581 |
| 64 | random | f=0.3 | 0 | 0.2972 | 0.3000 | 3.77e-05 | 4.09e-05 | 1.0209 | 1.0420 |  |  | 57680 |
| 64 | random | f=0.3 | 1 | 0.0896 | 0.0900 | 8.39e-06 | 9.50e-06 | 1.0174 | 1.0315 |  |  | 156770 |
| 64 | random | f=0.3 | 2 | 0.0266 | 0.0270 | 2.37e-06 | 2.65e-06 | 1.0165 | 1.0280 |  |  | 244003 |



## F3: packet erasure of coreset points

### B1: erasure invariance (Bernoulli, `ht-realized`)

| instance | m | packetization | slope of log(distortion-1) vs log 1/(1-rho) | 95% CI | excess ratio rho=0.5 : 0 |
|---|---|---|---|---|---|
| zipf | 1000 | per-draw | 0.563 | [0.463, 0.655] | 1.458 |
| zipf | 1000 | random-b64 | 0.504 | [0.381, 0.635] | 1.410 |
| zipf | 1000 | merged | 0.576 | [0.448, 0.701] | 1.503 |
| zipf | 1000 | rateless | 0.114 | [-0.022, 0.251] | 1.069 |
| zipf | 4000 | per-draw | 0.492 | [0.384, 0.600] | 1.403 |
| zipf | 4000 | random-b64 | 0.534 | [0.428, 0.648] | 1.445 |
| zipf | 4000 | merged | 0.482 | [0.368, 0.599] | 1.391 |
| zipf | 4000 | rateless | 0.026 | [-0.101, 0.143] | 1.016 |
| zipf-heavy | 1000 | per-draw | 0.529 | [0.381, 0.673] | 1.422 |
| zipf-heavy | 1000 | random-b64 | 0.532 | [0.373, 0.696] | 1.444 |
| zipf-heavy | 1000 | merged | 0.384 | [0.244, 0.520] | 1.290 |
| zipf-heavy | 1000 | rateless | -0.032 | [-0.199, 0.126] | 0.988 |
| zipf-heavy | 4000 | per-draw | 0.576 | [0.421, 0.724] | 1.486 |
| zipf-heavy | 4000 | random-b64 | 0.493 | [0.331, 0.642] | 1.427 |
| zipf-heavy | 4000 | merged | 0.650 | [0.506, 0.802] | 1.570 |
| zipf-heavy | 4000 | rateless | -0.044 | [-0.213, 0.119] | 0.968 |
| esa22 | 1000 | per-draw | 0.541 | [0.418, 0.672] | 1.441 |
| esa22 | 1000 | random-b64 | 0.509 | [0.356, 0.687] | 1.440 |
| esa22 | 1000 | merged | 0.527 | [0.371, 0.673] | 1.414 |
| esa22 | 1000 | rateless | 0.007 | [-0.151, 0.174] | 1.008 |
| esa22 | 4000 | per-draw | 0.547 | [0.398, 0.712] | 1.455 |
| esa22 | 4000 | random-b64 | 0.536 | [0.380, 0.705] | 1.473 |
| esa22 | 4000 | merged | 0.565 | [0.406, 0.715] | 1.502 |
| esa22 | 4000 | rateless | 0.055 | [-0.121, 0.218] | 1.032 |

Mean distortion by rho (zipf, m = 1000, Bernoulli, `ht-realized`):

| rho | per-draw | merged | random-b8 | random-b64 | sorted-b8 | rateless |
|---|---|---|---|---|---|---|
| 0.0 | 1.0950 | 1.0950 | 1.0950 | 1.0950 | 1.0950 | 1.0950 |
| 0.01 | 1.0948 | 1.0956 | 1.0953 | 1.0949 | 1.0975 | 1.0955 |
| 0.05 | 1.0972 | 1.0980 | 1.0983 | 1.0989 | 1.1135 | 1.0959 |
| 0.1 | 1.1015 | 1.0998 | 1.1017 | 1.1004 | 1.1345 | 1.0923 |
| 0.2 | 1.1095 | 1.1052 | 1.1113 | 1.1073 | 1.1690 | 1.0952 |
| 0.3 | 1.1195 | 1.1131 | 1.1158 | 1.1167 | 1.2061 | 1.1012 |
| 0.5 | 1.1385 | 1.1428 | 1.1440 | 1.1339 | 1.3177 | 1.1016 |

Decoders at rho = 0.3 (zipf, m = 1000, per-draw, Bernoulli): none mean distortion 1.5620; ht-prior mean distortion 1.1300; ht-realized mean distortion 1.1195; 

### Bursts (zipf, m = 1000, rho = 0.3, `ht-realized`)

| channel | packetization | distortion mean | sd | p99 | max | sd(m') |
|---|---|---|---|---|---|---|
| bernoulli | per-draw | 1.1195 | 0.0538 | 1.2734 | 1.2886 | 15.8 |
| bernoulli | merged | 1.1131 | 0.0383 | 1.2169 | 1.2565 | 16.0 |
| bernoulli | random-b64 | 1.1167 | 0.0420 | 1.2377 | 1.2757 | 119.1 |
| bernoulli | sorted-b8 | 1.2061 | 0.0792 | 1.4340 | 1.5324 | 38.7 |
| ge-b1 | per-draw | 1.1182 | 0.0439 | 1.2453 | 1.2809 | 8.9 |
| ge-b1 | merged | 1.1165 | 0.0506 | 1.3074 | 1.3211 | 9.0 |
| ge-b1 | random-b64 | 1.1178 | 0.0429 | 1.2429 | 1.2741 | 76.0 |
| ge-b1 | sorted-b8 | 1.1556 | 0.0618 | 1.3305 | 1.5222 | 29.3 |
| ge-b5 | per-draw | 1.1155 | 0.0446 | 1.2599 | 1.2826 | 31.6 |
| ge-b5 | merged | 1.1161 | 0.0513 | 1.2636 | 1.3242 | 31.6 |
| ge-b5 | random-b64 | Infinity | NaN | Infinity | Infinity | 280.5 |
| ge-b5 | sorted-b8 | 1.4102 | 0.2501 | 2.0965 | 2.5354 | 103.9 |
| ge-b20 | per-draw | 1.1222 | 0.0486 | 1.2668 | 1.2698 | 79.1 |
| ge-b20 | merged | 1.1185 | 0.0466 | 1.2536 | 1.2561 | 79.8 |
| ge-b20 | random-b64 | Infinity | NaN | NaN | Infinity | 408.4 |
| ge-b20 | sorted-b8 | 2.0303 | 1.2835 | 7.9926 | 8.5441 | 212.4 |

### B2: merged duplicates versus split packets (Bernoulli, `ht-prior`)

RMS of the erasure perturbation r_rho(Q*) - r_0(Q*) (same sample) for the candidate Q* with the largest kappa_pt. Predictions from Theorem B2 with a = sum_p c_p^2/q_p / cost^2 measured on each seed's proposal: split rho/(1-rho) a/m; merged rho/(1-rho) ((a - kappa)/m + (1 - 1/m) kappa); floor sqrt(rho/(1-rho) kappa).

| instance | kappa_pt(Q*) | a(Q*) | rho | m | distinct/m | split RMS | predicted | merged RMS | predicted | floor |
|---|---|---|---|---|---|---|---|---|---|---|
| zipf | 7.48e-05 | 4.40 | 0.1 | 250 | 0.998 | 0.0449 | 0.0442 | 0.0356 | 0.0443 | 0.0029 |
| zipf | 7.48e-05 | 4.40 | 0.1 | 500 | 0.996 | 0.0308 | 0.0313 | 0.0358 | 0.0314 | 0.0029 |
| zipf | 7.48e-05 | 4.40 | 0.1 | 1000 | 0.993 | 0.0247 | 0.0221 | 0.0224 | 0.0223 | 0.0029 |
| zipf | 7.48e-05 | 4.40 | 0.1 | 2000 | 0.985 | 0.0154 | 0.0156 | 0.0154 | 0.0159 | 0.0029 |
| zipf | 7.48e-05 | 4.40 | 0.1 | 4000 | 0.971 | 0.0112 | 0.0111 | 0.0110 | 0.0114 | 0.0029 |
| zipf | 7.48e-05 | 4.40 | 0.1 | 8000 | 0.944 | 0.0076 | 0.0078 | 0.0078 | 0.0083 | 0.0029 |
| zipf | 7.48e-05 | 4.40 | 0.3 | 250 | 0.998 | 0.0894 | 0.0869 | 0.0867 | 0.0870 | 0.0057 |
| zipf | 7.48e-05 | 4.40 | 0.3 | 500 | 0.996 | 0.0628 | 0.0614 | 0.0586 | 0.0617 | 0.0057 |
| zipf | 7.48e-05 | 4.40 | 0.3 | 1000 | 0.993 | 0.0424 | 0.0434 | 0.0415 | 0.0438 | 0.0057 |
| zipf | 7.48e-05 | 4.40 | 0.3 | 2000 | 0.985 | 0.0287 | 0.0307 | 0.0262 | 0.0312 | 0.0057 |
| zipf | 7.48e-05 | 4.40 | 0.3 | 4000 | 0.971 | 0.0216 | 0.0217 | 0.0240 | 0.0224 | 0.0057 |
| zipf | 7.48e-05 | 4.40 | 0.3 | 8000 | 0.944 | 0.0159 | 0.0154 | 0.0159 | 0.0164 | 0.0057 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.1 | 250 | 0.993 | 0.0403 | 0.0419 | 0.0439 | 0.0419 | 0.0027 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.1 | 500 | 0.988 | 0.0264 | 0.0296 | 0.0281 | 0.0297 | 0.0027 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.1 | 1000 | 0.977 | 0.0220 | 0.0209 | 0.0254 | 0.0211 | 0.0027 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.1 | 2000 | 0.956 | 0.0154 | 0.0148 | 0.0148 | 0.0151 | 0.0027 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.1 | 4000 | 0.921 | 0.0103 | 0.0105 | 0.0103 | 0.0108 | 0.0027 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.1 | 8000 | 0.867 | 0.0077 | 0.0074 | 0.0077 | 0.0079 | 0.0027 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.3 | 250 | 0.993 | 0.0890 | 0.0822 | 0.0931 | 0.0824 | 0.0054 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.3 | 500 | 0.988 | 0.0519 | 0.0581 | 0.0572 | 0.0584 | 0.0054 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.3 | 1000 | 0.977 | 0.0379 | 0.0411 | 0.0364 | 0.0415 | 0.0054 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.3 | 2000 | 0.956 | 0.0277 | 0.0291 | 0.0331 | 0.0296 | 0.0054 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.3 | 4000 | 0.921 | 0.0208 | 0.0206 | 0.0200 | 0.0212 | 0.0054 |
| zipf-heavy | 6.74e-05 | 3.94 | 0.3 | 8000 | 0.867 | 0.0122 | 0.0145 | 0.0124 | 0.0155 | 0.0054 |
| esa22 | 1.04e-05 | 1.21 | 0.1 | 250 | 0.998 | 0.0219 | 0.0232 | 0.0206 | 0.0233 | 0.0011 |
| esa22 | 1.04e-05 | 1.21 | 0.1 | 500 | 0.997 | 0.0143 | 0.0164 | 0.0156 | 0.0165 | 0.0011 |
| esa22 | 1.04e-05 | 1.21 | 0.1 | 1000 | 0.994 | 0.0110 | 0.0116 | 0.0124 | 0.0117 | 0.0011 |
| esa22 | 1.04e-05 | 1.21 | 0.1 | 2000 | 0.989 | 0.0079 | 0.0082 | 0.0077 | 0.0083 | 0.0011 |
| esa22 | 1.04e-05 | 1.21 | 0.1 | 4000 | 0.976 | 0.0066 | 0.0058 | 0.0070 | 0.0059 | 0.0011 |
| esa22 | 1.04e-05 | 1.21 | 0.1 | 8000 | 0.954 | 0.0039 | 0.0041 | 0.0043 | 0.0042 | 0.0011 |
| esa22 | 1.04e-05 | 1.21 | 0.3 | 250 | 0.998 | 0.0457 | 0.0456 | 0.0457 | 0.0457 | 0.0021 |
| esa22 | 1.04e-05 | 1.21 | 0.3 | 500 | 0.997 | 0.0305 | 0.0323 | 0.0318 | 0.0323 | 0.0021 |
| esa22 | 1.04e-05 | 1.21 | 0.3 | 1000 | 0.994 | 0.0265 | 0.0228 | 0.0265 | 0.0229 | 0.0021 |
| esa22 | 1.04e-05 | 1.21 | 0.3 | 2000 | 0.989 | 0.0159 | 0.0161 | 0.0150 | 0.0163 | 0.0021 |
| esa22 | 1.04e-05 | 1.21 | 0.3 | 4000 | 0.976 | 0.0124 | 0.0114 | 0.0116 | 0.0116 | 0.0021 |
| esa22 | 1.04e-05 | 1.21 | 0.3 | 8000 | 0.954 | 0.0078 | 0.0081 | 0.0086 | 0.0083 | 0.0021 |
| zipf-outliers | 0.0139 | 25.0 | 0.1 | 250 | 0.798 | 0.1057 | 0.1054 | 0.1123 | 0.1125 | 0.0393 |
| zipf-outliers | 0.0139 | 25.0 | 0.1 | 500 | 0.776 | 0.0776 | 0.0745 | 0.0832 | 0.0842 | 0.0393 |
| zipf-outliers | 0.0139 | 25.0 | 0.1 | 1000 | 0.766 | 0.0502 | 0.0527 | 0.0623 | 0.0657 | 0.0393 |
| zipf-outliers | 0.0139 | 25.0 | 0.1 | 2000 | 0.757 | 0.0399 | 0.0373 | 0.0482 | 0.0542 | 0.0393 |
| zipf-outliers | 0.0139 | 25.0 | 0.1 | 4000 | 0.742 | 0.0258 | 0.0264 | 0.0453 | 0.0473 | 0.0393 |
| zipf-outliers | 0.0139 | 25.0 | 0.1 | 8000 | 0.721 | 0.0185 | 0.0186 | 0.0537 | 0.0435 | 0.0393 |
| zipf-outliers | 0.0139 | 25.0 | 0.3 | 250 | 0.798 | 0.1805 | 0.2070 | 0.2320 | 0.2209 | 0.0773 |
| zipf-outliers | 0.0139 | 25.0 | 0.3 | 500 | 0.776 | 0.1495 | 0.1464 | 0.1538 | 0.1655 | 0.0773 |
| zipf-outliers | 0.0139 | 25.0 | 0.3 | 1000 | 0.766 | 0.0988 | 0.1035 | 0.1449 | 0.1291 | 0.0773 |
| zipf-outliers | 0.0139 | 25.0 | 0.3 | 2000 | 0.757 | 0.0668 | 0.0732 | 0.1104 | 0.1064 | 0.0773 |
| zipf-outliers | 0.0139 | 25.0 | 0.3 | 4000 | 0.742 | 0.0494 | 0.0518 | 0.0946 | 0.0930 | 0.0773 |
| zipf-outliers | 0.0139 | 25.0 | 0.3 | 8000 | 0.721 | 0.0370 | 0.0366 | 0.0837 | 0.0855 | 0.0773 |

### B4: mean coreset (Bernoulli cell loss)

| instance | rho | variant | distortion mean | p50 | p99 | max | P[> 2] | predicted 1-(1-rho)^far | downstream mean | messages/cells - 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| zipf | 0 | no-loss | 1.0247 | | | | | | | 0 |
| zipf | 0.01 | global-none | 1.0330 | 1.0238 | 1.1489 | 1.26 | 0.000 | 0.000 | 1.0001 | 0.00 |
| zipf | 0.01 | global-realized | 1.0335 | 1.0273 | 1.1395 | 1.24 | 0.000 | 0.000 | 1.0001 | 0.00 |
| zipf | 0.01 | cf-realized | 1.0336 | 1.0273 | 1.1396 | 1.24 | 0.000 | 0.000 | 1.0001 | 0.00 |
| zipf | 0.01 | split | 1.0261 | 1.0262 | 1.0319 | 1.03 | 0.000 | 0.000 | 1.0000 | 4.11 |
| zipf | 0.01 | replicate | 1.0331 | 1.0272 | 1.1283 | 1.22 | 0.000 | 0.000 | 1.0001 | 0.00 |
| zipf | 0.01 | fec | 1.0247 | 1.0246 | 1.0257 | 1.03 | 0.000 | 0.000 | 1.0000 | 0.03 |
| zipf | 0.05 | global-none | 1.1005 | 1.0820 | 1.2877 | 1.32 | 0.000 | 0.000 | 1.0573 | 0.00 |
| zipf | 0.05 | global-realized | 1.0670 | 1.0476 | 1.2379 | 1.27 | 0.000 | 0.000 | 1.0573 | 0.00 |
| zipf | 0.05 | cf-realized | 1.0671 | 1.0476 | 1.2386 | 1.27 | 0.000 | 0.000 | 1.0573 | 0.00 |
| zipf | 0.05 | split | 1.0305 | 1.0309 | 1.0435 | 1.05 | 0.000 | 0.000 | 1.0001 | 4.11 |
| zipf | 0.05 | replicate | 1.0270 | 1.0250 | 1.0834 | 1.22 | 0.000 | 0.000 | 1.0000 | 1.00 |
| zipf | 0.05 | fec | 1.0247 | 1.0246 | 1.0257 | 1.03 | 0.000 | 0.000 | 1.0000 | 0.10 |
| zipf | 0.1 | global-none | 1.1821 | 1.1638 | 1.4159 | 1.52 | 0.000 | 0.000 | 1.2193 | 0.00 |
| zipf | 0.1 | global-realized | 1.0989 | 1.0780 | 1.3217 | 1.34 | 0.000 | 0.000 | 1.2193 | 0.00 |
| zipf | 0.1 | cf-realized | 1.0991 | 1.0784 | 1.3211 | 1.34 | 0.000 | 0.000 | 1.2193 | 0.00 |
| zipf | 0.1 | split | 1.0361 | 1.0349 | 1.0600 | 1.06 | 0.000 | 0.000 | 1.0003 | 4.11 |
| zipf | 0.1 | replicate | 1.0313 | 1.0272 | 1.1076 | 1.14 | 0.000 | 0.000 | 1.0001 | 1.00 |
| zipf | 0.1 | fec | 1.0257 | 1.0246 | 1.0257 | 1.22 | 0.000 | 0.000 | 1.0000 | 0.18 |
| zipf | 0.2 | global-none | 1.3353 | 1.3217 | 1.5386 | 1.77 | 0.000 | 0.000 | 1.2651 | 0.00 |
| zipf | 0.2 | global-realized | 1.1318 | 1.1146 | 1.3172 | 1.60 | 0.000 | 0.000 | 1.2651 | 0.00 |
| zipf | 0.2 | cf-realized | 1.1321 | 1.1144 | 1.3166 | 1.60 | 0.000 | 0.000 | 1.2651 | 0.00 |
| zipf | 0.2 | split | 1.0500 | 1.0479 | 1.0899 | 1.10 | 0.000 | 0.000 | 1.0006 | 4.11 |
| zipf | 0.2 | replicate | 1.0298 | 1.0268 | 1.0941 | 1.18 | 0.000 | 0.000 | 1.0001 | 2.00 |
| zipf | 0.2 | fec | 1.0284 | 1.0246 | 1.0293 | 1.40 | 0.000 | 0.000 | 1.0000 | 0.34 |
| zipf | 0.3 | global-none | 1.5440 | 1.5173 | 1.8807 | 2.19 | 0.005 | 0.000 | 1.7951 | 0.00 |
| zipf | 0.3 | global-realized | 1.1825 | 1.1724 | 1.3838 | 1.60 | 0.000 | 0.000 | 1.7951 | 0.00 |
| zipf | 0.3 | cf-realized | 1.1827 | 1.1727 | 1.3808 | 1.60 | 0.000 | 0.000 | 1.7951 | 0.00 |
| zipf | 0.3 | split | 1.0590 | 1.0564 | 1.1081 | 1.11 | 0.000 | 0.000 | 1.0011 | 4.11 |
| zipf | 0.3 | replicate | 1.0300 | 1.0265 | 1.0766 | 1.22 | 0.000 | 0.000 | 1.0001 | 3.00 |
| zipf | 0.3 | fec | 1.0292 | 1.0246 | 1.2392 | 1.28 | 0.000 | 0.000 | 1.0000 | 0.56 |
| zipf | 0.5 | global-none | 2.2212 | 2.1996 | 2.7825 | 2.88 | 0.875 | 0.000 | 2.8374 | 0.00 |
| zipf | 0.5 | global-realized | 1.2857 | 1.2632 | 1.6114 | 1.62 | 0.000 | 0.000 | 2.8374 | 0.00 |
| zipf | 0.5 | cf-realized | 1.2860 | 1.2651 | 1.6134 | 1.63 | 0.000 | 0.000 | 2.8374 | 0.00 |
| zipf | 0.5 | split | 1.0970 | 1.0930 | 1.1906 | 1.29 | 0.000 | 0.000 | 1.0024 | 4.11 |
| zipf | 0.5 | replicate | 1.0311 | 1.0265 | 1.2246 | 1.24 | 0.000 | 0.000 | 1.0394 | 6.00 |
| zipf | 0.5 | fec | 1.0265 | 1.0246 | 1.0257 | 1.40 | 0.000 | 0.000 | 1.0000 | 1.25 |
| zipf-heavy | 0 | no-loss | 1.1069 | | | | | | | 0 |
| zipf-heavy | 0.01 | global-none | 1.1007 | 1.1019 | 1.1111 | 1.15 | 0.000 | 0.000 | 0.9448 | 0.00 |
| zipf-heavy | 0.01 | global-realized | 1.1073 | 1.1080 | 1.1144 | 1.13 | 0.000 | 0.000 | 0.9448 | 0.00 |
| zipf-heavy | 0.01 | cf-realized | 1.1073 | 1.1079 | 1.1148 | 1.13 | 0.000 | 0.000 | 0.9448 | 0.00 |
| zipf-heavy | 0.01 | split | 1.1073 | 1.1072 | 1.1143 | 1.11 | 0.000 | 0.000 | 0.9471 | 4.09 |
| zipf-heavy | 0.01 | replicate | 1.1082 | 1.1082 | 1.1171 | 1.14 | 0.000 | 0.000 | 0.9754 | 0.00 |
| zipf-heavy | 0.01 | fec | 1.1069 | 1.1067 | 1.1109 | 1.11 | 0.000 | 0.000 | 0.9443 | 0.03 |
| zipf-heavy | 0.05 | global-none | 1.0950 | 1.0869 | 1.2192 | 1.32 | 0.000 | 0.000 | 0.9903 | 0.00 |
| zipf-heavy | 0.05 | global-realized | 1.1143 | 1.1157 | 1.1623 | 1.29 | 0.000 | 0.000 | 0.9903 | 0.00 |
| zipf-heavy | 0.05 | cf-realized | 1.1143 | 1.1158 | 1.1630 | 1.29 | 0.000 | 0.000 | 0.9903 | 0.00 |
| zipf-heavy | 0.05 | split | 1.1072 | 1.1073 | 1.1202 | 1.12 | 0.000 | 0.000 | 0.9457 | 4.09 |
| zipf-heavy | 0.05 | replicate | 1.1069 | 1.1067 | 1.1119 | 1.12 | 0.000 | 0.000 | 0.9519 | 1.00 |
| zipf-heavy | 0.05 | fec | 1.1070 | 1.1067 | 1.1109 | 1.13 | 0.000 | 0.000 | 0.9443 | 0.10 |
| zipf-heavy | 0.1 | global-none | 1.1517 | 1.1410 | 1.3819 | 1.43 | 0.000 | 0.000 | 1.0641 | 0.00 |
| zipf-heavy | 0.1 | global-realized | 1.1254 | 1.1252 | 1.2440 | 1.29 | 0.000 | 0.000 | 1.0641 | 0.00 |
| zipf-heavy | 0.1 | cf-realized | 1.1255 | 1.1256 | 1.2449 | 1.30 | 0.000 | 0.000 | 1.0641 | 0.00 |
| zipf-heavy | 0.1 | split | 1.1085 | 1.1080 | 1.1295 | 1.13 | 0.000 | 0.000 | 0.9492 | 4.09 |
| zipf-heavy | 0.1 | replicate | 1.1079 | 1.1083 | 1.1162 | 1.12 | 0.000 | 0.000 | 0.9547 | 1.00 |
| zipf-heavy | 0.1 | fec | 1.1071 | 1.1067 | 1.1110 | 1.15 | 0.000 | 0.000 | 0.9443 | 0.18 |
| zipf-heavy | 0.2 | global-none | 1.2968 | 1.2741 | 1.5706 | 1.94 | 0.000 | 0.000 | 1.1044 | 0.00 |
| zipf-heavy | 0.2 | global-realized | 1.1462 | 1.1398 | 1.2784 | 1.60 | 0.000 | 0.000 | 1.1044 | 0.00 |
| zipf-heavy | 0.2 | cf-realized | 1.1465 | 1.1403 | 1.2807 | 1.61 | 0.000 | 0.000 | 1.1044 | 0.00 |
| zipf-heavy | 0.2 | split | 1.1121 | 1.1122 | 1.1400 | 1.14 | 0.000 | 0.000 | 0.9493 | 4.09 |
| zipf-heavy | 0.2 | replicate | 1.1091 | 1.1080 | 1.1188 | 1.22 | 0.000 | 0.000 | 0.9565 | 2.00 |
| zipf-heavy | 0.2 | fec | 1.1072 | 1.1067 | 1.1259 | 1.14 | 0.000 | 0.000 | 0.9641 | 0.34 |
| zipf-heavy | 0.3 | global-none | 1.5140 | 1.4899 | 1.8652 | 2.07 | 0.010 | 0.000 | 1.3505 | 0.00 |
| zipf-heavy | 0.3 | global-realized | 1.1873 | 1.1783 | 1.4343 | 1.48 | 0.000 | 0.000 | 1.3505 | 0.00 |
| zipf-heavy | 0.3 | cf-realized | 1.1878 | 1.1788 | 1.4382 | 1.48 | 0.000 | 0.000 | 1.3505 | 0.00 |
| zipf-heavy | 0.3 | split | 1.1137 | 1.1130 | 1.1558 | 1.16 | 0.000 | 0.000 | 0.9586 | 4.09 |
| zipf-heavy | 0.3 | replicate | 1.1077 | 1.1080 | 1.1148 | 1.13 | 0.000 | 0.000 | 0.9584 | 3.00 |
| zipf-heavy | 0.3 | fec | 1.1069 | 1.1065 | 1.1109 | 1.11 | 0.000 | 0.000 | 0.9443 | 0.56 |
| zipf-heavy | 0.5 | global-none | 2.0650 | 2.0260 | 2.7561 | 2.84 | 0.575 | 0.000 | 1.7696 | 0.00 |
| zipf-heavy | 0.5 | global-realized | 1.2667 | 1.2503 | 1.5750 | 1.65 | 0.000 | 0.000 | 1.7696 | 0.00 |
| zipf-heavy | 0.5 | cf-realized | 1.2673 | 1.2502 | 1.5755 | 1.66 | 0.000 | 0.000 | 1.7696 | 0.00 |
| zipf-heavy | 0.5 | split | 1.1225 | 1.1219 | 1.1917 | 1.22 | 0.000 | 0.000 | 0.9548 | 4.09 |
| zipf-heavy | 0.5 | replicate | 1.1075 | 1.1077 | 1.1146 | 1.12 | 0.000 | 0.000 | 0.9544 | 6.00 |
| zipf-heavy | 0.5 | fec | 1.1069 | 1.1065 | 1.1109 | 1.11 | 0.000 | 0.000 | 0.9443 | 1.25 |



## F4: stragglers with coding (n = 16, m = 500)

| partition | s | stragglers | scheme | distortion mean | p90 | downstream mean | max | received points | received bytes | sent bytes | max node load | units lost | draw fraction |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 1 | none | uncoded | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 1525652 | 1525652 | 12500 | 0.00 | 1.000 |
| random | 1 | none | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 3051304 | 3051304 | 25000 | 0.00 | 1.000 |
| random | 1 | none | fr | 1.0537 | 1.0764 | 1.0038 | 1.005 | 3886 | 784891 | 1569782 | 25000 | 0.00 | 1.000 |
| random | 1 | none | rateless-mu1 | 1.0474 | 1.0672 | 1.0037 | 1.004 | 3887 | 785120 | 1569399 | 25000 | 0.00 | 1.000 |
| random | 1 | none | rateless-mu2 | 1.0587 | 1.0786 | 1.0061 | 1.008 | 2454 | 500388 | 800600 | 50000 | 0.00 | 1.000 |
| random | 1 | none | rateless-mu4 | 1.0825 | 1.1100 | 1.0103 | 1.013 | 1483 | 302111 | 402795 | 75000 | 0.00 | 1.000 |
| random | 1 | none | offline-g | 1.0444 | 1.0582 | 1.0038 | 1.005 | 3881 | 784009 | 784009 | 25000 | 0.00 | 1.000 |
| random | 1 | random-s | uncoded | 1.0353 | 1.0469 | 1.0020 | 1.002 | 7082 | 1430598 | 1430598 | 12500 | 1.00 | 1.000 |
| random | 1 | random-s | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 2860946 | 2860946 | 25000 | 0.00 | 1.000 |
| random | 1 | random-s | fr | 1.0537 | 1.0764 | 1.0038 | 1.005 | 3886 | 784891 | 1471718 | 25000 | 0.00 | 1.000 |
| random | 1 | random-s | rateless-mu1 | 1.0504 | 1.0631 | 1.0037 | 1.005 | 3886 | 785060 | 1471233 | 25000 | 0.00 | 1.000 |
| random | 1 | random-s | rateless-mu2 | 1.0578 | 1.0750 | 1.0061 | 1.008 | 2454 | 500341 | 750544 | 50000 | 0.00 | 1.000 |
| random | 1 | random-s | rateless-mu4 | 1.0814 | 1.1099 | 1.0101 | 1.013 | 1483 | 302084 | 377592 | 75000 | 0.00 | 1.000 |
| random | 1 | random-s+1 | uncoded | 1.0372 | 1.0543 | 1.0022 | 1.003 | 6608 | 1334904 | 1334904 | 12500 | 2.00 | 1.000 |
| random | 1 | random-s+1 | replicate | 1.0331 | 1.0481 | 1.0019 | 1.002 | 7506 | 2669356 | 2669356 | 25000 | 0.10 | 1.000 |
| random | 1 | random-s+1 | fr | 1.0536 | 1.0764 | 1.0039 | 1.005 | 3853 | 778299 | 1373600 | 25000 | 0.07 | 1.000 |
| random | 1 | random-s+1 | rateless-mu1 | 1.0493 | 1.0677 | 1.0037 | 1.005 | 3854 | 778481 | 1373196 | 25000 | 0.07 | 0.992 |
| random | 1 | random-s+1 | rateless-mu2 | 1.0655 | 1.0908 | 1.0062 | 1.009 | 2438 | 497014 | 700523 | 50000 | 0.00 | 0.993 |
| random | 1 | random-s+1 | rateless-mu4 | 1.0893 | 1.1303 | 1.0102 | 1.012 | 1471 | 299499 | 352409 | 75000 | 0.00 | 0.992 |
| random | 1 | random-s+2 | uncoded | 1.0369 | 1.0537 | 1.0024 | 1.003 | 6136 | 1239445 | 1239445 | 12500 | 3.00 | 1.000 |
| random | 1 | random-s+2 | replicate | 1.0341 | 1.0484 | 1.0019 | 1.002 | 7395 | 2479301 | 2479301 | 25000 | 0.33 | 1.000 |
| random | 1 | random-s+2 | fr | 1.0540 | 1.0805 | 1.0039 | 1.005 | 3805 | 768549 | 1275401 | 25000 | 0.17 | 1.000 |
| random | 1 | random-s+2 | rateless-mu1 | 1.0469 | 1.0712 | 1.0038 | 1.005 | 3807 | 769007 | 1275576 | 25000 | 0.17 | 0.979 |
| random | 1 | random-s+2 | rateless-mu2 | 1.0656 | 1.0804 | 1.0063 | 1.008 | 2398 | 488739 | 650528 | 50000 | 0.00 | 0.977 |
| random | 1 | random-s+2 | rateless-mu4 | 1.0844 | 1.1021 | 1.0107 | 1.013 | 1418 | 288658 | 327253 | 75000 | 0.00 | 0.956 |
| random | 1 | adversarial-s | uncoded | 1.0338 | 1.0491 | 1.0020 | 1.002 | 7078 | 1429749 | 1429749 | 12500 | 1.00 | 1.000 |
| random | 1 | adversarial-s | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 2860212 | 2860212 | 25000 | 0.00 | 1.000 |
| random | 1 | adversarial-s | fr | 1.0537 | 1.0764 | 1.0038 | 1.005 | 3886 | 784891 | 1471523 | 25000 | 0.00 | 1.000 |
| random | 1 | adversarial-s | rateless-mu1 | 1.0496 | 1.0699 | 1.0037 | 1.004 | 3887 | 785241 | 1471395 | 25000 | 0.00 | 1.000 |
| random | 1 | adversarial-s | rateless-mu2 | 1.0624 | 1.0809 | 1.0061 | 1.008 | 2454 | 500361 | 750430 | 50000 | 0.00 | 1.000 |
| random | 1 | adversarial-s | rateless-mu4 | 1.0879 | 1.1161 | 1.0101 | 1.012 | 1484 | 302111 | 377605 | 75000 | 0.00 | 1.000 |
| random | 1 | adversarial-s+1 | uncoded | 1.0366 | 1.0506 | 1.0022 | 1.003 | 6607 | 1334520 | 1334520 | 12500 | 2.00 | 1.000 |
| random | 1 | adversarial-s+1 | replicate | 1.0338 | 1.0481 | 1.0019 | 1.002 | 7458 | 2669167 | 2669167 | 25000 | 0.20 | 1.000 |
| random | 1 | adversarial-s+1 | fr | 1.0555 | 1.0814 | 1.0044 | 1.005 | 3399 | 686632 | 1373263 | 25000 | 1.00 | 1.000 |
| random | 1 | adversarial-s+1 | rateless-mu1 | 1.0529 | 1.0796 | 1.0042 | 1.005 | 3402 | 687123 | 1373277 | 25000 | 1.00 | 0.875 |
| random | 1 | adversarial-s+1 | rateless-mu2 | 1.0598 | 1.0807 | 1.0061 | 1.008 | 2455 | 500415 | 700334 | 50000 | 0.00 | 1.000 |
| random | 1 | adversarial-s+1 | rateless-mu4 | 1.0877 | 1.1166 | 1.0102 | 1.012 | 1483 | 302071 | 352402 | 75000 | 0.00 | 1.000 |
| random | 2 | none | uncoded | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 1525652 | 1525652 | 12500 | 0.00 | 1.000 |
| random | 2 | none | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 4576956 | 4576956 | 37500 | 0.00 | 1.000 |
| random | 2 | none | fr | 1.0597 | 1.0813 | 1.0063 | 1.009 | 2453 | 495452 | 1585942 | 50000 | 0.00 | 1.000 |
| random | 2 | none | rateless-mu1 | 1.0667 | 1.0881 | 1.0063 | 1.008 | 2454 | 495627 | 1585767 | 50000 | 0.00 | 1.000 |
| random | 2 | none | rateless-mu2 | 1.0653 | 1.0934 | 1.0073 | 1.009 | 1972 | 401172 | 802081 | 50000 | 0.00 | 1.000 |
| random | 2 | none | rateless-mu4 | 1.1002 | 1.1360 | 1.0257 | 1.334 | 992 | 201616 | 403239 | 100000 | 0.00 | 1.000 |
| random | 2 | none | offline-g | 1.0590 | 1.0898 | 1.0061 | 1.007 | 2455 | 495950 | 495950 | 40000 | 0.00 | 1.000 |
| random | 2 | random-s | uncoded | 1.0370 | 1.0480 | 1.0021 | 1.002 | 6608 | 1334850 | 1334850 | 12500 | 2.00 | 1.000 |
| random | 2 | random-s | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 4005209 | 4005209 | 37500 | 0.00 | 1.000 |
| random | 2 | random-s | fr | 1.0597 | 1.0813 | 1.0063 | 1.009 | 2453 | 495452 | 1387619 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s | rateless-mu1 | 1.0672 | 1.0906 | 1.0062 | 1.007 | 2453 | 495459 | 1387262 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s | rateless-mu2 | 1.0666 | 1.0991 | 1.0074 | 1.009 | 1971 | 401185 | 701896 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s | rateless-mu4 | 1.0960 | 1.1215 | 1.0145 | 1.018 | 992 | 201589 | 352820 | 100000 | 0.00 | 1.000 |
| random | 2 | random-s+1 | uncoded | 1.0346 | 1.0467 | 1.0024 | 1.003 | 6133 | 1238792 | 1238792 | 12500 | 3.00 | 1.000 |
| random | 2 | random-s+1 | replicate | 1.0335 | 1.0481 | 1.0019 | 1.002 | 7537 | 3718315 | 3718315 | 37500 | 0.03 | 1.000 |
| random | 2 | random-s+1 | fr | 1.0597 | 1.0813 | 1.0063 | 1.009 | 2453 | 495452 | 1288504 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s+1 | rateless-mu1 | 1.0605 | 1.0710 | 1.0064 | 1.008 | 2454 | 495674 | 1288154 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s+1 | rateless-mu2 | 1.0676 | 1.0898 | 1.0075 | 1.010 | 1971 | 401058 | 651571 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s+1 | rateless-mu4 | 1.1106 | 1.1574 | 1.0151 | 1.019 | 992 | 201583 | 327610 | 100000 | 0.00 | 1.000 |
| random | 2 | random-s+2 | uncoded | 1.0413 | 1.0596 | 1.0025 | 1.003 | 5665 | 1144418 | 1144418 | 12500 | 4.00 | 1.000 |
| random | 2 | random-s+2 | replicate | 1.0336 | 1.0481 | 1.0019 | 1.002 | 7521 | 3433051 | 3433051 | 37500 | 0.07 | 1.000 |
| random | 2 | random-s+2 | fr | 1.0597 | 1.0813 | 1.0063 | 1.009 | 2453 | 495452 | 1189235 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s+2 | rateless-mu1 | 1.0648 | 1.0864 | 1.0062 | 1.008 | 2454 | 495641 | 1189490 | 50000 | 0.00 | 1.000 |
| random | 2 | random-s+2 | rateless-mu2 | 1.0653 | 1.0986 | 1.0078 | 1.011 | 1955 | 397718 | 601468 | 50000 | 0.00 | 0.992 |
| random | 2 | random-s+2 | rateless-mu4 | 1.1040 | 1.1415 | 1.0150 | 1.018 | 993 | 201596 | 302448 | 100000 | 0.00 | 1.000 |
| random | 2 | adversarial-s | uncoded | 1.0366 | 1.0506 | 1.0022 | 1.003 | 6607 | 1334520 | 1334520 | 12500 | 2.00 | 1.000 |
| random | 2 | adversarial-s | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 4003963 | 4003963 | 37500 | 0.00 | 1.000 |
| random | 2 | adversarial-s | fr | 1.0597 | 1.0813 | 1.0063 | 1.009 | 2453 | 495452 | 1386770 | 50000 | 0.00 | 1.000 |
| random | 2 | adversarial-s | rateless-mu1 | 1.0675 | 1.0881 | 1.0063 | 1.008 | 2454 | 495634 | 1386750 | 50000 | 0.00 | 1.000 |
| random | 2 | adversarial-s | rateless-mu2 | 1.0636 | 1.0861 | 1.0075 | 1.009 | 1970 | 400990 | 701768 | 50000 | 0.00 | 1.000 |
| random | 2 | adversarial-s | rateless-mu4 | 1.1048 | 1.1439 | 1.0256 | 1.334 | 993 | 201636 | 352840 | 100000 | 0.00 | 1.000 |
| random | 2 | adversarial-s+1 | uncoded | 1.0372 | 1.0464 | 1.0024 | 1.003 | 6136 | 1239411 | 1239411 | 12500 | 3.00 | 1.000 |
| random | 2 | adversarial-s+1 | replicate | 1.0339 | 1.0481 | 1.0019 | 1.002 | 7537 | 3718214 | 3718214 | 37500 | 0.03 | 1.000 |
| random | 2 | adversarial-s+1 | fr | 1.0597 | 1.0813 | 1.0063 | 1.009 | 2453 | 495452 | 1287184 | 50000 | 0.00 | 1.000 |
| random | 2 | adversarial-s+1 | rateless-mu1 | 1.0686 | 1.0901 | 1.0061 | 1.007 | 2454 | 495742 | 1287373 | 50000 | 0.00 | 1.000 |
| random | 2 | adversarial-s+1 | rateless-mu2 | 1.0711 | 1.0987 | 1.0094 | 1.012 | 1727 | 350955 | 651733 | 50000 | 0.00 | 0.875 |
| random | 2 | adversarial-s+1 | rateless-mu4 | 1.0987 | 1.1251 | 1.0151 | 1.019 | 993 | 201636 | 327644 | 100000 | 0.00 | 1.000 |
| random | 3 | none | uncoded | 1.0337 | 1.0481 | 1.0154 | 1.409 | 7553 | 1525652 | 1525652 | 12500 | 0.00 | 1.000 |
| random | 3 | none | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 6102609 | 6102609 | 50000 | 0.00 | 1.000 |
| random | 3 | none | fr | 1.0678 | 1.0939 | 1.0074 | 1.009 | 1971 | 398122 | 1592487 | 50000 | 0.00 | 1.000 |
| random | 3 | none | rateless-mu1 | 1.0733 | 1.0998 | 1.0215 | 1.424 | 1972 | 398250 | 1593235 | 50000 | 0.00 | 1.000 |
| random | 3 | none | rateless-mu2 | 1.0883 | 1.1102 | 1.0103 | 1.014 | 1483 | 301182 | 803401 | 75000 | 0.00 | 1.000 |
| random | 3 | none | rateless-mu4 | 1.0959 | 1.1388 | 1.0285 | 1.423 | 993 | 201717 | 403360 | 100000 | 0.00 | 1.000 |
| random | 3 | none | offline-g | 1.0766 | 1.1145 | 1.0077 | 1.010 | 1970 | 398001 | 398001 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s | uncoded | 1.0429 | 1.0596 | 1.0023 | 1.003 | 6135 | 1239310 | 1239310 | 12500 | 3.00 | 1.000 |
| random | 3 | random-s | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 4957329 | 4957329 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s | fr | 1.0678 | 1.0939 | 1.0074 | 1.009 | 1971 | 398122 | 1293891 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s | rateless-mu1 | 1.0695 | 1.0977 | 1.0075 | 1.010 | 1971 | 398182 | 1294429 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s | rateless-mu2 | 1.0888 | 1.1270 | 1.0101 | 1.012 | 1484 | 301256 | 652783 | 75000 | 0.00 | 1.000 |
| random | 3 | random-s | rateless-mu4 | 1.1011 | 1.1531 | 1.0152 | 1.022 | 993 | 201704 | 327711 | 100000 | 0.00 | 1.000 |
| random | 3 | random-s+1 | uncoded | 1.0429 | 1.0619 | 1.0025 | 1.003 | 5667 | 1144768 | 1144768 | 12500 | 4.00 | 1.000 |
| random | 3 | random-s+1 | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7537 | 4578202 | 4578202 | 50000 | 0.03 | 1.000 |
| random | 3 | random-s+1 | fr | 1.0678 | 1.0939 | 1.0074 | 1.009 | 1971 | 398122 | 1194453 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s+1 | rateless-mu1 | 1.0760 | 1.1027 | 1.0350 | 1.424 | 1972 | 398310 | 1194830 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s+1 | rateless-mu2 | 1.0815 | 1.1167 | 1.0102 | 1.013 | 1483 | 301243 | 602546 | 75000 | 0.00 | 1.000 |
| random | 3 | random-s+1 | rateless-mu4 | 1.0925 | 1.1211 | 1.0291 | 1.428 | 993 | 201717 | 302529 | 100000 | 0.00 | 1.000 |
| random | 3 | random-s+2 | uncoded | 1.0439 | 1.0578 | 1.0028 | 1.003 | 5194 | 1049094 | 1049094 | 12500 | 5.00 | 1.000 |
| random | 3 | random-s+2 | replicate | 1.0336 | 1.0463 | 1.0019 | 1.002 | 7521 | 4195681 | 4195681 | 50000 | 0.07 | 1.000 |
| random | 3 | random-s+2 | fr | 1.0678 | 1.0939 | 1.0074 | 1.009 | 1971 | 398122 | 1094880 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s+2 | rateless-mu1 | 1.0713 | 1.0924 | 1.0076 | 1.009 | 1972 | 398357 | 1095554 | 50000 | 0.00 | 1.000 |
| random | 3 | random-s+2 | rateless-mu2 | 1.0905 | 1.1275 | 1.0102 | 1.013 | 1475 | 299526 | 552335 | 75000 | 0.00 | 0.994 |
| random | 3 | random-s+2 | rateless-mu4 | 1.0930 | 1.1407 | 1.0148 | 1.018 | 993 | 201697 | 277299 | 100000 | 0.00 | 1.000 |
| random | 3 | adversarial-s | uncoded | 1.0372 | 1.0464 | 1.0024 | 1.003 | 6136 | 1239411 | 1239411 | 12500 | 3.00 | 1.000 |
| random | 3 | adversarial-s | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 4958460 | 4958460 | 50000 | 0.00 | 1.000 |
| random | 3 | adversarial-s | fr | 1.0678 | 1.0939 | 1.0074 | 1.009 | 1971 | 398122 | 1293285 | 50000 | 0.00 | 1.000 |
| random | 3 | adversarial-s | rateless-mu1 | 1.0701 | 1.0988 | 1.0216 | 1.424 | 1972 | 398283 | 1294382 | 50000 | 0.00 | 1.000 |
| random | 3 | adversarial-s | rateless-mu2 | 1.0807 | 1.1094 | 1.0101 | 1.014 | 1483 | 301115 | 652729 | 75000 | 0.00 | 1.000 |
| random | 3 | adversarial-s | rateless-mu4 | 1.0948 | 1.1234 | 1.0286 | 1.425 | 993 | 201717 | 327732 | 100000 | 0.00 | 1.000 |
| random | 3 | adversarial-s+1 | uncoded | 1.0393 | 1.0503 | 1.0025 | 1.003 | 5664 | 1144027 | 1144027 | 12500 | 4.00 | 1.000 |
| random | 3 | adversarial-s+1 | replicate | 1.0337 | 1.0481 | 1.0019 | 1.002 | 7553 | 4576714 | 4576714 | 50000 | 0.00 | 1.000 |
| random | 3 | adversarial-s+1 | fr | 1.0767 | 1.1082 | 1.0102 | 1.012 | 1477 | 298388 | 1193551 | 50000 | 1.00 | 1.000 |
| random | 3 | adversarial-s+1 | rateless-mu1 | 1.0803 | 1.1098 | 1.0102 | 1.013 | 1478 | 298610 | 1194709 | 50000 | 1.00 | 0.750 |
| random | 3 | adversarial-s+1 | rateless-mu2 | 1.0851 | 1.1162 | 1.0101 | 1.014 | 1483 | 301189 | 602499 | 75000 | 0.00 | 1.000 |
| random | 3 | adversarial-s+1 | rateless-mu4 | 1.0928 | 1.1134 | 1.0153 | 1.020 | 993 | 201690 | 302522 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 1 | none | uncoded | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 1542445 | 1542445 | 12500 | 0.00 | 1.000 |
| cluster-sorted | 1 | none | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 3084890 | 3084890 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | none | fr | 1.0630 | 1.0939 | 1.0200 | 1.427 | 3910 | 789901 | 1579802 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | none | rateless-mu1 | 1.0593 | 1.0881 | 1.0058 | 1.007 | 3909 | 789645 | 1579014 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | none | rateless-mu2 | 1.0756 | 1.0967 | 1.0087 | 1.011 | 2464 | 501377 | 802001 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 1 | none | rateless-mu4 | 1.0746 | 1.1000 | 1.0126 | 1.016 | 1486 | 302286 | 403057 | 75000 | 0.00 | 1.000 |
| cluster-sorted | 1 | none | offline-g | 1.0645 | 1.0882 | 1.0060 | 1.008 | 3904 | 788696 | 788696 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | random-s | uncoded | 1.1539 | 1.3301 | 1.8773 | 7.464 | 7159 | 1446185 | 1446185 | 12500 | 1.00 | 1.000 |
| cluster-sorted | 1 | random-s | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 2892162 | 2892162 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | random-s | fr | 1.0630 | 1.0939 | 1.0200 | 1.427 | 3910 | 789901 | 1481246 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | random-s | rateless-mu1 | 1.0603 | 1.0935 | 1.0058 | 1.007 | 3909 | 789645 | 1480168 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | random-s | rateless-mu2 | 1.0745 | 1.0953 | 1.0087 | 1.012 | 2463 | 501398 | 751932 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 1 | random-s | rateless-mu4 | 1.0799 | 1.1004 | 1.0125 | 1.016 | 1486 | 302266 | 377881 | 75000 | 0.00 | 1.000 |
| cluster-sorted | 1 | random-s+1 | uncoded | 1.2447 | 1.3898 | 3.2219 | 13.721 | 6683 | 1350060 | 1350060 | 12500 | 2.00 | 1.000 |
| cluster-sorted | 1 | random-s+1 | replicate | 1.0710 | 1.1217 | 1.3214 | 6.762 | 7541 | 2699407 | 2699407 | 25000 | 0.20 | 1.000 |
| cluster-sorted | 1 | random-s+1 | fr | 1.1020 | 1.1561 | 1.4690 | 13.702 | 3845 | 776643 | 1382299 | 25000 | 0.13 | 1.000 |
| cluster-sorted | 1 | random-s+1 | rateless-mu1 | 1.1026 | 1.1740 | 1.4693 | 13.714 | 3843 | 776225 | 1381693 | 25000 | 0.13 | 0.983 |
| cluster-sorted | 1 | random-s+1 | rateless-mu2 | 1.0778 | 1.1190 | 1.0087 | 1.012 | 2439 | 496368 | 701788 | 50000 | 0.00 | 0.990 |
| cluster-sorted | 1 | random-s+1 | rateless-mu4 | 1.0801 | 1.1102 | 1.0125 | 1.016 | 1461 | 297250 | 352672 | 75000 | 0.00 | 0.983 |
| cluster-sorted | 1 | random-s+2 | uncoded | 1.3577 | 1.6284 | 4.3040 | 13.454 | 6202 | 1252710 | 1252710 | 12500 | 3.00 | 1.000 |
| cluster-sorted | 1 | random-s+2 | replicate | 1.1094 | 1.2187 | 1.2222 | 5.699 | 7430 | 2506658 | 2506658 | 25000 | 0.43 | 1.000 |
| cluster-sorted | 1 | random-s+2 | fr | 1.0938 | 1.1497 | 1.5092 | 9.249 | 3829 | 773458 | 1283319 | 25000 | 0.17 | 1.000 |
| cluster-sorted | 1 | random-s+2 | rateless-mu1 | 1.0894 | 1.1515 | 1.4948 | 9.247 | 3827 | 773121 | 1282801 | 25000 | 0.17 | 0.979 |
| cluster-sorted | 1 | random-s+2 | rateless-mu2 | 1.0744 | 1.1009 | 1.0239 | 1.440 | 2365 | 481204 | 651665 | 50000 | 0.00 | 0.960 |
| cluster-sorted | 1 | random-s+2 | rateless-mu4 | 1.0804 | 1.1066 | 1.0134 | 1.018 | 1407 | 286315 | 327469 | 75000 | 0.00 | 0.947 |
| cluster-sorted | 1 | adversarial-s | uncoded | 1.3626 | 1.6341 | 4.6783 | 7.562 | 7156 | 1445431 | 1445431 | 12500 | 1.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 2891186 | 2891186 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s | fr | 1.0630 | 1.0939 | 1.0200 | 1.427 | 3910 | 789901 | 1481145 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s | rateless-mu1 | 1.0610 | 1.0923 | 1.0059 | 1.007 | 3909 | 789706 | 1479933 | 25000 | 0.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s | rateless-mu2 | 1.0744 | 1.1143 | 1.0087 | 1.011 | 2463 | 501297 | 751837 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s | rateless-mu4 | 1.0834 | 1.1178 | 1.0126 | 1.017 | 1486 | 302286 | 377854 | 75000 | 0.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s+1 | uncoded | 1.5439 | 1.8696 | 8.2971 | 16.306 | 6677 | 1348767 | 1348767 | 12500 | 2.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s+1 | replicate | 1.2274 | 1.6341 | 3.3282 | 8.628 | 7395 | 2697481 | 2697481 | 25000 | 0.50 | 1.000 |
| cluster-sorted | 1 | adversarial-s+1 | fr | 1.4727 | 1.7138 | 9.6854 | 16.966 | 3422 | 691244 | 1382488 | 25000 | 1.00 | 1.000 |
| cluster-sorted | 1 | adversarial-s+1 | rateless-mu1 | 1.4658 | 1.7299 | 9.6910 | 16.893 | 3419 | 690651 | 1380879 | 25000 | 1.00 | 0.875 |
| cluster-sorted | 1 | adversarial-s+1 | rateless-mu2 | 1.0836 | 1.1166 | 1.0106 | 1.016 | 2325 | 472949 | 701775 | 50000 | 0.00 | 0.943 |
| cluster-sorted | 1 | adversarial-s+1 | rateless-mu4 | 1.0855 | 1.1189 | 1.0134 | 1.017 | 1428 | 290530 | 352652 | 75000 | 0.00 | 0.961 |
| cluster-sorted | 2 | none | uncoded | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 1542445 | 1542445 | 12500 | 0.00 | 1.000 |
| cluster-sorted | 2 | none | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 4627335 | 4627335 | 37500 | 0.00 | 1.000 |
| cluster-sorted | 2 | none | fr | 1.0843 | 1.1135 | 1.0089 | 1.012 | 2464 | 497735 | 1592992 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | none | rateless-mu1 | 1.0847 | 1.1122 | 1.0086 | 1.011 | 2463 | 497425 | 1592157 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | none | rateless-mu2 | 1.0706 | 1.1136 | 1.0252 | 1.457 | 1974 | 401381 | 802526 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | none | rateless-mu4 | 1.0932 | 1.1249 | 1.0170 | 1.021 | 993 | 201690 | 403441 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 2 | none | offline-g | 1.0657 | 1.0839 | 1.0085 | 1.010 | 2463 | 497452 | 497452 | 40000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s | uncoded | 1.2500 | 1.3614 | 3.6290 | 8.101 | 6683 | 1350033 | 1350033 | 12500 | 2.00 | 1.000 |
| cluster-sorted | 2 | random-s | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 4048693 | 4048693 | 37500 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s | fr | 1.0843 | 1.1135 | 1.0089 | 1.012 | 2464 | 497735 | 1394029 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s | rateless-mu1 | 1.0816 | 1.1071 | 1.0086 | 1.011 | 2463 | 497553 | 1393349 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s | rateless-mu2 | 1.0742 | 1.1206 | 1.0102 | 1.012 | 1975 | 401408 | 702260 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s | rateless-mu4 | 1.0958 | 1.1358 | 1.0171 | 1.021 | 994 | 201731 | 353015 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s+1 | uncoded | 1.3259 | 1.5538 | 4.6685 | 13.247 | 6205 | 1253376 | 1253376 | 12500 | 3.00 | 1.000 |
| cluster-sorted | 2 | random-s+1 | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 3760048 | 3760048 | 37500 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s+1 | fr | 1.0843 | 1.1135 | 1.0089 | 1.012 | 2464 | 497735 | 1294524 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s+1 | rateless-mu1 | 1.0850 | 1.1164 | 1.0237 | 1.456 | 2463 | 497553 | 1293534 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s+1 | rateless-mu2 | 1.0742 | 1.1196 | 1.0103 | 1.013 | 1949 | 396311 | 651968 | 50000 | 0.00 | 0.988 |
| cluster-sorted | 2 | random-s+1 | rateless-mu4 | 1.0935 | 1.1187 | 1.0167 | 1.021 | 993 | 201751 | 327819 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 2 | random-s+2 | uncoded | 1.4193 | 1.5991 | 6.7608 | 22.184 | 5730 | 1157366 | 1157366 | 12500 | 4.00 | 1.000 |
| cluster-sorted | 2 | random-s+2 | replicate | 1.0606 | 1.0907 | 1.2201 | 5.385 | 7525 | 3471168 | 3471168 | 37500 | 0.23 | 1.000 |
| cluster-sorted | 2 | random-s+2 | fr | 1.0940 | 1.1306 | 1.0587 | 2.498 | 2431 | 491116 | 1194965 | 50000 | 0.07 | 1.000 |
| cluster-sorted | 2 | random-s+2 | rateless-mu1 | 1.0917 | 1.1330 | 1.0586 | 2.502 | 2430 | 490766 | 1193732 | 50000 | 0.07 | 0.987 |
| cluster-sorted | 2 | random-s+2 | rateless-mu2 | 1.0769 | 1.1031 | 1.0240 | 1.416 | 1917 | 389658 | 601953 | 50000 | 0.00 | 0.971 |
| cluster-sorted | 2 | random-s+2 | rateless-mu4 | 1.1006 | 1.1414 | 1.0169 | 1.024 | 994 | 201697 | 302569 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s | uncoded | 1.5439 | 1.8696 | 8.2976 | 16.312 | 6677 | 1348767 | 1348767 | 12500 | 2.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 4046989 | 4046989 | 37500 | 0.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s | fr | 1.0843 | 1.1135 | 1.0089 | 1.012 | 2464 | 497735 | 1393389 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s | rateless-mu1 | 1.0805 | 1.1186 | 1.0087 | 1.011 | 2462 | 497230 | 1392931 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s | rateless-mu2 | 1.0754 | 1.1135 | 1.0102 | 1.012 | 1974 | 401266 | 702152 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s | rateless-mu4 | 1.1013 | 1.1355 | 1.0170 | 1.024 | 994 | 201717 | 353015 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s+1 | uncoded | 1.8538 | 2.4650 | 12.7620 | 23.009 | 6200 | 1252481 | 1252481 | 12500 | 3.00 | 1.000 |
| cluster-sorted | 2 | adversarial-s+1 | replicate | 1.2361 | 1.6341 | 3.4502 | 8.628 | 7379 | 3757133 | 3757133 | 37500 | 0.53 | 1.000 |
| cluster-sorted | 2 | adversarial-s+1 | fr | 1.4819 | 2.1150 | 9.2350 | 24.673 | 2184 | 441235 | 1293588 | 50000 | 0.57 | 1.000 |
| cluster-sorted | 2 | adversarial-s+1 | rateless-mu1 | 1.4857 | 2.1572 | 9.2420 | 24.740 | 2183 | 440865 | 1293480 | 50000 | 0.57 | 0.887 |
| cluster-sorted | 2 | adversarial-s+1 | rateless-mu2 | 1.0835 | 1.1065 | 1.0146 | 1.021 | 1729 | 351177 | 652063 | 50000 | 0.00 | 0.875 |
| cluster-sorted | 2 | adversarial-s+1 | rateless-mu4 | 1.1000 | 1.1303 | 1.0167 | 1.023 | 994 | 201724 | 327785 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 3 | none | uncoded | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 1542445 | 1542445 | 12500 | 0.00 | 1.000 |
| cluster-sorted | 3 | none | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 6169780 | 6169780 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | none | fr | 1.0820 | 1.1253 | 1.0099 | 1.013 | 1979 | 399664 | 1598655 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | none | rateless-mu1 | 1.0809 | 1.1169 | 1.0100 | 1.012 | 1977 | 399367 | 1596265 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | none | rateless-mu2 | 1.0847 | 1.1128 | 1.0126 | 1.016 | 1486 | 301613 | 804445 | 75000 | 0.00 | 1.000 |
| cluster-sorted | 3 | none | rateless-mu4 | 1.1116 | 1.1687 | 1.0169 | 1.024 | 993 | 201670 | 403374 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 3 | none | offline-g | 1.0802 | 1.1164 | 1.0097 | 1.012 | 1977 | 399361 | 399361 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s | uncoded | 1.3076 | 1.5100 | 4.5378 | 13.286 | 6205 | 1253376 | 1253376 | 12500 | 3.00 | 1.000 |
| cluster-sorted | 3 | random-s | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 5012502 | 5012502 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s | fr | 1.0820 | 1.1253 | 1.0099 | 1.013 | 1979 | 399664 | 1298698 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s | rateless-mu1 | 1.0808 | 1.1179 | 1.0100 | 1.012 | 1977 | 399327 | 1296948 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s | rateless-mu2 | 1.0850 | 1.1128 | 1.0126 | 1.016 | 1486 | 301593 | 653611 | 75000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s | rateless-mu4 | 1.1022 | 1.1377 | 1.0166 | 1.021 | 993 | 201697 | 327758 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+1 | uncoded | 1.3795 | 1.5525 | 6.7023 | 17.647 | 5726 | 1156639 | 1156639 | 12500 | 4.00 | 1.000 |
| cluster-sorted | 3 | random-s+1 | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 4626581 | 4626581 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+1 | fr | 1.0820 | 1.1253 | 1.0099 | 1.013 | 1979 | 399664 | 1198776 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+1 | rateless-mu1 | 1.0812 | 1.1089 | 1.0100 | 1.013 | 1977 | 399374 | 1197651 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+1 | rateless-mu2 | 1.0848 | 1.1198 | 1.0127 | 1.016 | 1486 | 301566 | 603233 | 75000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+1 | rateless-mu4 | 1.1169 | 1.1874 | 1.0308 | 1.442 | 994 | 201643 | 302502 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+2 | uncoded | 1.4420 | 1.7041 | 8.7792 | 21.967 | 5252 | 1060857 | 1060857 | 12500 | 5.00 | 1.000 |
| cluster-sorted | 3 | random-s+2 | replicate | 1.0651 | 1.0935 | 1.2023 | 6.972 | 7588 | 4240209 | 4240209 | 50000 | 0.10 | 1.000 |
| cluster-sorted | 3 | random-s+2 | fr | 1.0820 | 1.1253 | 1.0099 | 1.013 | 1979 | 399664 | 1099089 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+2 | rateless-mu1 | 1.0786 | 1.1098 | 1.0371 | 1.421 | 1975 | 399044 | 1097378 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | random-s+2 | rateless-mu2 | 1.0808 | 1.1024 | 1.0127 | 1.016 | 1469 | 298159 | 552975 | 75000 | 0.00 | 0.989 |
| cluster-sorted | 3 | random-s+2 | rateless-mu4 | 1.1046 | 1.1826 | 1.0166 | 1.020 | 990 | 200842 | 277306 | 100000 | 0.00 | 0.996 |
| cluster-sorted | 3 | adversarial-s | uncoded | 1.8538 | 2.4650 | 12.7613 | 23.040 | 6200 | 1252481 | 1252481 | 12500 | 3.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s | replicate | 1.0506 | 1.0725 | 1.0033 | 1.004 | 7636 | 5010219 | 5010219 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s | fr | 1.0820 | 1.1253 | 1.0099 | 1.013 | 1979 | 399664 | 1298665 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s | rateless-mu1 | 1.0788 | 1.1118 | 1.0100 | 1.012 | 1976 | 399071 | 1296611 | 50000 | 0.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s | rateless-mu2 | 1.0834 | 1.1140 | 1.0126 | 1.016 | 1487 | 301754 | 653611 | 75000 | 0.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s | rateless-mu4 | 1.1080 | 1.1517 | 1.0167 | 1.020 | 993 | 201704 | 327745 | 100000 | 0.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s+1 | uncoded | 2.0166 | 2.8707 | 16.6731 | 28.892 | 5723 | 1156059 | 1156059 | 12500 | 4.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s+1 | replicate | 1.2382 | 1.6341 | 3.5087 | 8.628 | 7363 | 4623969 | 4623969 | 50000 | 0.57 | 1.000 |
| cluster-sorted | 3 | adversarial-s+1 | fr | 1.9990 | 3.1593 | 19.0795 | 30.464 | 1484 | 299667 | 1198668 | 50000 | 1.00 | 1.000 |
| cluster-sorted | 3 | adversarial-s+1 | rateless-mu1 | 2.0469 | 3.0767 | 19.0860 | 30.680 | 1482 | 299418 | 1196958 | 50000 | 1.00 | 0.750 |
| cluster-sorted | 3 | adversarial-s+1 | rateless-mu2 | 1.0885 | 1.1278 | 1.0153 | 1.025 | 1372 | 278268 | 603253 | 75000 | 0.00 | 0.922 |
| cluster-sorted | 3 | adversarial-s+1 | rateless-mu4 | 1.1060 | 1.1523 | 1.0168 | 1.021 | 994 | 201663 | 302535 | 100000 | 0.00 | 1.000 |



## F5: merge-reduce depth

`MS r` = mean over candidates of the mean-square r(Q) over seeds (the reduces are unbiased, so this is the variance); `E1 bound` = mean over candidates of (S/m) sum over sampling levels of kappa_l(Q), S = 92.0.

| order | algorithm | m | blocks | L | sampled reduces | sampled levels | distortion mean | MS r | E1 bound | MS / bound |
|---|---|---|---|---|---|---|---|---|---|---|
| random | sensitivity | 500 | 1 | 0 | 1.0 | 1 | 1.1448 | 0.00387 | 0.184 | 0.021 |
| random | uniform | 500 | 1 | 0 | 1.0 | 1 | 1.2324 | 0.00771 | 0.184 | 0.042 |
| random | sensitivity | 1000 | 1 | 0 | 1.0 | 1 | 1.0959 | 0.00202 | 0.0920 | 0.022 |
| random | uniform | 1000 | 1 | 0 | 1.0 | 1 | 1.1488 | 0.00321 | 0.0920 | 0.035 |
| random | sensitivity-lossy | 1000 | 1 | 0 | 1.0 | 1 | 1.1007 | 0.00229 | 0.0920 | 0.025 |
| random | sensitivity | 4000 | 1 | 0 | 1.0 | 1 | 1.0483 | 0.000537 | 0.0230 | 0.023 |
| random | uniform | 4000 | 1 | 0 | 1.0 | 1 | 1.0765 | 0.000894 | 0.0230 | 0.039 |
| random | sensitivity | 500 | 2 | 1 | 3.0 | 2 | 1.1768 | 0.00625 | 0.276 | 0.023 |
| random | uniform | 500 | 2 | 1 | 3.0 | 2 | 1.2696 | 0.0107 | 0.276 | 0.039 |
| random | sensitivity | 1000 | 2 | 1 | 3.0 | 2 | 1.1283 | 0.00315 | 0.138 | 0.023 |
| random | uniform | 1000 | 2 | 1 | 3.0 | 2 | 1.1926 | 0.00533 | 0.138 | 0.039 |
| random | sensitivity-lossy | 1000 | 2 | 1 | 3.0 | 2 | 1.1267 | 0.00350 | 0.138 | 0.025 |
| random | sensitivity | 4000 | 2 | 1 | 3.0 | 2 | 1.0593 | 0.000798 | 0.0345 | 0.023 |
| random | uniform | 4000 | 2 | 1 | 3.0 | 2 | 1.0943 | 0.00138 | 0.0345 | 0.040 |
| random | sensitivity | 500 | 4 | 2 | 7.0 | 3 | 1.1985 | 0.00780 | 0.322 | 0.024 |
| random | uniform | 500 | 4 | 2 | 7.0 | 3 | 1.2898 | 0.0119 | 0.322 | 0.037 |
| random | sensitivity | 1000 | 4 | 2 | 7.0 | 3 | 1.1324 | 0.00368 | 0.161 | 0.023 |
| random | uniform | 1000 | 4 | 2 | 7.0 | 3 | 1.2024 | 0.00628 | 0.161 | 0.039 |
| random | sensitivity-lossy | 1000 | 4 | 2 | 7.0 | 3 | 1.1445 | 0.00406 | 0.161 | 0.025 |
| random | sensitivity | 4000 | 4 | 2 | 7.0 | 3 | 1.0660 | 0.000986 | 0.0403 | 0.024 |
| random | uniform | 4000 | 4 | 2 | 7.0 | 3 | 1.0972 | 0.00155 | 0.0403 | 0.038 |
| random | sensitivity | 500 | 8 | 3 | 15.0 | 4 | 1.2054 | 0.00816 | 0.345 | 0.024 |
| random | uniform | 500 | 8 | 3 | 15.0 | 4 | 1.3115 | 0.0125 | 0.345 | 0.036 |
| random | sensitivity | 1000 | 8 | 3 | 15.0 | 4 | 1.1415 | 0.00416 | 0.173 | 0.024 |
| random | uniform | 1000 | 8 | 3 | 15.0 | 4 | 1.2041 | 0.00615 | 0.173 | 0.036 |
| random | sensitivity-lossy | 1000 | 8 | 3 | 15.0 | 4 | 1.1430 | 0.00403 | 0.173 | 0.023 |
| random | sensitivity | 4000 | 8 | 3 | 15.0 | 4 | 1.0687 | 0.00107 | 0.0431 | 0.025 |
| random | uniform | 4000 | 8 | 3 | 15.0 | 4 | 1.0995 | 0.00173 | 0.0431 | 0.040 |
| random | sensitivity | 500 | 16 | 4 | 31.0 | 5 | 1.2091 | 0.00865 | 0.357 | 0.024 |
| random | uniform | 500 | 16 | 4 | 31.0 | 5 | 1.3063 | 0.0126 | 0.357 | 0.035 |
| random | sensitivity | 1000 | 16 | 4 | 31.0 | 5 | 1.1438 | 0.00441 | 0.178 | 0.025 |
| random | uniform | 1000 | 16 | 4 | 31.0 | 5 | 1.2251 | 0.00697 | 0.178 | 0.039 |
| random | sensitivity-lossy | 1000 | 16 | 4 | 31.0 | 5 | 1.1431 | 0.00431 | 0.178 | 0.024 |
| random | sensitivity | 4000 | 16 | 4 | 31.0 | 5 | 1.0678 | 0.000995 | 0.0446 | 0.022 |
| random | uniform | 4000 | 16 | 4 | 31.0 | 5 | 1.1094 | 0.00184 | 0.0446 | 0.041 |
| random | sensitivity | 500 | 32 | 5 | 63.0 | 6 | 1.2185 | 0.00890 | 0.362 | 0.025 |
| random | uniform | 500 | 32 | 5 | 63.0 | 6 | 1.3318 | 0.0145 | 0.362 | 0.040 |
| random | sensitivity | 1000 | 32 | 5 | 63.0 | 6 | 1.1416 | 0.00413 | 0.181 | 0.023 |
| random | uniform | 1000 | 32 | 5 | 63.0 | 6 | 1.2185 | 0.00687 | 0.181 | 0.038 |
| random | sensitivity-lossy | 1000 | 32 | 5 | 63.0 | 6 | 1.1458 | 0.00461 | 0.181 | 0.025 |
| random | sensitivity | 4000 | 32 | 5 | 63.0 | 6 | 1.0660 | 0.00101 | 0.0453 | 0.022 |
| random | uniform | 4000 | 32 | 5 | 63.0 | 6 | 1.1039 | 0.00176 | 0.0453 | 0.039 |
| random | sensitivity | 500 | 64 | 6 | 127.0 | 7 | 1.2053 | 0.00841 | 0.365 | 0.023 |
| random | uniform | 500 | 64 | 6 | 127.0 | 7 | 1.3326 | 0.0141 | 0.365 | 0.039 |
| random | sensitivity | 1000 | 64 | 6 | 127.0 | 7 | 1.1394 | 0.00399 | 0.183 | 0.022 |
| random | uniform | 1000 | 64 | 6 | 127.0 | 7 | 1.2093 | 0.00636 | 0.183 | 0.035 |
| random | sensitivity-lossy | 1000 | 64 | 6 | 127.0 | 7 | 1.1545 | 0.00486 | 0.183 | 0.027 |
| random | sensitivity | 4000 | 64 | 6 | 63.0 | 6 | 1.0705 | 0.00107 | 0.0453 | 0.024 |
| random | uniform | 4000 | 64 | 6 | 63.0 | 6 | 1.1016 | 0.00162 | 0.0453 | 0.036 |
| random | sensitivity | 500 | 128 | 7 | 255.0 | 8 | 1.2013 | 0.00798 | 0.367 | 0.022 |
| random | uniform | 500 | 128 | 7 | 255.0 | 8 | 1.3210 | 0.0134 | 0.367 | 0.036 |
| random | sensitivity | 1000 | 128 | 7 | 254.7 | 8 | 1.1378 | 0.00410 | 0.183 | 0.022 |
| random | uniform | 1000 | 128 | 7 | 255.0 | 8 | 1.2267 | 0.00737 | 0.183 | 0.040 |
| random | sensitivity-lossy | 1000 | 128 | 7 | 240.3 | 8 | 1.1483 | 0.00457 | 0.183 | 0.025 |
| random | sensitivity | 4000 | 128 | 7 | 63.0 | 6 | 1.0656 | 0.000951 | 0.0453 | 0.021 |
| random | uniform | 4000 | 128 | 7 | 63.0 | 6 | 1.1063 | 0.00176 | 0.0453 | 0.039 |
| random | sensitivity | 500 | 256 | 8 | 509.8 | 9 | 1.2028 | 0.00774 | 0.367 | 0.021 |
| random | uniform | 500 | 256 | 8 | 511.0 | 9 | 1.3389 | 0.0147 | 0.367 | 0.040 |
| random | sensitivity | 1000 | 256 | 8 | 254.6 | 8 | 1.1411 | 0.00399 | 0.183 | 0.022 |
| random | uniform | 1000 | 256 | 8 | 255.0 | 8 | 1.2293 | 0.00799 | 0.183 | 0.044 |
| random | sensitivity-lossy | 1000 | 256 | 8 | 212.9 | 8 | 1.1589 | 0.00489 | 0.183 | 0.027 |
| random | sensitivity | 4000 | 256 | 8 | 63.0 | 6 | 1.0694 | 0.00100 | 0.0453 | 0.022 |
| random | uniform | 4000 | 256 | 8 | 63.0 | 6 | 1.1090 | 0.00185 | 0.0453 | 0.041 |
| cluster-sorted | sensitivity | 500 | 1 | 0 | 1.0 | 1 | 1.1486 | 0.00421 | 0.184 | 0.023 |
| cluster-sorted | uniform | 500 | 1 | 0 | 1.0 | 1 | 1.2218 | 0.00720 | 0.184 | 0.039 |
| cluster-sorted | sensitivity | 1000 | 1 | 0 | 1.0 | 1 | 1.0959 | 0.00194 | 0.0920 | 0.021 |
| cluster-sorted | uniform | 1000 | 1 | 0 | 1.0 | 1 | 1.1553 | 0.00347 | 0.0920 | 0.038 |
| cluster-sorted | sensitivity-lossy | 1000 | 1 | 0 | 1.0 | 1 | 1.1019 | 0.00222 | 0.0920 | 0.024 |
| cluster-sorted | sensitivity | 4000 | 1 | 0 | 1.0 | 1 | 1.0469 | 0.000495 | 0.0230 | 0.022 |
| cluster-sorted | uniform | 4000 | 1 | 0 | 1.0 | 1 | 1.0706 | 0.000798 | 0.0230 | 0.035 |
| cluster-sorted | sensitivity | 500 | 2 | 1 | 3.0 | 2 | 1.1867 | 0.00662 | 0.289 | 0.023 |
| cluster-sorted | uniform | 500 | 2 | 1 | 3.0 | 2 | 1.2789 | 0.0106 | 0.289 | 0.037 |
| cluster-sorted | sensitivity | 1000 | 2 | 1 | 3.0 | 2 | 1.1173 | 0.00297 | 0.144 | 0.021 |
| cluster-sorted | uniform | 1000 | 2 | 1 | 3.0 | 2 | 1.1849 | 0.00523 | 0.144 | 0.036 |
| cluster-sorted | sensitivity-lossy | 1000 | 2 | 1 | 3.0 | 2 | 1.1290 | 0.00331 | 0.144 | 0.023 |
| cluster-sorted | sensitivity | 4000 | 2 | 1 | 3.0 | 2 | 1.0576 | 0.000731 | 0.0361 | 0.020 |
| cluster-sorted | uniform | 4000 | 2 | 1 | 3.0 | 2 | 1.0907 | 0.00141 | 0.0361 | 0.039 |
| cluster-sorted | sensitivity | 500 | 4 | 2 | 7.0 | 3 | 1.1964 | 0.00686 | 0.349 | 0.020 |
| cluster-sorted | uniform | 500 | 4 | 2 | 7.0 | 3 | 1.2960 | 0.0123 | 0.349 | 0.035 |
| cluster-sorted | sensitivity | 1000 | 4 | 2 | 7.0 | 3 | 1.1412 | 0.00397 | 0.174 | 0.023 |
| cluster-sorted | uniform | 1000 | 4 | 2 | 7.0 | 3 | 1.2008 | 0.00582 | 0.174 | 0.033 |
| cluster-sorted | sensitivity-lossy | 1000 | 4 | 2 | 7.0 | 3 | 1.1406 | 0.00385 | 0.174 | 0.022 |
| cluster-sorted | sensitivity | 4000 | 4 | 2 | 7.0 | 3 | 1.0671 | 0.000963 | 0.0436 | 0.022 |
| cluster-sorted | uniform | 4000 | 4 | 2 | 7.0 | 3 | 1.0996 | 0.00163 | 0.0436 | 0.037 |
| cluster-sorted | sensitivity | 500 | 8 | 3 | 15.0 | 4 | 1.2027 | 0.00791 | 0.384 | 0.021 |
| cluster-sorted | uniform | 500 | 8 | 3 | 15.0 | 4 | 1.3319 | 0.0147 | 0.384 | 0.038 |
| cluster-sorted | sensitivity | 1000 | 8 | 3 | 15.0 | 4 | 1.1401 | 0.00400 | 0.192 | 0.021 |
| cluster-sorted | uniform | 1000 | 8 | 3 | 15.0 | 4 | 1.2161 | 0.00695 | 0.192 | 0.036 |
| cluster-sorted | sensitivity-lossy | 1000 | 8 | 3 | 14.8 | 4 | 1.1456 | 0.00410 | 0.192 | 0.021 |
| cluster-sorted | sensitivity | 4000 | 8 | 3 | 15.0 | 4 | 1.0688 | 0.000954 | 0.0479 | 0.020 |
| cluster-sorted | uniform | 4000 | 8 | 3 | 15.0 | 4 | 1.1051 | 0.00168 | 0.0479 | 0.035 |
| cluster-sorted | sensitivity | 500 | 16 | 4 | 31.0 | 5 | 1.2137 | 0.00820 | 0.404 | 0.020 |
| cluster-sorted | uniform | 500 | 16 | 4 | 31.0 | 5 | 1.3155 | 0.0146 | 0.404 | 0.036 |
| cluster-sorted | sensitivity | 1000 | 16 | 4 | 31.0 | 5 | 1.1558 | 0.00471 | 0.202 | 0.023 |
| cluster-sorted | uniform | 1000 | 16 | 4 | 31.0 | 5 | 1.2153 | 0.00632 | 0.202 | 0.031 |
| cluster-sorted | sensitivity-lossy | 1000 | 16 | 4 | 29.8 | 5 | 1.1491 | 0.00423 | 0.202 | 0.021 |
| cluster-sorted | sensitivity | 4000 | 16 | 4 | 30.9 | 5 | 1.0703 | 0.000978 | 0.0505 | 0.019 |
| cluster-sorted | uniform | 4000 | 16 | 4 | 31.0 | 5 | 1.1048 | 0.00173 | 0.0505 | 0.034 |
| cluster-sorted | sensitivity | 500 | 32 | 5 | 62.5 | 6 | 1.2160 | 0.00832 | 0.417 | 0.020 |
| cluster-sorted | uniform | 500 | 32 | 5 | 63.0 | 6 | 1.3205 | 0.0143 | 0.417 | 0.034 |
| cluster-sorted | sensitivity | 1000 | 32 | 5 | 62.4 | 6 | 1.1406 | 0.00401 | 0.208 | 0.019 |
| cluster-sorted | uniform | 1000 | 32 | 5 | 63.0 | 6 | 1.2197 | 0.00705 | 0.208 | 0.034 |
| cluster-sorted | sensitivity-lossy | 1000 | 32 | 5 | 59.7 | 6 | 1.1489 | 0.00441 | 0.208 | 0.021 |
| cluster-sorted | sensitivity | 4000 | 32 | 5 | 61.6 | 6 | 1.0683 | 0.000975 | 0.0521 | 0.019 |
| cluster-sorted | uniform | 4000 | 32 | 5 | 63.0 | 6 | 1.1048 | 0.00172 | 0.0521 | 0.033 |
| cluster-sorted | sensitivity | 500 | 64 | 6 | 125.4 | 7 | 1.2187 | 0.00885 | 0.424 | 0.021 |
| cluster-sorted | uniform | 500 | 64 | 6 | 127.0 | 7 | 1.3264 | 0.0139 | 0.424 | 0.033 |
| cluster-sorted | sensitivity | 1000 | 64 | 6 | 124.9 | 7 | 1.1532 | 0.00438 | 0.212 | 0.021 |
| cluster-sorted | uniform | 1000 | 64 | 6 | 127.0 | 7 | 1.2176 | 0.00720 | 0.212 | 0.034 |
| cluster-sorted | sensitivity-lossy | 1000 | 64 | 6 | 118.8 | 7 | 1.1397 | 0.00360 | 0.212 | 0.017 |
| cluster-sorted | sensitivity | 4000 | 64 | 6 | 61.6 | 6 | 1.0710 | 0.00102 | 0.0521 | 0.020 |
| cluster-sorted | uniform | 4000 | 64 | 6 | 63.0 | 6 | 1.0989 | 0.00151 | 0.0521 | 0.029 |
| cluster-sorted | sensitivity | 500 | 128 | 7 | 250.7 | 8 | 1.2228 | 0.00917 | 0.429 | 0.021 |
| cluster-sorted | uniform | 500 | 128 | 7 | 255.0 | 8 | 1.3106 | 0.0125 | 0.429 | 0.029 |
| cluster-sorted | sensitivity | 1000 | 128 | 7 | 247.2 | 8 | 1.1402 | 0.00396 | 0.215 | 0.018 |
| cluster-sorted | uniform | 1000 | 128 | 7 | 255.0 | 8 | 1.2129 | 0.00703 | 0.215 | 0.033 |
| cluster-sorted | sensitivity-lossy | 1000 | 128 | 7 | 229.8 | 8 | 1.1472 | 0.00433 | 0.215 | 0.020 |
| cluster-sorted | sensitivity | 4000 | 128 | 7 | 61.6 | 6 | 1.0701 | 0.000984 | 0.0521 | 0.019 |
| cluster-sorted | uniform | 4000 | 128 | 7 | 63.0 | 6 | 1.1008 | 0.00154 | 0.0521 | 0.029 |
| cluster-sorted | sensitivity | 500 | 256 | 8 | 496.8 | 9 | 1.2157 | 0.00892 | 0.432 | 0.021 |
| cluster-sorted | uniform | 500 | 256 | 8 | 511.0 | 9 | 1.3374 | 0.0146 | 0.432 | 0.034 |
| cluster-sorted | sensitivity | 1000 | 256 | 8 | 247.3 | 8 | 1.1486 | 0.00424 | 0.215 | 0.020 |
| cluster-sorted | uniform | 1000 | 256 | 8 | 255.0 | 8 | 1.2307 | 0.00770 | 0.215 | 0.036 |
| cluster-sorted | sensitivity-lossy | 1000 | 256 | 8 | 227.3 | 8 | 1.1477 | 0.00435 | 0.215 | 0.020 |
| cluster-sorted | sensitivity | 4000 | 256 | 8 | 61.6 | 6 | 1.0712 | 0.00106 | 0.0521 | 0.020 |
| cluster-sorted | uniform | 4000 | 256 | 8 | 63.0 | 6 | 1.1055 | 0.00182 | 0.0521 | 0.035 |

Slope of log MS r vs log(L+1), L >= 3 (sensitivity): random m=500: -0.08 +/- 0.21 (sum kappa: 0.07); random m=1000: -0.08 +/- 0.13 (sum kappa: 0.07); random m=4000: -0.07 +/- 0.17 (sum kappa: 0.06); cluster-sorted m=500: 0.18 +/- 0.09 (sum kappa: 0.15); cluster-sorted m=1000: -0.02 +/- 0.31 (sum kappa: 0.14); cluster-sorted m=4000: 0.10 +/- 0.10 (sum kappa: 0.10); 

Through-origin fit MS r = c (S/m) sum kappa over all sensitivity cells: c = 0.022 (R^2 0.995).



## F6: aggregation tree with crashing nodes (64 leaves, exact data)

| partition | arity | L | f | checkpoint | internal HT | lost fraction | 1-(1-f^(r+1))^L | mean abs r | distortion p99 | max |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 2 | 6 | 0.01 | 0 | true | 0.0557 | 0.0585 | 0.0007218 | 1.0116 | 1.02 |
| random | 2 | 6 | 0.01 | 0 | false | 0.0557 | 0.0585 | 0.0005356 | 1.0093 | 1.03 |
| random | 2 | 6 | 0.01 | 1 | true | 0.0003 | 0.0006 | 8.208e-06 | 1.0010 | 1.00 |
| random | 2 | 6 | 0.01 | 1 | false | 0.0003 | 0.0006 | 5.667e-06 | 1.0005 | 1.00 |
| random | 2 | 6 | 0.05 | 0 | true | 0.2693 | 0.2649 | 0.004290 | 1.0268 | Infinity |
| random | 2 | 6 | 0.05 | 0 | false | 0.2693 | 0.2649 | 0.003763 | 1.0218 | Infinity |
| random | 2 | 6 | 0.05 | 1 | true | 0.0161 | 0.0149 | 0.0002348 | 1.0088 | 1.02 |
| random | 2 | 6 | 0.05 | 1 | false | 0.0161 | 0.0149 | 0.0001817 | 1.0081 | 1.01 |
| random | 2 | 6 | 0.1 | 0 | true | 0.4711 | 0.4686 | 0.01369 | Infinity | Infinity |
| random | 2 | 6 | 0.1 | 0 | false | 0.4711 | 0.4686 | 0.01286 | Infinity | Infinity |
| random | 2 | 6 | 0.1 | 1 | true | 0.0593 | 0.0585 | 0.0007518 | 1.0130 | 1.02 |
| random | 2 | 6 | 0.1 | 1 | false | 0.0593 | 0.0585 | 0.0005731 | 1.0124 | 1.02 |
| random | 4 | 3 | 0.01 | 0 | true | 0.0298 | 0.0297 | 0.0003798 | 1.0065 | 1.01 |
| random | 4 | 3 | 0.01 | 0 | false | 0.0298 | 0.0297 | 0.0003427 | 1.0066 | 1.01 |
| random | 4 | 3 | 0.01 | 1 | true | 0.0001 | 0.0003 | 1.625e-06 | 1.0000 | 1.00 |
| random | 4 | 3 | 0.01 | 1 | false | 0.0001 | 0.0003 | 1.460e-06 | 1.0000 | 1.00 |
| random | 4 | 3 | 0.05 | 0 | true | 0.1401 | 0.1426 | 0.001236 | 1.0150 | 1.03 |
| random | 4 | 3 | 0.05 | 0 | false | 0.1401 | 0.1426 | 0.001113 | 1.0128 | 1.03 |
| random | 4 | 3 | 0.05 | 1 | true | 0.0065 | 0.0075 | 9.906e-05 | 1.0042 | 1.01 |
| random | 4 | 3 | 0.05 | 1 | false | 0.0065 | 0.0075 | 8.999e-05 | 1.0040 | 1.01 |
| random | 4 | 3 | 0.1 | 0 | true | 0.2735 | 0.2710 | 0.001983 | 1.0198 | 1.04 |
| random | 4 | 3 | 0.1 | 0 | false | 0.2735 | 0.2710 | 0.001779 | 1.0181 | 1.04 |
| random | 4 | 3 | 0.1 | 1 | true | 0.0304 | 0.0297 | 0.0003746 | 1.0079 | 1.01 |
| random | 4 | 3 | 0.1 | 1 | false | 0.0304 | 0.0297 | 0.0003412 | 1.0075 | 1.01 |
| random | 8 | 2 | 0.01 | 0 | true | 0.0188 | 0.0199 | 0.0002589 | 1.0052 | 1.01 |
| random | 8 | 2 | 0.01 | 0 | false | 0.0188 | 0.0199 | 0.0002484 | 1.0051 | 1.01 |
| random | 8 | 2 | 0.01 | 1 | true | 0.0003 | 0.0002 | 4.187e-06 | 1.0000 | 1.00 |
| random | 8 | 2 | 0.01 | 1 | false | 0.0003 | 0.0002 | 4.254e-06 | 1.0000 | 1.00 |
| random | 8 | 2 | 0.05 | 0 | true | 0.1026 | 0.0975 | 0.0009364 | 1.0094 | 1.01 |
| random | 8 | 2 | 0.05 | 0 | false | 0.1026 | 0.0975 | 0.0009022 | 1.0096 | 1.01 |
| random | 8 | 2 | 0.05 | 1 | true | 0.0056 | 0.0050 | 8.617e-05 | 1.0037 | 1.01 |
| random | 8 | 2 | 0.05 | 1 | false | 0.0056 | 0.0050 | 8.283e-05 | 1.0037 | 1.01 |
| random | 8 | 2 | 0.1 | 0 | true | 0.1881 | 0.1900 | 0.001395 | 1.0132 | 1.02 |
| random | 8 | 2 | 0.1 | 0 | false | 0.1881 | 0.1900 | 0.001337 | 1.0131 | 1.02 |
| random | 8 | 2 | 0.1 | 1 | true | 0.0182 | 0.0199 | 0.0002641 | 1.0056 | 1.01 |
| random | 8 | 2 | 0.1 | 1 | false | 0.0182 | 0.0199 | 0.0002507 | 1.0056 | 1.01 |
| random | 64 | 1 | 0.01 | 0 | true | 0.0085 | 0.0100 | 0.0001699 | 1.0029 | 1.01 |
| random | 64 | 1 | 0.01 | 0 | false | 0.0085 | 0.0100 | 0.0001699 | 1.0029 | 1.01 |
| random | 64 | 1 | 0.01 | 1 | true | 0.0001 | 0.0001 | 2.323e-06 | 1.0000 | 1.00 |
| random | 64 | 1 | 0.01 | 1 | false | 0.0001 | 0.0001 | 2.323e-06 | 1.0000 | 1.00 |
| random | 64 | 1 | 0.05 | 0 | true | 0.0513 | 0.0500 | 0.0006473 | 1.0061 | 1.01 |
| random | 64 | 1 | 0.05 | 0 | false | 0.0513 | 0.0500 | 0.0006473 | 1.0061 | 1.01 |
| random | 64 | 1 | 0.05 | 1 | true | 0.0027 | 0.0025 | 5.878e-05 | 1.0020 | 1.00 |
| random | 64 | 1 | 0.05 | 1 | false | 0.0027 | 0.0025 | 5.878e-05 | 1.0020 | 1.00 |
| random | 64 | 1 | 0.1 | 0 | true | 0.0988 | 0.1000 | 0.0009389 | 1.0082 | 1.01 |
| random | 64 | 1 | 0.1 | 0 | false | 0.0988 | 0.1000 | 0.0009389 | 1.0082 | 1.01 |
| random | 64 | 1 | 0.1 | 1 | true | 0.0102 | 0.0100 | 0.0001961 | 1.0032 | 1.00 |
| random | 64 | 1 | 0.1 | 1 | false | 0.0102 | 0.0100 | 0.0001961 | 1.0032 | 1.00 |
| projection-sorted | 2 | 6 | 0.01 | 0 | true | 0.0543 | 0.0585 | 0.01832 | 2.0881 | 2.73 |
| projection-sorted | 2 | 6 | 0.01 | 0 | false | 0.0543 | 0.0585 | 0.02312 | 2.0098 | 2.62 |
| projection-sorted | 2 | 6 | 0.01 | 1 | true | 0.0005 | 0.0006 | 0.0001013 | 1.0000 | 1.15 |
| projection-sorted | 2 | 6 | 0.01 | 1 | false | 0.0005 | 0.0006 | 0.0002476 | 1.0000 | 1.27 |
| projection-sorted | 2 | 6 | 0.05 | 0 | true | 0.2638 | 0.2649 | 0.08734 | 6.1346 | Infinity |
| projection-sorted | 2 | 6 | 0.05 | 0 | false | 0.2638 | 0.2649 | 0.09462 | 6.1555 | Infinity |
| projection-sorted | 2 | 6 | 0.05 | 1 | true | 0.0140 | 0.0149 | 0.004938 | 1.6249 | 3.36 |
| projection-sorted | 2 | 6 | 0.05 | 1 | false | 0.0140 | 0.0149 | 0.006677 | 1.5413 | 3.36 |
| projection-sorted | 2 | 6 | 0.1 | 0 | true | 0.4603 | 0.4686 | 0.1464 | 13.8979 | Infinity |
| projection-sorted | 2 | 6 | 0.1 | 0 | false | 0.4603 | 0.4686 | 0.1509 | 15.5046 | Infinity |
| projection-sorted | 2 | 6 | 0.1 | 1 | true | 0.0542 | 0.0585 | 0.01836 | 1.8757 | 4.41 |
| projection-sorted | 2 | 6 | 0.1 | 1 | false | 0.0542 | 0.0585 | 0.02333 | 1.9173 | 4.33 |
| projection-sorted | 4 | 3 | 0.01 | 0 | true | 0.0300 | 0.0297 | 0.01036 | 1.6069 | 3.34 |
| projection-sorted | 4 | 3 | 0.01 | 0 | false | 0.0300 | 0.0297 | 0.01322 | 1.6059 | 3.34 |
| projection-sorted | 4 | 3 | 0.01 | 1 | true | 0.0003 | 0.0003 | 0.0001700 | 1.0010 | 1.18 |
| projection-sorted | 4 | 3 | 0.01 | 1 | false | 0.0003 | 0.0003 | 0.0001891 | 1.0185 | 1.21 |
| projection-sorted | 4 | 3 | 0.05 | 0 | true | 0.1438 | 0.1426 | 0.04497 | 2.8003 | 9.69 |
| projection-sorted | 4 | 3 | 0.05 | 0 | false | 0.1438 | 0.1426 | 0.05230 | 3.1231 | 10.1 |
| projection-sorted | 4 | 3 | 0.05 | 1 | true | 0.0095 | 0.0075 | 0.003271 | 1.3138 | 2.55 |
| projection-sorted | 4 | 3 | 0.05 | 1 | false | 0.0095 | 0.0075 | 0.004309 | 1.3167 | 2.55 |
| projection-sorted | 4 | 3 | 0.1 | 0 | true | 0.2709 | 0.2710 | 0.07867 | 3.4829 | 10.2 |
| projection-sorted | 4 | 3 | 0.1 | 0 | false | 0.2709 | 0.2710 | 0.08709 | 3.6537 | 10.2 |
| projection-sorted | 4 | 3 | 0.1 | 1 | true | 0.0314 | 0.0297 | 0.01134 | 1.8464 | 2.77 |
| projection-sorted | 4 | 3 | 0.1 | 1 | false | 0.0314 | 0.0297 | 0.01461 | 1.8459 | 2.74 |
| projection-sorted | 8 | 2 | 0.01 | 0 | true | 0.0208 | 0.0199 | 0.007447 | 1.5367 | 2.02 |
| projection-sorted | 8 | 2 | 0.01 | 0 | false | 0.0208 | 0.0199 | 0.009182 | 1.5325 | 2.02 |
| projection-sorted | 8 | 2 | 0.01 | 1 | true | 0.0001 | 0.0002 | 2.691e-05 | 1.0000 | 1.04 |
| projection-sorted | 8 | 2 | 0.01 | 1 | false | 0.0001 | 0.0002 | 2.423e-05 | 1.0000 | 1.05 |
| projection-sorted | 8 | 2 | 0.05 | 0 | true | 0.0975 | 0.0975 | 0.03107 | 1.8832 | 2.30 |
| projection-sorted | 8 | 2 | 0.05 | 0 | false | 0.0975 | 0.0975 | 0.03517 | 1.8781 | 2.32 |
| projection-sorted | 8 | 2 | 0.05 | 1 | true | 0.0060 | 0.0050 | 0.002079 | 1.2307 | 2.33 |
| projection-sorted | 8 | 2 | 0.05 | 1 | false | 0.0060 | 0.0050 | 0.002664 | 1.2319 | 2.32 |
| projection-sorted | 8 | 2 | 0.1 | 0 | true | 0.1945 | 0.1900 | 0.05511 | 2.1651 | 2.92 |
| projection-sorted | 8 | 2 | 0.1 | 0 | false | 0.1945 | 0.1900 | 0.05924 | 2.2061 | 2.96 |
| projection-sorted | 8 | 2 | 0.1 | 1 | true | 0.0199 | 0.0199 | 0.007655 | 1.5929 | 2.03 |
| projection-sorted | 8 | 2 | 0.1 | 1 | false | 0.0199 | 0.0199 | 0.009358 | 1.5929 | 2.06 |
| projection-sorted | 64 | 1 | 0.01 | 0 | true | 0.0099 | 0.0100 | 0.004352 | 1.1471 | 1.28 |
| projection-sorted | 64 | 1 | 0.01 | 0 | false | 0.0099 | 0.0100 | 0.004352 | 1.1471 | 1.28 |
| projection-sorted | 64 | 1 | 0.01 | 1 | true | 0.0000 | 0.0001 | 1.470e-05 | 1.0000 | 1.03 |
| projection-sorted | 64 | 1 | 0.01 | 1 | false | 0.0000 | 0.0001 | 1.470e-05 | 1.0000 | 1.03 |
| projection-sorted | 64 | 1 | 0.05 | 0 | true | 0.0505 | 0.0500 | 0.01584 | 1.2418 | 1.55 |
| projection-sorted | 64 | 1 | 0.05 | 0 | false | 0.0505 | 0.0500 | 0.01584 | 1.2418 | 1.55 |
| projection-sorted | 64 | 1 | 0.05 | 1 | true | 0.0028 | 0.0025 | 0.001322 | 1.0811 | 1.22 |
| projection-sorted | 64 | 1 | 0.05 | 1 | false | 0.0028 | 0.0025 | 0.001322 | 1.0811 | 1.22 |
| projection-sorted | 64 | 1 | 0.1 | 0 | true | 0.0982 | 0.1000 | 0.02475 | 1.3795 | 1.66 |
| projection-sorted | 64 | 1 | 0.1 | 0 | false | 0.0982 | 0.1000 | 0.02475 | 1.3795 | 1.66 |
| projection-sorted | 64 | 1 | 0.1 | 1 | true | 0.0098 | 0.0100 | 0.004412 | 1.1728 | 1.65 |
| projection-sorted | 64 | 1 | 0.1 | 1 | false | 0.0098 | 0.0100 | 0.004412 | 1.1728 | 1.65 |



## Instances

| id | n | d | k | s_max/S | S | smallest cluster | prepare s |
|---|---|---|---|---|---|---|---|
| esa22 | 100000 | 50 | 10 | 1.97e-05 | 52.0 | 10000 | 5.7 |
| zipf | 200000 | 20 | 20 | 4.28e-05 | 92.0 | 1030 | 5.5 |
| zipf-100k | 100000 | 20 | 20 | 0.000173 | 92.0 | 515 | 2.6 |
| zipf-400k | 400000 | 20 | 20 | 2.13e-05 | 92.0 | 2060 | 10.4 |
| zipf-heavy | 200000 | 20 | 20 | 0.000489 | 92.0 | 125 | 7.0 |
| zipf-outliers | 200050 | 20 | 20 | 0.0435 | 92.0 | 1 | 4.2 |
