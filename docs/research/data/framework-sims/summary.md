# Framework simulations (scale full, seed 42)

## Verdicts

24 checks, 24 PASS.

| id | prediction | measured | verdict |
|---|---|---|---|
| A1 HT floor | rho=0.3, HT assumes rho=0.2: fixed-query ratio -> 0.875 (a 12.5% floor), failure -> 1 for eps < 0.125 | m=100000: ratio 0.8751 [0.8747, 0.8755], sd 0.0045; P[\|ratio-1\|>0.1] = 1.000; mean-ratio CI covers 0.875 in 7/7 cells | **PASS** |
| A1 RC rate | RC error decays as M'^-1/2 (slope -0.5 +- 0.05) | slope of log RMS on log M' = -0.505 [-0.515, -0.495] over m = 100..100000 | **PASS** |
| A2 header copies | 3 adjacent copies all lost w.p. ~0.083; 100 apart ~1.0e-3 | adjacent 0.08281 [0.08265, 0.08297] (pred 0.08281); spaced 1.004e-03 [9.859e-04, 1.022e-03] (pred 1.0005e-03); ratio 82x | **PASS** |
| A3 header in every packet | a header in every packet adds no failures | 0/200000 trials differ from the reliable-header decoder; P[fail] 0.0014 vs 0.0014; 3 adjacent header packets: P[fail] 0.0834 | **PASS** |
| B1 pair plateau | RC and midpoint decoders plateau at relative error 0.20 on both inputs | m=100000: X_A RC -0.1999; X_A midpoint HT -0.1999; X_B RC +0.1999; X_B midpoint HT +0.2000; | **PASS** |
| B1 pair same law | the decoder input has the same law on X_A and X_B | RC estimate at m=1000, X_A vs X_B: KS p = 0.933, AD p = 0.694 | **PASS** |
| B2 RC limit | RC-worst instance: RC error -> 0.49 | m=100000: mean 0.4937 [0.4906, 0.4967], sd 0.0492; RMS distance to 0.4926 falls with slope -0.502 [-0.510, -0.492] in log m | **PASS** |
| B2 midpoint limit | RC-worst instance: midpoint error -> 0.20 | m=100000: mean 0.2007 [0.1983, 0.2032], sd 0.0394; RMS distance to 0.2000 falls with slope -0.501 [-0.510, -0.492] in log m | **PASS** |
| B2 known p / per-class | known p or per-class header counts: error -> 0 at rate m^-1/2 | m=100000: mean err known +0.0006 [-0.0014, +0.0027], per-class +0.0007 [-0.0012, +0.0026]; RMS slope known -0.501 [-0.510, -0.492], per-class -0.504 [-0.512, -0.495] | **PASS** |
| B3 mixture bias | content-correlated loss on a mixture: RC and midpoint keep a bias that does not shrink with m (equal to the tilted-family bias) | largest predicted \|bias\| 0.0773; measured vs predicted max \|z\| = 2.43 over 4 m x 8 queries x 2 decoders | **PASS** |
| B3 per-class removes bias | per-class header counts remove the bias, error ~ m^-1/2 | per-class and known-p max \|mean err\|/se = 1.91 (m >= 2000); RMS slope of max_Q error -0.502 [-0.513, -0.490] | **PASS** |
| C1 normal approximation | deficit quantile ~0.034 (eta=1e-2) and ~0.054 (eta=1e-4), within 15% | 0.0356 [0.0356, 0.0357] (+5.2%) and 0.0593 [0.0591, 0.0597] (+9.8%) vs 0.0338 and 0.0540 | **PASS** |
| C1 Markov bound | Markov bound 0.097 (eta=1e-2) and 0.132 (eta=1e-4) holds | upper CI 0.0357 <= 0.0969 and 0.0597 <= 0.1325 (slack x2.7, x2.2) | **PASS** |
| C1 Cantelli bound | Cantelli 0.145 (eta=1e-2) and 1.45 (eta=1e-4, vacuous) | 0.0357 <= 0.145; 1.45 >= 1 is vacuous | **PASS** |
| C1 growth | quantiles grow like sqrt(ln 1/eta): ratio eta=1e-4 : 1e-2 in [1.2, 2.0] (sqrt-log 1.41, Gaussian 1.60; 1/sqrt(eta) shape 10.0) | ratio 1.669 [1.657, 1.678] | **PASS** |
| C2 identical given M' | given M', the error law is the same under Bernoulli, Gilbert-Elliott and deterministic loss with random packing (all p > 0.001; 28 tests) | min p over random-packing arms, Bernoulli/sorted and the 3 pairs = 0.2605 | **PASS** |
| C2 sorted + block | sorted packing with deterministic block loss breaks the conditional law | min p of the 4 tests vs loss-free = 0.00e+00; mean err Q0 -0.0829 vs +0.0001 | **PASS** |
| C2 sorted + bursts | sorted packing with Gilbert bursts breaks the conditional law | min p of the 4 tests vs loss-free = 0.00e+00; sd err Q0 0.02104 vs 0.01316 | **PASS** |
| D1 forwarding price | H forwarding hops deliver prod(1-rho_h) of the records: price 1/prod(1-rho_h) = 1.984 | delivered fraction 0.5040 [0.5037, 0.5042] vs 0.5040 (\|z\| = 0.30) | **PASS** |
| D1 forwarding accuracy | forwarding leaves the error unchanged given M' (same law as a loss-free sample of M' draws) | min p over 3 pairs x 2 metrics x (KS, AD) = 0.0276 | **PASS** |
| D2 resampling accuracy | resampling at each level compounds the error as prod(1+eps_l) - 1 | per-run composition violated in 0/1200 runs; q95 root error <= prod(1 + q95 eps_l) - 1 at all L: true; RMS grows with slope 0.238 [0.203, 0.274] in log L | **PASS** |
| D2 resampling loss | resampling resets the loss: every edge delivers 1-rho = 0.8 whatever the depth | max \|z\| of the per-edge delivered fraction against 0.8 over depths = 0.90 | **PASS** |
| D2 forwarding price | forwarding with m/prod(1-rho_h) source draws keeps the loss-free accuracy | mean square of max_Q error vs loss-free m=2000 (RMS 0.0614): max \|z\| over depths = 1.00 | **PASS** |
| E k-median scaling | the master theorem holds for k-median: error ~ eps/sqrt(1-rho) (log-log slope 0.5 +- 0.05), as for k-means | z=1: Q0 0.505 [0.486, 0.525], max_Q 0.502 [0.491, 0.512]; z=2: Q0 0.506 [0.486, 0.526], max_Q 0.510 [0.499, 0.520]; | **PASS** |

## Results

### A1. Misspecified HT versus RC (rho = 0.3, HT assumes 0.2)

Bernoulli channel, B = 1, sensitivity sample of the mixture, 500 repetitions per m. The HT decoder multiplies by 1/(1-0.2); RC divides by M'. Fixed query Q0. Predicted HT ratio (1-0.3)/(1-0.2) = 0.875; RC error ~ M'^-1/2. Slope of log RMS(RC) on log mean M': -0.505 [-0.515, -0.495].

| m | mean M' | HT ratio mean [95% CI] | HT ratio sd | P[HT fails, eps=0.1] | RC RMS rel. err. Q0 [95% CI] | RC mean rel. err. Q0 | P[RC fails, eps=0.1] | RC RMS max_Q err. |
|---|---|---|---|---|---|---|---|---|
| 100 | 69.9 | 0.8738 [0.8613, 0.8863] | 0.1426 | 0.630 [0.587, 0.671] | 0.1505 [0.1409, 0.1595] | +0.0001 | 0.514 [0.470, 0.558] | 0.3343 |
| 300 | 209.4 | 0.8747 [0.8676, 0.8819] | 0.0817 | 0.626 [0.583, 0.667] | 0.0865 [0.0807, 0.0920] | +0.0025 | 0.246 [0.210, 0.286] | 0.1883 |
| 1000 | 699.9 | 0.8767 [0.8725, 0.8809] | 0.0478 | 0.690 [0.648, 0.729] | 0.0490 [0.0456, 0.0522] | +0.0020 | 0.044 [0.029, 0.066] | 0.1050 |
| 3000 | 2101.0 | 0.8747 [0.8724, 0.8771] | 0.0264 | 0.822 [0.786, 0.853] | 0.0281 [0.0263, 0.0298] | -0.0007 | 0.000 [0.000, 0.008] | 0.0588 |
| 10000 | 6998.3 | 0.8755 [0.8742, 0.8768] | 0.0146 | 0.948 [0.925, 0.964] | 0.0151 [0.0141, 0.0160] | +0.0008 | 0.000 [0.000, 0.008] | 0.0328 |
| 30000 | 21001.2 | 0.8747 [0.8740, 0.8754] | 0.0081 | 1.000 [0.992, 1.000] | 0.0082 [0.0076, 0.0087] | -0.0004 | 0.000 [0.000, 0.008] | 0.0186 |
| 100000 | 69995.1 | 0.8751 [0.8747, 0.8755] | 0.0045 | 1.000 [0.992, 1.000] | 0.0047 [0.0044, 0.0050] | +0.0002 | 0.000 [0.000, 0.008] | 0.0101 |

### A2. Header copies on the Gilbert channel

Gilbert channel alpha = 0.01, beta_GE = 0.09 (rho = 0.10, lambda = 0.90, mean burst 11.1), stationary start, 201-packet stream, 20000000 trials. Header copies at packets {0,1,2} or {0,100,200}.

| event | predicted | measured [99% Wilson CI] |
|---|---|---|
| one packet lost (loss rate) | 0.10000 | 0.10001 [0.09984, 0.10018] |
| 3 adjacent copies all lost | 0.08281 = pi_B (1-beta)^2 | 0.08281 [0.08265, 0.08297] |
| 3 copies 100 apart all lost | 1.0005e-03 = pi_B (pi_B + (1-pi_B) lambda^100)^2 | 1.0040e-03 [9.8587e-04, 1.0224e-03] |
| memoryless reference rho^3 | 1.0000e-03 | - |

### A3. Header placement

m = 2000 draws in 200 packets of B = 10 on the Gilbert channel above (one 203-slot realization per trial, 200000 trials); failure means no readable m or max_Q RC error > 0.15. Trials whose failure differs between the reliable header and m in every packet: 0.

| scheme | P[header unreadable] [95% CI] | P[failure] [95% CI] | predicted P[header unreadable] |
|---|---|---|---|
| reliable header (oracle m) | 0 | 0.0014 [0.0013, 0.0016] | 0 |
| m in every data packet | 0.00e+00 [0.00e+00, 1.92e-05] | 0.0014 [0.0013, 0.0016] | 7.1e-10 (all 200 packets lost) |
| 3 adjacent header packets | 0.0821 [0.0809, 0.0833] | 0.0834 [0.0822, 0.0846] | 0.0828 |
| 3 header packets 101 apart | 1.05e-03 [9.13e-04, 1.20e-03] | 0.0025 [0.0022, 0.0027] | 1.00e-03 |

### B1. The lower-bound pair (rho_min = 0.1, rho_max = 0.4, r = 1.5)

X_A: 300 points at a, 300 at b, channel delivers a w.p. 0.9 and b w.p. 0.6. X_B: 400 at a, 200 at b, delivers a w.p. 0.675 and b w.p. 0.9. Uniform sampling (weight 600/m per record), one record per packet, query Q = {a} (cost 300 on X_A, 200 on X_B). Predicted: RC and midpoint errors -0.2 on X_A and +0.2 on X_B for every m; the decoder input has the same law on both (two-sample tests of the RC cost estimate at m = 1000: KS p = 0.933, AD p = 0.694). 1000 repetitions per cell.

| input | m | RC mean err [95% CI] | midpoint mean err [95% CI] | known-p mean err [95% CI] | per-class mean err [95% CI] | known-p RMS | per-class RMS |
|---|---|---|---|---|---|---|---|
| X_A | 100 | -0.1955 [-0.2026, -0.1883] | -0.1941 [-0.2019, -0.1863] | +0.0074 [-0.0023, +0.0171] | +0.0028 [-0.0035, +0.0090] | 0.1568 | 0.1010 |
| X_A | 1000 | -0.2011 [-0.2032, -0.1989] | -0.2003 [-0.2027, -0.1979] | -0.0004 [-0.0033, +0.0026] | -0.0016 [-0.0036, +0.0004] | 0.0478 | 0.0316 |
| X_A | 10000 | -0.2006 [-0.2013, -0.1999] | -0.2007 [-0.2014, -0.1999] | -0.0008 [-0.0018, +0.0001] | -0.0006 [-0.0013, -0.0000] | 0.0154 | 0.0099 |
| X_A | 100000 | -0.1999 [-0.2001, -0.1996] | -0.1999 [-0.2001, -0.1996] | +0.0002 [-0.0001, +0.0005] | +0.0001 [-0.0001, +0.0003] | 0.0049 | 0.0031 |
| X_B | 100 | +0.1945 [+0.1839, +0.2051] | +0.1927 [+0.1814, +0.2040] | -0.0061 [-0.0155, +0.0034] | -0.0047 [-0.0137, +0.0043] | 0.1524 | 0.1446 |
| X_B | 1000 | +0.1991 [+0.1957, +0.2024] | +0.1985 [+0.1949, +0.2021] | -0.0012 [-0.0042, +0.0017] | -0.0008 [-0.0036, +0.0020] | 0.0481 | 0.0449 |
| X_B | 10000 | +0.2006 [+0.1995, +0.2016] | +0.2005 [+0.1994, +0.2017] | +0.0005 [-0.0005, +0.0014] | +0.0007 [-0.0002, +0.0016] | 0.0156 | 0.0144 |
| X_B | 100000 | +0.1999 [+0.1996, +0.2003] | +0.2000 [+0.1996, +0.2003] | -0.0000 [-0.0003, +0.0003] | +0.0000 [-0.0003, +0.0003] | 0.0049 | 0.0045 |

### B2. The RC-worst instance

100 unit points at a delivered w.p. 0.6, one point b delivered w.p. 0.9, uniform sampling, Q = {a} (cost 1, carried by b). Predicted limits: RC 0.9/(60.9/101) - 1 = 0.4926, midpoint 0.2000, known p and per-class 0 at rate m^-1/2. Slopes of log RMS on log m: known p -0.501 [-0.510, -0.492], per-class -0.504 [-0.512, -0.495]; of the RMS distance to the predicted limit: RC -0.502 [-0.510, -0.492], midpoint -0.501 [-0.510, -0.492]. 1000 repetitions per m.

| m | RC mean err [95% CI] | RC sd | midpoint mean err [95% CI] | midpoint sd | known-p mean err [95% CI] | known-p RMS | per-class mean err [95% CI] | per-class RMS |
|---|---|---|---|---|---|---|---|---|
| 100 | +0.5610 [+0.4635, +0.6586] | 1.5741 | +0.2511 [+0.1733, +0.3288] | 1.2538 | +0.0425 [-0.0222, +0.1073] | 1.0452 | +0.0120 [-0.0510, +0.0751] | 1.0171 |
| 1000 | +0.5065 [+0.4762, +0.5369] | 0.4896 | +0.2097 [+0.1853, +0.2341] | 0.3937 | +0.0081 [-0.0122, +0.0284] | 0.3280 | +0.0070 [-0.0122, +0.0261] | 0.3089 |
| 10000 | +0.4898 [+0.4802, +0.4994] | 0.1551 | +0.1974 [+0.1897, +0.2051] | 0.1246 | -0.0022 [-0.0086, +0.0043] | 0.1038 | -0.0019 [-0.0081, +0.0043] | 0.0996 |
| 100000 | +0.4937 [+0.4906, +0.4967] | 0.0492 | +0.2007 [+0.1983, +0.2032] | 0.0394 | +0.0006 [-0.0014, +0.0027] | 0.0328 | +0.0007 [-0.0012, +0.0026] | 0.0311 |

### B3. Content-correlated loss on the Gaussian mixture

Delivery probability per mixture component, from 0.9 (component mean nearest the origin) to 0.6 (farthest), linear in the rank; the class (component) is readable from the record. Sensitivity sampling, one record per packet, 400 repetitions per m. The RC and midpoint biases are predicted exactly (tilted family). Per-class RMS slope on log m: -0.502 [-0.513, -0.490]. Biases at m = 32000 per query:

| m | decoder | max_Q \|mean err\| | max_Q \|mean - predicted\| / se | RMS of max_Q \|err\| [95% CI] |
|---|---|---|---|---|
| 500 | RC | 0.0647 | 2.23 | 0.1631 [0.1571, 0.1690] |
| 500 | midpoint HT | 0.0736 | 1.90 | 0.1736 [0.1670, 0.1799] |
| 500 | known p | 0.0078 | 1.91 | 0.1462 [0.1407, 0.1515] |
| 500 | per-class RC | 0.0099 | 2.28 | 0.1242 [0.1198, 0.1284] |
| 2000 | RC | 0.0720 | 1.15 | 0.1104 [0.1076, 0.1131] |
| 2000 | midpoint HT | 0.0781 | 1.07 | 0.1169 [0.1140, 0.1197] |
| 2000 | known p | 0.0029 | 1.15 | 0.0757 [0.0728, 0.0784] |
| 2000 | per-class RC | 0.0027 | 1.25 | 0.0635 [0.0614, 0.0656] |
| 8000 | RC | 0.0712 | 1.68 | 0.0848 [0.0834, 0.0861] |
| 8000 | midpoint HT | 0.0768 | 1.17 | 0.0903 [0.0888, 0.0917] |
| 8000 | known p | 0.0014 | 1.18 | 0.0374 [0.0361, 0.0386] |
| 8000 | per-class RC | 0.0011 | 1.15 | 0.0312 [0.0301, 0.0322] |
| 32000 | RC | 0.0708 | 2.43 | 0.0746 [0.0739, 0.0753] |
| 32000 | midpoint HT | 0.0768 | 1.95 | 0.0807 [0.0799, 0.0814] |
| 32000 | known p | 0.0008 | 1.91 | 0.0183 [0.0176, 0.0189] |
| 32000 | per-class RC | 0.0007 | 1.77 | 0.0155 [0.0150, 0.0160] |

| query | predicted RC bias | measured RC bias [95% CI] | predicted midpoint bias | measured midpoint bias | per-class bias [95% CI] |
|---|---|---|---|---|---|
| Q0 | +0.0711 | +0.0708 [+0.0699, +0.0716] | +0.0773 | +0.0768 [+0.0759, +0.0777] | -0.0003 [-0.0009, +0.0004] |
| Q1 | -0.0509 | -0.0511 [-0.0521, -0.0500] | -0.0454 | -0.0457 [-0.0468, -0.0446] | +0.0001 [-0.0008, +0.0010] |
| Q2 | -0.0328 | -0.0321 [-0.0331, -0.0312] | -0.0272 | -0.0266 [-0.0277, -0.0256] | +0.0004 [-0.0005, +0.0012] |
| Q3 | +0.0613 | +0.0612 [+0.0600, +0.0624] | +0.0675 | +0.0672 [+0.0660, +0.0684] | -0.0004 [-0.0014, +0.0006] |
| Q4 | -0.0437 | -0.0439 [-0.0450, -0.0428] | -0.0381 | -0.0385 [-0.0396, -0.0374] | +0.0001 [-0.0009, +0.0010] |
| Q5 | -0.0377 | -0.0376 [-0.0384, -0.0368] | -0.0321 | -0.0321 [-0.0330, -0.0313] | +0.0001 [-0.0006, +0.0009] |
| Q6 | +0.0609 | +0.0620 [+0.0611, +0.0628] | +0.0671 | +0.0680 [+0.0670, +0.0689] | +0.0007 [-0.0001, +0.0014] |
| Q7 | +0.0525 | +0.0523 [+0.0515, +0.0532] | +0.0586 | +0.0583 [+0.0574, +0.0592] | -0.0004 [-0.0011, +0.0003] |

### C1. Lower tail of M' on Gilbert's channel

alpha = 0.01, beta_GE = 0.09: rho = 0.10, lambda = 0.90, kappa = 19, mean burst 11.1 packets; m = 10000, B = 1, 10000000 channel realizations from the stationary law. Deficit = ((1-rho)m - M') / ((1-rho)m) at quantile 1-eta. Mean M' 8999.93 (predicted 9000); sd 130.77 (asymptotic 130.77, exact finite-m 130.71). Normal approximation z_{1-eta} sd/((1-rho)m); Cantelli sd sqrt((1-eta)/eta)/((1-rho)m). Log-log slope of the deficit quantile on ln(1/eta) over eta = 1e-01..1e-05: 0.803 empirical, 0.747 for the normal quantile (sqrt(ln 1/eta) is 0.5 only asymptotically).

| eta | empirical deficit [95% CI] | normal approx. | empirical / normal | Markov bound theta_MM | empirical / theta_MM | Cantelli bound | samples beyond the quantile |
|---|---|---|---|---|---|---|---|
| 1e-01 | 0.0189 [0.0189, 0.0189] | 0.0186 | 1.014 | 0.0729 | 0.259 | 0.0436 | 1000000 |
| 1e-02 | 0.0356 [0.0356, 0.0357] | 0.0338 | 1.052 | 0.0969 | 0.367 | 0.1446 | 100000 |
| 1e-03 | 0.0484 [0.0483, 0.0484] | 0.0449 | 1.079 | 0.1161 | 0.417 | 0.4592 | 10000 |
| 1e-04 | 0.0593 [0.0591, 0.0597] | 0.0540 | 1.098 | 0.1325 | 0.448 | 1.4529 | 1000 |
| 1e-05 | 0.0687 [0.0679, 0.0696] | 0.0620 | 1.108 | 0.1471 | 0.467 | 4.5947 | 101 |

### C2. The RC error given M' = 9000

Sensitivity sample of the mixture, m = 10000, B = 1, 4000 samples per arm, each conditioned exactly on M' = 9000 by rejection on the channel alone (the draws are made after acceptance). Random packing = uniform permutation of the draws; sorted packing = draws sorted by mixture component. Block = the middle 1000 packets lost every time. Tests against the loss-free sample of 9000 draws: two-sample KS (asymptotic p) and Anderson-Darling (midrank, 999 permutations).

| arm | mean err Q0 | sd err Q0 | mean max_Q \|err\| | KS p (Q0) | AD p (Q0) | KS p (max_Q) | AD p (max_Q) |
|---|---|---|---|---|---|---|---|
| loss-free m'=9000 | +0.00010 | 0.01316 | 0.02736 | - | - | - | - |
| Bernoulli / random | +0.00005 | 0.01315 | 0.02728 | 0.9530 | 0.8840 | 0.8088 | 0.8600 |
| Gilbert / random | -0.00017 | 0.01322 | 0.02708 | 0.6441 | 0.4160 | 0.4128 | 0.2630 |
| block / random | +0.00005 | 0.01320 | 0.02724 | 0.9342 | 0.8190 | 0.2605 | 0.4020 |
| Bernoulli / sorted | +0.00032 | 0.01333 | 0.02750 | 0.5877 | 0.3840 | 0.5507 | 0.6340 |
| Gilbert / sorted | -0.00023 | 0.02104 | 0.04429 | 0.0000 | 0.0010 | 0.0000 | 0.0010 |
| block / sorted | -0.08292 | 0.01811 | 0.14191 | 0.0000 | 0.0010 | 0.0000 | 0.0010 |

| pair | KS p (Q0) | AD p (Q0) | KS p (max_Q) | AD p (max_Q) |
|---|---|---|---|---|
| Bernoulli / random vs Gilbert / random | 0.9118 | 0.5950 | 0.2847 | 0.3700 |
| Bernoulli / random vs block / random | 0.7916 | 0.9350 | 0.8860 | 0.6890 |
| Gilbert / random vs block / random | 0.5877 | 0.8130 | 0.3104 | 0.3030 |

### D1. Forwarding relays

m = 8000 draws, B = 8, hops with rho = 0.1, 0.2, 0.3; every relay unpacks, permutes uniformly and repacks; m travels in the headers. End-to-end delivered fraction over 20000 channel runs: 0.5040 [0.5037, 0.5042] (predicted prod(1-rho_h) = 0.504, price 1/0.504 = 1.984). Error law given M' = 4032 (exact conditioning by rejection on the channels), 3000 samples per arm:

| arm / pair | mean err Q0 | sd err Q0 | mean max_Q \|err\| | KS p (Q0) | AD p (Q0) | KS p (max_Q) | AD p (max_Q) |
|---|---|---|---|---|---|---|---|
| loss-free m'=4032 | -0.00016 | 0.01982 | 0.04155 | - | - | - | - |
| 3 forwarding hops | +0.00025 | 0.01990 | 0.04129 | - | - | - | - |
| 1 hop, rho = 1 - prod(1-rho_h) | -0.00002 | 0.02022 | 0.04095 | - | - | - | - |
| 3 forwarding hops vs loss-free m'=4032 |  |  |  | 0.6252 | 0.5670 | 0.4980 | 0.4180 |
| 1 hop, rho = 1 - prod(1-rho_h) vs loss-free m'=4032 |  |  |  | 0.7549 | 0.8800 | 0.0276 | 0.0620 |
| 3 forwarding hops vs 1 hop, rho = 1 - prod(1-rho_h) |  |  |  | 0.8154 | 0.8740 | 0.2333 | 0.2190 |

### D2. Resampling tree versus forwarding

Binary tree of depth L over a uniformly random partition of the mixture into 2^L leaves. Every non-root node draws a fresh sensitivity sample of m = 2000 from its input (leaf block, or union of its children's RC-decoded messages) and sends it, B = 1, over a Bernoulli channel with rho = 0.2; the root outputs the union of its children's decoded messages. eps_l = max over level-l nodes and queries of the node's relative error against its own input. Forwarding: one source sends m (or m/0.8^L) draws through L relays with the same channel. Loss-free reference RMS max_Q error at m = 2000: 0.0614 [0.0601, 0.0626]. Growth of the resampling RMS with L (log-log slope): 0.238 [0.203, 0.274]. 300 runs per depth.

| depth L | resampling: RMS root max_Q err [95% CI] | resampling: q95 | q95 of eps_l per level | bound prod(1+q95 eps_l) - 1 | per-run composition violations | per-edge delivered | forwarding m=2000: delivered (pred 0.8^L) | forwarding: RMS max_Q err | forwarding m/0.8^L: RMS max_Q err [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.0463 [0.0445, 0.0480] | 0.0713 | 0.1114 | 0.1114 | 0 | 0.8002 [0.7995, 0.8009] | 0.8000 (0.8000) | 0.0683 | 0.0600 [0.0576, 0.0623] |
| 2 | 0.0592 [0.0568, 0.0614] | 0.0909 | 0.1256, 0.1153 | 0.2554 | 0 | 0.8002 [0.7998, 0.8006] | 0.6401 (0.6400) | 0.0805 | 0.0617 [0.0593, 0.0640] |
| 3 | 0.0626 [0.0600, 0.0651] | 0.0977 | 0.1372, 0.1378, 0.1238 | 0.4541 | 0 | 0.8001 [0.7998, 0.8004] | 0.5116 (0.5120) | 0.0844 | 0.0608 [0.0581, 0.0634] |
| 4 | 0.0639 [0.0614, 0.0663] | 0.0948 | 0.1466, 0.1312, 0.1244, 0.1081 | 0.6159 | 0 | 0.8001 [0.7999, 0.8003] | 0.4102 (0.4096) | 0.1005 | 0.0612 [0.0585, 0.0637] |

### E. k-median (z = 1) and k-means (z = 2)

Mixture, two-term sensitivity proposal around a D^z seeding, m = 2000 draws, B = 1, Bernoulli loss, RC decoder, costs dist^z via the testbed's Problem objective, 1000 repetitions per cell. Slopes of log RMS on log 1/(1-rho) (predicted 0.5): z=1: Q0 0.505 [0.486, 0.525], max_Q 0.502 [0.491, 0.512]; z=2: Q0 0.506 [0.486, 0.526], max_Q 0.510 [0.499, 0.520];

| z | rho | mean M' | RMS err Q0 [95% CI] | RMS max_Q err [95% CI] | RMS Q0 x sqrt(1-rho) / RMS Q0 at rho=0 | RMS max_Q x sqrt(1-rho) / at rho=0 |
|---|---|---|---|---|---|---|
| 1 | 0.0 | 2000.0 | 0.0151 [0.0144, 0.0158] | 0.0318 [0.0311, 0.0325] | 1.000 | 1.000 |
| 1 | 0.1 | 1800.2 | 0.0160 [0.0153, 0.0166] | 0.0323 [0.0316, 0.0330] | 1.001 | 0.965 |
| 1 | 0.2 | 1600.2 | 0.0172 [0.0165, 0.0179] | 0.0355 [0.0347, 0.0363] | 1.017 | 0.999 |
| 1 | 0.3 | 1399.3 | 0.0180 [0.0171, 0.0188] | 0.0370 [0.0362, 0.0379] | 0.995 | 0.975 |
| 1 | 0.5 | 1000.3 | 0.0229 [0.0217, 0.0239] | 0.0454 [0.0444, 0.0464] | 1.068 | 1.010 |
| 1 | 0.7 | 600.9 | 0.0272 [0.0260, 0.0284] | 0.0574 [0.0560, 0.0588] | 0.986 | 0.989 |
| 1 | 0.8 | 400.4 | 0.0336 [0.0321, 0.0351] | 0.0709 [0.0691, 0.0726] | 0.994 | 0.997 |
| 1 | 0.9 | 199.9 | 0.0496 [0.0475, 0.0515] | 0.0992 [0.0970, 0.1014] | 1.036 | 0.986 |
| 2 | 0.0 | 2000.0 | 0.0287 [0.0275, 0.0299] | 0.0607 [0.0594, 0.0620] | 1.000 | 1.000 |
| 2 | 0.1 | 1800.3 | 0.0292 [0.0279, 0.0304] | 0.0648 [0.0633, 0.0663] | 0.964 | 1.013 |
| 2 | 0.2 | 1600.5 | 0.0316 [0.0302, 0.0329] | 0.0681 [0.0666, 0.0696] | 0.983 | 1.004 |
| 2 | 0.3 | 1400.5 | 0.0338 [0.0323, 0.0353] | 0.0741 [0.0725, 0.0756] | 0.986 | 1.021 |
| 2 | 0.5 | 998.7 | 0.0395 [0.0378, 0.0411] | 0.0861 [0.0842, 0.0879] | 0.972 | 1.002 |
| 2 | 0.7 | 600.0 | 0.0516 [0.0494, 0.0537] | 0.1134 [0.1109, 0.1158] | 0.984 | 1.023 |
| 2 | 0.8 | 399.1 | 0.0648 [0.0619, 0.0676] | 0.1348 [0.1318, 0.1378] | 1.009 | 0.993 |
| 2 | 0.9 | 200.2 | 0.0898 [0.0856, 0.0938] | 0.2004 [0.1954, 0.2052] | 0.989 | 1.044 |

