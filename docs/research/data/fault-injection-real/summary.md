# Fault-injection results (scale full, seed 42)

## Verdicts

| id | prediction | measured | verdict |
|---|---|---|---|
| REAL/covertype-k10/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 0.985, 0.945, 0.993 | **PASS** |
| REAL/covertype-k10/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 2.70 (exact law 2.65); c = 0.361 at phi=1/8 with s_max/S = 0.000476 | **FAIL** |
| REAL/covertype-k10/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 1.0060 (no loss 1.0063, max 1.0102) | **PASS** |
| REAL/covertype-k10/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.58; P[> 2 \| whole] = 0.00; max ds 1.15 | **FAIL** |
| REAL/covertype-k10/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 0.71 | **PASS** |
| REAL/covertype-k10/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.428 [0.273, 0.584]; m=4000: 0.503 [0.352, 0.655] | **PASS** |
| REAL/covertype-k10/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope -0.194 [-0.373, -0.008] | **FAIL** |
| REAL/covertype-k10/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 1.045; sd 0.0545 vs 0.0464 | **PASS** |
| REAL/covertype-k10/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0259 (B2 0.0241), merged 0.0249 (B2 0.0260), floor 0.0098, kappa_pt 0.000222 | **PASS** |
| REAL/covertype-k10/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | covertype-k10: m=1000 slope 0.46 [-0.14, 1.03], m=4000 -0.29 [-0.93, 0.40] | **INCONCLUSIVE** |
| REAL/covertype-k10/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.034 (S/m) sum kappa, R^2 0.957 | **PASS** |
| REAL/covertype-k25/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 1.124, 1.082, 0.875 | **PASS** |
| REAL/covertype-k25/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 1.90 (exact law 2.65); c = 0.129 at phi=1/8 with s_max/S = 0.0119 | **FAIL** |
| REAL/covertype-k25/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 1.0093 (no loss 1.0093, max 1.0203) | **PASS** |
| REAL/covertype-k25/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.92; P[> 2 \| whole] = 0.00; max ds 1.38 | **FAIL** |
| REAL/covertype-k25/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 0.85 | **PASS** |
| REAL/covertype-k25/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.537 [0.401, 0.671]; m=4000: 0.570 [0.460, 0.672] | **PASS** |
| REAL/covertype-k25/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope -0.013 [-0.163, 0.141] | **PASS** |
| REAL/covertype-k25/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 1.016; sd 0.0587 vs 0.0578 | **PASS** |
| REAL/covertype-k25/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0318 (B2 0.0308), merged 0.0328 (B2 0.0346), floor 0.0157, kappa_pt 0.000574 | **PASS** |
| REAL/covertype-k25/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | covertype-k25: m=1000 slope 0.05 [-0.50, 0.61], m=4000 0.14 [-0.30, 0.53] | **PASS** |
| REAL/covertype-k25/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.024 (S/m) sum kappa, R^2 0.979 | **PASS** |
| REAL/census1990-k10/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 0.981, 0.953, 1.029 | **PASS** |
| REAL/census1990-k10/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 2.76 (exact law 2.65); c = 0.093 at phi=1/8 with s_max/S = 1.21e-05 | **FAIL** |
| REAL/census1990-k10/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 0.9915 (no loss 0.9904, max 1.0051) | **PASS** |
| REAL/census1990-k10/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.30; P[> 2 \| whole] = 0.00; max ds 1.15 | **FAIL** |
| REAL/census1990-k10/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 0.85 | **PASS** |
| REAL/census1990-k10/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.425 [0.299, 0.544]; m=4000: 0.490 [0.375, 0.616] | **PASS** |
| REAL/census1990-k10/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope 0.035 [-0.102, 0.181] | **PASS** |
| REAL/census1990-k10/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 1.051; sd 0.0235 vs 0.0209 | **PASS** |
| REAL/census1990-k10/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0137 (B2 0.0158), merged 0.0181 (B2 0.0158), floor 0.0007, kappa_pt 9.98e-07 | **PASS** |
| REAL/census1990-k10/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | census1990-500k-k10: m=1000 slope 0.34 [-0.11, 0.79], m=4000 -0.22 [-0.66, 0.23] | **INCONCLUSIVE** |
| REAL/census1990-k10/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.014 (S/m) sum kappa, R^2 0.960 | **PASS** |
| REAL/census1990-k25/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 0.988, 0.897, 0.954 | **PASS** |
| REAL/census1990-k25/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 2.76 (exact law 2.65); c = 0.110 at phi=1/8 with s_max/S = 7.28e-06 | **FAIL** |
| REAL/census1990-k25/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 1.0148 (no loss 1.0116, max 1.0356) | **PASS** |
| REAL/census1990-k25/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.90; P[> 2 \| whole] = 0.00; max ds 1.41 | **FAIL** |
| REAL/census1990-k25/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 0.97 | **PASS** |
| REAL/census1990-k25/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.433 [0.314, 0.558]; m=4000: 0.549 [0.434, 0.675] | **PASS** |
| REAL/census1990-k25/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope -0.155 [-0.295, -0.021] | **FAIL** |
| REAL/census1990-k25/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 0.983; sd 0.0321 vs 0.0283 | **PASS** |
| REAL/census1990-k25/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0129 (B2 0.0157), merged 0.0145 (B2 0.0157), floor 0.0006, kappa_pt 9.00e-07 | **PASS** |
| REAL/census1990-k25/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | census1990-500k-k25: m=1000 slope 0.30 [-0.26, 0.91], m=4000 -0.42 [-0.85, 0.03] | **INCONCLUSIVE** |
| REAL/census1990-k25/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.009 (S/m) sum kappa, R^2 0.972 | **PASS** |
| REAL/kddcup99-k10/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 0.960, 1.013, 0.968 | **PASS** |
| REAL/kddcup99-k10/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 2.63 (exact law 2.65); c = 0.508 at phi=1/8 with s_max/S = 0.00735 | **FAIL** |
| REAL/kddcup99-k10/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 1.0169 (no loss 1.0119, max 1.0621) | **PASS** |
| REAL/kddcup99-k10/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.58; P[> 2 \| whole] = 0.00; max ds 1.86 | **FAIL** |
| REAL/kddcup99-k10/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 1.03 | **PASS** |
| REAL/kddcup99-k10/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.586 [0.459, 0.710]; m=4000: 0.572 [0.432, 0.699] | **PASS** |
| REAL/kddcup99-k10/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope 0.124 [-0.026, 0.286] | **PASS** |
| REAL/kddcup99-k10/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 0.959; sd 0.0669 vs 0.0701 | **PASS** |
| REAL/kddcup99-k10/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0324 (B2 0.0294), merged 0.0953 (B2 0.0878), floor 0.0827, kappa_pt 0.0160 | **PASS** |
| REAL/kddcup99-k10/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | kddcup99-500k-k10: m=1000 slope 0.54 [0.08, 1.04], m=4000 0.14 [-0.38, 0.60] | **FAIL** |
| REAL/kddcup99-k10/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.051 (S/m) sum kappa, R^2 0.955 | **PASS** |
| REAL/kddcup99-k25/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 0.876, 1.048, 1.028 | **PASS** |
| REAL/kddcup99-k25/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 3.22 (exact law 2.65); c = 0.146 at phi=1/8 with s_max/S = 0.0357 | **PASS** |
| REAL/kddcup99-k25/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 1.0038 (no loss 0.9828, max 1.1248) | **PASS** |
| REAL/kddcup99-k25/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.60 vs P[whole cluster lost] = 0.82; P[> 2 \| whole] = 0.73; max ds 6.95 | **FAIL** |
| REAL/kddcup99-k25/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 0.97 | **PASS** |
| REAL/kddcup99-k25/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.516 [0.388, 0.635]; m=4000: 0.542 [0.419, 0.662] | **PASS** |
| REAL/kddcup99-k25/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope 0.030 [-0.112, 0.174] | **PASS** |
| REAL/kddcup99-k25/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 1.020; sd 0.0785 vs 0.0776 | **PASS** |
| REAL/kddcup99-k25/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0307 (B2 0.0309), merged 0.0604 (B2 0.0560), floor 0.0468, kappa_pt 0.00510 | **PASS** |
| REAL/kddcup99-k25/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | kddcup99-500k-k25: m=1000 slope 0.24 [-0.17, 0.71], m=4000 0.12 [-0.31, 0.54] | **PASS** |
| REAL/kddcup99-k25/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.036 (S/m) sum kappa, R^2 0.945 | **PASS** |
| REAL/higgs-1m-k10/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 1.004, 0.987, 0.932 | **PASS** |
| REAL/higgs-1m-k10/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 2.57 (exact law 2.65); c = 0.121 at phi=1/8 with s_max/S = 8.96e-06 | **FAIL** |
| REAL/higgs-1m-k10/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 1.0045 (no loss 1.0037, max 1.0070) | **PASS** |
| REAL/higgs-1m-k10/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.45; P[> 2 \| whole] = 0.00; max ds 1.12 | **FAIL** |
| REAL/higgs-1m-k10/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 0.86 | **PASS** |
| REAL/higgs-1m-k10/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.416 [0.296, 0.542]; m=4000: 0.425 [0.307, 0.543] | **PASS** |
| REAL/higgs-1m-k10/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope 0.018 [-0.115, 0.157] | **PASS** |
| REAL/higgs-1m-k10/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 1.035; sd 0.0177 vs 0.0172 | **PASS** |
| REAL/higgs-1m-k10/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0146 (B2 0.0124), merged 0.0118 (B2 0.0124), floor 0.0008, kappa_pt 1.62e-06 | **PASS** |
| REAL/higgs-1m-k10/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | higgs-500k-k10: m=1000 slope 0.05 [-0.35, 0.47], m=4000 0.58 [0.05, 1.15] | **FAIL** |
| REAL/higgs-1m-k10/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.008 (S/m) sum kappa, R^2 0.983 | **PASS** |
| REAL/higgs-1m-k25/A1 kappa law | RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2 | calibration 0.941, 1.040, 0.967 | **PASS** |
| REAL/higgs-1m-k25/A1 s_max shape | mean \|r\| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10% | ratio 2.76 (exact law 2.65); c = 0.127 at phi=1/8 with s_max/S = 7.76e-06 | **FAIL** |
| REAL/higgs-1m-k25/A1 downstream | downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node) | 1.0155 (no loss 1.0139, max 1.0195) | **PASS** |
| REAL/higgs-1m-k25/A1' cluster-sorted | downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency | P[ds > 2] = 0.00 vs P[whole cluster lost] = 1.00; P[> 2 \| whole] = 0.00; max ds 1.14 | **FAIL** |
| REAL/higgs-1m-k25/F2/A3 | coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5) | 8/8 loss-rate cells within 20%; error ratio 1.11 | **PASS** |
| REAL/higgs-1m-k25/F3/B1 slope | per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE) | m=1000: 0.403 [0.269, 0.536]; m=4000: 0.465 [0.340, 0.592] | **PASS** |
| REAL/higgs-1m-k25/F3/C3 rateless | rateless distortion independent of rho (\|slope\| < 0.1 or CI contains 0) | m=1000: slope -0.022 [-0.175, 0.144] | **PASS** |
| REAL/higgs-1m-k25/F3/B1 bursts | GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw | GE/Bernoulli mean excess 1.035; sd 0.0171 vs 0.0161 | **PASS** |
| REAL/higgs-1m-k25/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2 | split 0.0114 (B2 0.0126), merged 0.0116 (B2 0.0126), floor 0.0008, kappa_pt 1.65e-06 | **PASS** |
| REAL/higgs-1m-k25/F5/E1 random order | Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap) | higgs-500k-k25: m=1000 slope 0.12 [-0.40, 0.72], m=4000 0.03 [-0.49, 0.58] | **PASS** |
| REAL/higgs-1m-k25/F5/E1 kappa | Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported | Var = 0.005 (S/m) sum kappa, R^2 0.990 | **PASS** |
| REAL/higgs-k10/F3/B1 single rho | excess distortion x(1-rho)^-1/2 = x1.195 at rho = 0.3, per-draw and merged (HIGGS, all 11M rows) | m=1000 per-draw: excess x1.490; m=1000 merged: excess x1.358; m=4000 per-draw: excess x1.266; m=4000 merged: excess x1.323;  | **FAIL** |
| REAL/higgs-k10/F3/B2 formula | erasure RMS at rho = 0.3, m = 4000: split and merged within 35% of Theorem B2 | split 0.0111 (B2 0.0124), merged 0.0128 (B2 0.0124), floor 0.0003, kappa_pt 1.59e-07 | **PASS** |

## Real datasets (z-scored; reference = best of 10 Lloyd restarts)

| instance | n | d | k | s_max/S | max kappa_pt | smallest cluster | prepare s |
|---|---|---|---|---|---|---|---|
| covertype-k10 | 581012 | 54 | 10 | 0.000476 | 0.000222 | 6575 | 15 |
| covertype-k25 | 581012 | 54 | 25 | 0.0119 | 0.000574 | 474 | 26 |
| census1990-k10 | 2458285 | 68 | 10 | 1.21e-05 | 9.98e-07 | 105494 | 74 |
| census1990-500k-k10 | 500000 | 68 | 10 | 1.93e-05 | 4.26e-06 | 785 | 14 |
| census1990-k25 | 2458285 | 68 | 25 | 7.28e-06 | 9.00e-07 | 3711 | 157 |
| census1990-500k-k25 | 500000 | 68 | 25 | 4.65e-05 | 4.58e-06 | 352 | 31 |
| kddcup99-k10 | 4898431 | 38 | 10 | 0.00735 | 0.0160 | 1 | 52 |
| kddcup99-500k-k10 | 500000 | 38 | 10 | 0.0192 | 0.0108 | 4 | 5 |
| kddcup99-k25 | 4898431 | 38 | 25 | 0.0357 | 0.00510 | 1 | 139 |
| kddcup99-500k-k25 | 500000 | 38 | 25 | 0.0357 | 0.0118 | 1 | 9 |
| higgs-1m-k10 | 1000000 | 28 | 10 | 8.96e-06 | 1.62e-06 | 21955 | 17 |
| higgs-500k-k10 | 500000 | 28 | 10 | 2.11e-05 | 3.51e-06 | 10669 | 8 |
| higgs-1m-k25 | 1000000 | 28 | 25 | 7.76e-06 | 1.65e-06 | 5150 | 42 |
| higgs-500k-k25 | 500000 | 28 | 25 | 7.75e-05 | 3.41e-06 | 2289 | 22 |
| higgs-k10 | 11000000 | 28 | 10 | 1.75e-06 | 1.59e-07 | 240020 | 197 |

| check | covertype-k10 | covertype-k25 | census1990-k10 | census1990-k25 | kddcup99-k10 | kddcup99-k25 | higgs-1m-k10 | higgs-1m-k25 |
|---|---|---|---|---|---|---|---|---|
| A1 kappa law | **PASS** calibration 0.985, 0.945, 0.993 | **PASS** calibration 1.124, 1.082, 0.875 | **PASS** calibration 0.981, 0.953, 1.029 | **PASS** calibration 0.988, 0.897, 0.954 | **PASS** calibration 0.960, 1.013, 0.968 | **PASS** calibration 0.876, 1.048, 1.028 | **PASS** calibration 1.004, 0.987, 0.932 | **PASS** calibration 0.941, 1.040, 0.967 |
| A1 s_max shape | **FAIL** ratio 2.70 (exact law 2.65); c = 0.361 at phi=1/8 with s_max/S = 0.000476 | **FAIL** ratio 1.90 (exact law 2.65); c = 0.129 at phi=1/8 with s_max/S = 0.0119 | **FAIL** ratio 2.76 (exact law 2.65); c = 0.093 at phi=1/8 with s_max/S = 1.21e-05 | **FAIL** ratio 2.76 (exact law 2.65); c = 0.110 at phi=1/8 with s_max/S = 7.28e-06 | **FAIL** ratio 2.63 (exact law 2.65); c = 0.508 at phi=1/8 with s_max/S = 0.00735 | **PASS** ratio 3.22 (exact law 2.65); c = 0.146 at phi=1/8 with s_max/S = 0.0357 | **FAIL** ratio 2.57 (exact law 2.65); c = 0.121 at phi=1/8 with s_max/S = 8.96e-06 | **FAIL** ratio 2.76 (exact law 2.65); c = 0.127 at phi=1/8 with s_max/S = 7.76e-06 |
| A1 downstream | **PASS** 1.0060 (no loss 1.0063, max 1.0102) | **PASS** 1.0093 (no loss 1.0093, max 1.0203) | **PASS** 0.9915 (no loss 0.9904, max 1.0051) | **PASS** 1.0148 (no loss 1.0116, max 1.0356) | **PASS** 1.0169 (no loss 1.0119, max 1.0621) | **PASS** 1.0038 (no loss 0.9828, max 1.1248) | **PASS** 1.0045 (no loss 1.0037, max 1.0070) | **PASS** 1.0155 (no loss 1.0139, max 1.0195) |
| A1' cluster-sorted | **FAIL** P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.58; P[> 2 \| whole] = 0.00; max ds 1.15 | **FAIL** P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.92; P[> 2 \| whole] = 0.00; max ds 1.38 | **FAIL** P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.30; P[> 2 \| whole] = 0.00; max ds 1.15 | **FAIL** P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.90; P[> 2 \| whole] = 0.00; max ds 1.41 | **FAIL** P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.58; P[> 2 \| whole] = 0.00; max ds 1.86 | **FAIL** P[ds > 2] = 0.60 vs P[whole cluster lost] = 0.82; P[> 2 \| whole] = 0.73; max ds 6.95 | **FAIL** P[ds > 2] = 0.00 vs P[whole cluster lost] = 0.45; P[> 2 \| whole] = 0.00; max ds 1.12 | **FAIL** P[ds > 2] = 0.00 vs P[whole cluster lost] = 1.00; P[> 2 \| whole] = 0.00; max ds 1.14 |
| F2/A3 | **PASS** 8/8 loss-rate cells within 20%; error ratio 0.71 | **PASS** 8/8 loss-rate cells within 20%; error ratio 0.85 | **PASS** 8/8 loss-rate cells within 20%; error ratio 0.85 | **PASS** 8/8 loss-rate cells within 20%; error ratio 0.97 | **PASS** 8/8 loss-rate cells within 20%; error ratio 1.03 | **PASS** 8/8 loss-rate cells within 20%; error ratio 0.97 | **PASS** 8/8 loss-rate cells within 20%; error ratio 0.86 | **PASS** 8/8 loss-rate cells within 20%; error ratio 1.11 |
| F3/B1 slope | **PASS** m=1000: 0.428 [0.273, 0.584]; m=4000: 0.503 [0.352, 0.655] | **PASS** m=1000: 0.537 [0.401, 0.671]; m=4000: 0.570 [0.460, 0.672] | **PASS** m=1000: 0.425 [0.299, 0.544]; m=4000: 0.490 [0.375, 0.616] | **PASS** m=1000: 0.433 [0.314, 0.558]; m=4000: 0.549 [0.434, 0.675] | **PASS** m=1000: 0.586 [0.459, 0.710]; m=4000: 0.572 [0.432, 0.699] | **PASS** m=1000: 0.516 [0.388, 0.635]; m=4000: 0.542 [0.419, 0.662] | **PASS** m=1000: 0.416 [0.296, 0.542]; m=4000: 0.425 [0.307, 0.543] | **PASS** m=1000: 0.403 [0.269, 0.536]; m=4000: 0.465 [0.340, 0.592] |
| F3/C3 rateless | **FAIL** m=1000: slope -0.194 [-0.373, -0.008] | **PASS** m=1000: slope -0.013 [-0.163, 0.141] | **PASS** m=1000: slope 0.035 [-0.102, 0.181] | **FAIL** m=1000: slope -0.155 [-0.295, -0.021] | **PASS** m=1000: slope 0.124 [-0.026, 0.286] | **PASS** m=1000: slope 0.030 [-0.112, 0.174] | **PASS** m=1000: slope 0.018 [-0.115, 0.157] | **PASS** m=1000: slope -0.022 [-0.175, 0.144] |
| F3/B1 bursts | **PASS** GE/Bernoulli mean excess 1.045; sd 0.0545 vs 0.0464 | **PASS** GE/Bernoulli mean excess 1.016; sd 0.0587 vs 0.0578 | **PASS** GE/Bernoulli mean excess 1.051; sd 0.0235 vs 0.0209 | **PASS** GE/Bernoulli mean excess 0.983; sd 0.0321 vs 0.0283 | **PASS** GE/Bernoulli mean excess 0.959; sd 0.0669 vs 0.0701 | **PASS** GE/Bernoulli mean excess 1.020; sd 0.0785 vs 0.0776 | **PASS** GE/Bernoulli mean excess 1.035; sd 0.0177 vs 0.0172 | **PASS** GE/Bernoulli mean excess 1.035; sd 0.0171 vs 0.0161 |
| F3/B2 formula | **PASS** split 0.0259 (B2 0.0241), merged 0.0249 (B2 0.0260), floor 0.0098, kappa_pt 0.000222 | **PASS** split 0.0318 (B2 0.0308), merged 0.0328 (B2 0.0346), floor 0.0157, kappa_pt 0.000574 | **PASS** split 0.0137 (B2 0.0158), merged 0.0181 (B2 0.0158), floor 0.0007, kappa_pt 9.98e-07 | **PASS** split 0.0129 (B2 0.0157), merged 0.0145 (B2 0.0157), floor 0.0006, kappa_pt 9.00e-07 | **PASS** split 0.0324 (B2 0.0294), merged 0.0953 (B2 0.0878), floor 0.0827, kappa_pt 0.0160 | **PASS** split 0.0307 (B2 0.0309), merged 0.0604 (B2 0.0560), floor 0.0468, kappa_pt 0.00510 | **PASS** split 0.0146 (B2 0.0124), merged 0.0118 (B2 0.0124), floor 0.0008, kappa_pt 1.62e-06 | **PASS** split 0.0114 (B2 0.0126), merged 0.0116 (B2 0.0126), floor 0.0008, kappa_pt 1.65e-06 |
| F5/E1 random order | **INCONCLUSIVE** covertype-k10: m=1000 slope 0.46 [-0.14, 1.03], m=4000 -0.29 [-0.93, 0.40] | **PASS** covertype-k25: m=1000 slope 0.05 [-0.50, 0.61], m=4000 0.14 [-0.30, 0.53] | **INCONCLUSIVE** census1990-500k-k10: m=1000 slope 0.34 [-0.11, 0.79], m=4000 -0.22 [-0.66, 0.23] | **INCONCLUSIVE** census1990-500k-k25: m=1000 slope 0.30 [-0.26, 0.91], m=4000 -0.42 [-0.85, 0.03] | **FAIL** kddcup99-500k-k10: m=1000 slope 0.54 [0.08, 1.04], m=4000 0.14 [-0.38, 0.60] | **PASS** kddcup99-500k-k25: m=1000 slope 0.24 [-0.17, 0.71], m=4000 0.12 [-0.31, 0.54] | **FAIL** higgs-500k-k10: m=1000 slope 0.05 [-0.35, 0.47], m=4000 0.58 [0.05, 1.15] | **PASS** higgs-500k-k25: m=1000 slope 0.12 [-0.40, 0.72], m=4000 0.03 [-0.49, 0.58] |
| F5/E1 kappa | **PASS** Var = 0.034 (S/m) sum kappa, R^2 0.957 | **PASS** Var = 0.024 (S/m) sum kappa, R^2 0.979 | **PASS** Var = 0.014 (S/m) sum kappa, R^2 0.960 | **PASS** Var = 0.009 (S/m) sum kappa, R^2 0.972 | **PASS** Var = 0.051 (S/m) sum kappa, R^2 0.955 | **PASS** Var = 0.036 (S/m) sum kappa, R^2 0.945 | **PASS** Var = 0.008 (S/m) sum kappa, R^2 0.983 | **PASS** Var = 0.005 (S/m) sum kappa, R^2 0.990 |

Predictions:

- A1 kappa law: RMS r(Q) = sqrt(phi kappa_pt(Q)/(1-phi)) per candidate (random partition, exact data): calibration within 0.85-1.15 at phi = 1/8, 1/4, 1/2
- A1 s_max shape: mean |r| linear in sqrt(phi)/(1-phi): ratio phi=1/2 : 1/8 = 3.50 +/- 10%
- A1 downstream: downstream ratio < 1.02 at phi = 1/8 (random partition, sensitivity m=500/node)
- A1' cluster-sorted: downstream > 2 whenever a lost node held >= 90% of a cluster, with that frequency
- F2/A3: coreset loss rate = f^(r+1) (within 20%) and error = A1 at phi_eff (ratio 0.67-1.5)
- F3/B1 slope: per-draw + realized decoder: slope of log(distortion-1) vs log 1/(1-rho) = 0.5 +/- 0.1 at m = 1000 and 4000 (PASS: both points within; FAIL: a 95% bootstrap CI excludes 0.5; else INCONCLUSIVE)
- F3/C3 rateless: rateless distortion independent of rho (|slope| < 0.1 or CI contains 0)
- F3/B1 bursts: GE (mean burst 20) mean excess within 15% of Bernoulli at rho = 0.3, per-draw
- F3/B2 formula: erasure RMS at rho = 0.3, m = 4000, for the max-kappa_pt candidate: split and merged within 35% of Theorem B2
- F5/E1 random order: Var r(Q) flat in L for L >= 3: slope of log Var vs log(L+1) within 0 +/- 0.3 over blocks 8-128 at m = 1000 and 4000 (same PASS/FAIL/INCONCLUSIVE rule, seed bootstrap)
- F5/E1 kappa: Var r(Q) proportional to (S/m) sum_l kappa_l (through-origin R^2 >= 0.9); coefficient reported


