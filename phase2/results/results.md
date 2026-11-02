# Phase II results - Tongji palmprint database

Generated 2026-10-18 18:32:23 by `run_experiments.py`. Every number below is recomputed from the images; nothing is copied from a paper except where a column says so.

## Protocol

600 palms. Each is enrolled from one image, its first session-1 capture; every other image of it is a probe and is compared with every enrolled palm.

| probe group | probes | genuine comparisons | impostor comparisons |
|---|---|---|---|
| cross | 6,000 | 6,000 | 3,594,000 |
| same | 5,400 | 5,400 | 3,234,600 |

**cross** = taken in a different session from the enrolment; **same** = taken in the same sitting. Every headline figure uses cross probes. The 95% intervals come from 1000 bootstrap replicates that resample palms, shared by every configuration so that differences are paired.

## 1. RLOC against the 2003 baseline

| configuration | EER | 95% interval | d' | FRR @ FAR 1% | FRR @ FAR 0.1% | FRR @ FAR 0.01% | rank-1 |
|---|---|---|---|---|---|---|---|
| Exp 3  + cross neighbourhood  = RLOC | 0.11% | 0.04% - 0.16% | 6.54 | 0.00% | 0.12% | 0.38% | 99.95% |
| Han et al. 2003, Sobel features | 7.67% | 6.84% - 8.66% | 2.91 | 24.33% | 44.15% | 61.40% | 68.85% |

Difference in EER: -7.56 pt, 95% interval -8.54 pt to -6.74 pt; RLOC is better in 100.0% of bootstrap replicates.

### Confusion matrices at the equal-error threshold

**Exp 3  + cross neighbourhood  = RLOC** - threshold 0.681

| truth \ decision | accept | reject | rate |
|---|---|---|---|
| genuine (6,000) | 5,993 | 7 | FRR 0.117% |
| impostor (3,594,000) | 3,972 | 3,590,028 | FAR 0.111% |

**Han et al. 2003, Sobel features** - threshold 0.517

| truth \ decision | accept | reject | rate |
|---|---|---|---|
| genuine (6,000) | 5,539 | 461 | FRR 7.683% |
| impostor (3,594,000) | 275,598 | 3,318,402 | FAR 7.668% |

### Where the errors come from

| configuration | false rejects at the EER threshold | palms they come from (of 600) | worst palms |
|---|---|---|---|
| Exp 3  + cross neighbourhood  = RLOC | 7 | 5 | P014 (2), P069 (2), P025 (1), P272 (1), P519 (1) |
| Han et al. 2003, Sobel features | 461 | 127 | P017 (10), P034 (10), P041 (10), P136 (10), P349 (10) |

When a method's few errors sit on a handful of palms, its EER moves a lot with which palms happen to be in the test - which is why RLOC's interval is wide relative to its EER.

### At a stricter operating point: FAR = 0.1%

| configuration | threshold | genuine rejected | impostors accepted |
|---|---|---|---|
| Exp 3  + cross neighbourhood  = RLOC | 0.682 | 7 of 6,000 (0.12%) | 3,509 of 3,594,000 (0.098%) |
| Han et al. 2003, Sobel features | 0.735 | 2,649 of 6,000 (44.15%) | 3,594 of 3,594,000 (0.100%) |

## 2. Same session against cross-session

| configuration | EER, same session | EER, cross-session |
|---|---|---|
| Exp 3  + cross neighbourhood  = RLOC | 0.02% | 0.11% |
| Han et al. 2003, Sobel features | 3.41% | 7.67% |
| Exp 1  MFRAT, pixel-to-pixel, 1 template | 0.06% | 0.27% |
| Gabor + enlarged set + cross | 0.00% | 0.06% |

Probes taken in the enrolment's own sitting are far easier. Reporting them, or mixing them in, flatters every method.

## 3. Paper 2's Table 4, re-run

| configuration | EER | 95% interval | d' | FRR @ FAR 1% | FRR @ FAR 0.1% | FRR @ FAR 0.01% | rank-1 | Paper 2 (PolyU) |
|---|---|---|---|---|---|---|---|---|
| Exp 1  MFRAT, pixel-to-pixel, 1 template | 0.27% | 0.12% - 0.48% | 4.38 | 0.18% | 0.50% | 1.05% | 99.82% | 0.40% |
| Exp 2  + enlarged training set (7 templates) | 0.12% | 0.05% - 0.15% | 4.60 | 0.00% | 0.12% | 0.32% | 99.93% | 0.36% |
| Exp 3  + cross neighbourhood  = RLOC | 0.11% | 0.04% - 0.16% | 6.54 | 0.00% | 0.12% | 0.38% | 99.95% | 0.16% |
| Exp 4  3x3 square instead of cross | 0.32% | 0.15% - 0.53% | 5.93 | 0.15% | 0.60% | 2.00% | 99.55% | 0.27% |

- **adding the enlarged training set (7 templates)**: -0.15 pt (95% interval -0.40 pt to -0.01 pt; better in 98.3% of replicates)
- **adding the cross neighbourhood**: -0.01 pt (95% interval -0.06 pt to +0.06 pt; better in 49.6% of replicates)
- **3x3 square instead of the cross**: +0.21 pt (95% interval +0.08 pt to +0.38 pt; better in 0.0% of replicates)

## 4. MFRAT against Gabor, everything else identical

The Gabor code is a Competitive-Code-style filter bank feeding exactly the same lattice, rotated templates, matching rule and scorer as MFRAT; the filter is the only difference. Each Gabor configuration's wavelength was chosen with that configuration's own matching, on session-1 data only (1200 probes) - never on the session-2 data reported here. MFRAT's parameters are its paper's, untuned, so if anything the Gabor arm is favoured.

| Gabor omega | session-1 EER (d'), pixel, 1 template | session-1 EER (d'), pixel, 7 templates | session-1 EER (d'), cross, 7 templates |
|---|---|---|---|
| 0.15 | 1.333% (4.78) | 1.339% (4.65) | 1.333% (5.15) |
| 0.20 | 0.417% (5.28) | 0.448% (5.18) | 0.500% (6.12) |
| 0.25 | 0.167% (5.50) | 0.230% (5.43) | 0.211% (6.88) |
| 0.30 | 0.083% (5.52) | 0.083% (5.46) | 0.083% (7.49) |
| 0.40 | 0.083% (5.15) | 0.006% (5.16) | 0.003% (8.21) |
| 0.50 | 0.009% (4.55) **chosen** | 0.001% (4.57) | 0.001% (8.10) |
| 0.60 | 0.026% (3.95) | 0.000% (3.99) **chosen** | 0.000% (7.31) **chosen** |
| 0.70 | 0.020% (3.47) | 0.000% (3.50) | 0.000% (6.39) |
| 0.80 | 0.083% (3.05) | 0.003% (3.08) | 0.001% (5.39) |

Lowest session-1 EER wins; ties go to the larger d'.

| configuration | EER | 95% interval | d' | FRR @ FAR 1% | FRR @ FAR 0.1% | FRR @ FAR 0.01% | rank-1 |
|---|---|---|---|---|---|---|---|
| Exp 1  MFRAT, pixel-to-pixel, 1 template | 0.27% | 0.12% - 0.48% | 4.38 | 0.18% | 0.50% | 1.05% | 99.82% |
| Gabor, pixel-to-pixel, 1 template | 0.17% | 0.07% - 0.23% | 4.92 | 0.02% | 0.23% | 0.67% | 99.87% |
| Exp 2  + enlarged training set (7 templates) | 0.12% | 0.05% - 0.15% | 4.60 | 0.00% | 0.12% | 0.32% | 99.93% |
| Gabor + enlarged training set | 0.05% | 0.02% - 0.12% | 4.60 | 0.00% | 0.05% | 0.12% | 99.97% |
| Exp 3  + cross neighbourhood  = RLOC | 0.11% | 0.04% - 0.16% | 6.54 | 0.00% | 0.12% | 0.38% | 99.95% |
| Gabor + enlarged set + cross | 0.06% | 0.02% - 0.08% | 6.54 | 0.00% | 0.02% | 0.12% | 99.97% |

- **MFRAT instead of Gabor - no tricks**: +0.10 pt (95% interval +0.00 pt to +0.29 pt; better in 1.8% of replicates)
- **MFRAT instead of Gabor - with 7 templates**: +0.07 pt (95% interval -0.00 pt to +0.10 pt; better in 3.0% of replicates)
- **MFRAT instead of Gabor - full pipeline**: +0.06 pt (95% interval +0.01 pt to +0.10 pt; better in 0.6% of replicates)
- **both tricks, applied to the Gabor code**: -0.11 pt (95% interval -0.18 pt to -0.00 pt; better in 97.9% of replicates)
- **both tricks, applied to the MFRAT code**: -0.16 pt (95% interval -0.36 pt to -0.03 pt; better in 99.7% of replicates)

## 5. One enrolment template against seven

| configuration | EER | 95% interval | d' | FRR @ FAR 1% | FRR @ FAR 0.1% | FRR @ FAR 0.01% | rank-1 |
|---|---|---|---|---|---|---|---|
| Exp 3  + cross neighbourhood  = RLOC | 0.11% | 0.04% - 0.16% | 6.54 | 0.00% | 0.12% | 0.38% | 99.95% |
| RLOC with 1 template instead of 7 | 0.25% | 0.11% - 0.36% | 6.38 | 0.10% | 0.37% | 0.85% | 99.80% |
| Exp 2  + enlarged training set (7 templates) | 0.12% | 0.05% - 0.15% | 4.60 | 0.00% | 0.12% | 0.32% | 99.93% |
| Exp 1  MFRAT, pixel-to-pixel, 1 template | 0.27% | 0.12% - 0.48% | 4.38 | 0.18% | 0.50% | 1.05% | 99.82% |

- **adding the enlarged training set (7 templates)**: -0.15 pt (95% interval -0.40 pt to -0.01 pt; better in 98.3% of replicates)
- **1 template instead of 7**: +0.14 pt (95% interval +0.03 pt to +0.22 pt; better in 0.1% of replicates)

## 6. Paper 1's two feature types

| configuration | EER | 95% interval | d' | FRR @ FAR 1% | FRR @ FAR 0.1% | FRR @ FAR 0.01% | rank-1 |
|---|---|---|---|---|---|---|---|
| Han et al. 2003, Sobel features | 7.67% | 6.84% - 8.66% | 2.91 | 24.33% | 44.15% | 61.40% | 68.85% |
| Han et al. 2003, morphology features | 11.38% | 10.46% - 12.38% | 2.46 | 39.93% | 66.12% | 82.73% | 46.67% |

- **Paper 1: morphology instead of Sobel**: +3.71 pt (95% interval +2.77 pt to +4.76 pt; better in 0.0% of replicates)

## 7. What it costs (this computer)

| step | time |
|---|---|
| MFRAT code, one ROI | 4.2 ms |
| Gabor code, one ROI (FFT, omega 0.60) | 3.5 ms |
| MFRAT enrolment, 7 rotated codes | 36.4 ms |
| Paper 1 Sobel features, one ROI | 2.8 ms |
| Paper 1 morphology features, one ROI | 1.2 ms |
| RLOC match, one probe against one palm (7 templates) | 2.80 ms |
| Han match, one probe against one palm | 0.043 ms |
| RLOC, every probe against every palm, batched | 176 s |

Template size: RLOC 2317 bytes per palm (7 codes of 32 x 32 values from 6 directions, 331 bytes each if packed); Han 256 bytes (64 floats).

## 8. The matcher as published, against our first version

Paper 2 scores a pair as the larger of s(probe, template) and s(template, probe) - equation (9) - and searches +/-2 code pixels of translation. Our first version kept one direction and searched +/-1, to save matching time. Every other configuration in this report matches as the paper does; these rows show what the two details are worth.

| configuration | EER | 95% interval | d' | FRR @ FAR 1% | FRR @ FAR 0.1% | FRR @ FAR 0.01% | rank-1 |
|---|---|---|---|---|---|---|---|
| Exp 3  + cross neighbourhood  = RLOC | 0.11% | 0.04% - 0.16% | 6.54 | 0.00% | 0.12% | 0.38% | 99.95% |
| RLOC, one direction only (no eq. 9 maximum) | 0.16% | 0.06% - 0.24% | 6.33 | 0.00% | 0.20% | 0.67% | 99.95% |
| RLOC, +/-1 shift search instead of +/-2 | 0.63% | 0.25% - 1.02% | 6.10 | 0.57% | 0.93% | 1.35% | 99.25% |
| RLOC as we first built it (+/-1 shift, one direction) | 0.69% | 0.27% - 1.10% | 5.95 | 0.60% | 1.03% | 1.67% | 99.20% |

- **eq. 9's maximum over both directions**: -0.05 pt (95% interval -0.10 pt to -0.00 pt; better in 98.3% of replicates)
- **+/-2 shift search instead of +/-1**: -0.52 pt (95% interval -0.87 pt to -0.16 pt; better in 100.0% of replicates)
- **the matcher as published, against our first version**: -0.58 pt (95% interval -0.97 pt to -0.18 pt; better in 100.0% of replicates)

## Figures

- `figures/det_main.png`
- `figures/det_ablation.png`
- `figures/det_filter.png`
- `figures/det_matcher.png`
- `figures/far_frr_rloc.png`
- `figures/far_frr_han.png`
- `figures/hist_main.png`
- `figures/eer_all.png`
