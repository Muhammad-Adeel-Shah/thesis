# P2 report — streams, fusion and the go / no-go gate

**Project:** Decoder-Bank Probing (DBP) · 24i-8017, MS Data Science, FAST-NUCES
**Date:** 2026-10-05 · **Branch:** `p2-streams`
**Analysis plan:** `docs/P2_PREREGISTRATION.md` with Amendment 1 (`I-047`) and
Amendment 2 (commit `5a0b250`, committed before any panel score).
**Code that produced every panel number:** commit `5e76fed`, run once by
`scripts/p2_session_e.py real` on CPU (marker `SESSION_E_REAL_STARTED.json` → `_DONE.json`).

Every number below is read from a committed file under
`reports/p2_session_e/real/` (panel) or `reports/p2_session_e/validation/` (validation);
the file is named beside each table. Points are percentage points; CIs are 95% paired
bootstrap percentile intervals, 1,000 resamples, seed 2026, fakes resampled within slice
and reals within the view's pool (Amendment 2, I-061).

---

## 1. Verdict

**RQ1 gate (governing protocol P-A, Amendment 2): NO-GO.** Both seeds fail; GO needed
both. The governing fusion F-mean(A2) (S1 + S2 + S3-Set, robust z on validation reals)
does not beat the best single head by ≥ 2.0 points on M1 or M2 with a CI excluding 0.
The best single head is **S2** in every run, on both metrics, in 100% of bootstrap
replicates.

Source: `real/analysis/<run id>/go_no_go.json` (`governing`, `analysis_a`) and
`real/analysis/p2_verdict_p_a_sd1.json`.

| run | metric | F-mean(A2) | best single head | Δ points [95% CI] | passes |
|---|---|---:|---|---:|---|
| **p_a_sd1 seed 0** | M1 (TPR@5%FPR) | 87.63 | S2 88.52 | **−0.89** [−1.77, −0.08] | no |
| | M2 (AUC) | 74.46 | S2 80.36 | **−5.89** [−6.96, −4.80] | no |
| **p_a_sd1 seed 1** | M1 | 91.58 | S2 89.92 | **+1.66** [+1.00, +2.32] | no (< 2.0) |
| | M2 | 75.10 | S2 80.13 | **−5.03** [−6.14, −3.98] | no |
| **P-A verdict** | | | | both seeds NO-GO | **NO-GO** |

**Analysis A — the original §6, reported unchanged, no gate role** (F-stack over all five
heads vs the best single head fixed on validation AUC; that head is S3-LR in both seeds,
validation AUC 1.000):

| run | metric | F-stack | S3-LR (fixed on validation) | Δ points [95% CI] | passes |
|---|---|---:|---:|---:|---|
| p_a_sd1 seed 0 | M1 | 35.88 | 31.97 | +3.91 [+3.35, +4.47] | **yes** |
| | M2 | 55.70 | 54.18 | +1.52 [+1.45, +1.60] | no |
| p_a_sd1 seed 1 | M1 | 33.68 | 31.97 | +1.71 [+1.33, +2.11] | no |
| | M2 | 54.97 | 54.18 | +0.79 [+0.75, +0.83] | no |

Analysis A meets the original criterion on seed 0 (M1) and not on seed 1. It does so as
I-060 predicted: F-stack is ~97% S3-LR, and both sit **53–57 points below S2** on M1
(35.9 and 32.0 vs 88.5). The original §6 would have compared the fusion with the
weakest-but-one head and called it a gain. This comparison was demoted by Amendment 2
**before** any panel number existed; it is reported, not used.

**Secondary protocols** (same procedure, no gate role; unseen slices are those the
protocol did not train on):

| run | M1: F-mean(A2) vs S2, Δ [CI] | M2: F-mean(A2) vs S2, Δ [CI] | governing rule | Analysis A |
|---|---|---|---|---|
| 4ch_only seed 0 (11 / 6 slices) | 90.28 vs 84.39, **+5.89** [+5.11, +6.64] | 71.11 vs 76.99, **−5.88** [−6.97, −4.88] | "GO" on M1 only | NO-GO |
| full_bank seed 0 (9 / 4 slices) | 95.33 vs 89.14, **+6.19** [+5.41, +6.94] | 73.14 vs 77.40, **−4.26** [−5.93, −2.63] | "GO" on M1 only | GO on M1 (F-stack +3.78 [+3.01, +4.58]) |

In both secondary protocols M1 and M2 disagree in sign. That is recorded, not resolved
(§6 clause 3). **Read the M1 gains against §3 below**: on these protocols F-mean(A2)
flags more panel reals than S2 does, so part of its M1 "gain" is false positives.

### What the verdict says

On unseen generators, content and degradations, **S2 (the LoRA-tuned DINOv3) is the best
single detector, and fusing it with S1 and S3-Set lowers AUC by 4–6 points in every
run.** The fusion's M1 advantage, where it exists, does not survive the threshold-shift
reading of §3. Following §1 of the pre-registration, the NO-GO points P3 at the
deployment-oriented study of **the best single stream (S2) plus calibration**, and §3
shows calibration is the problem that study must solve.

---

## 2. Macro per head (P-A)

Source: `real/analysis/p_a_sd1_seed<k>/go_no_go.json`, `macro_per_stream`.

| head | M1 seed 0 [CI] | M2 seed 0 [CI] | M1 seed 1 | M2 seed 1 |
|---|---|---|---:|---:|
| S1 (frozen probe) | 71.78 [70.69, 72.76] | 66.40 [65.01, 67.74] | 71.78 | 66.40 |
| **S2 (LoRA)** | **88.52** [87.92, 89.18] | **80.36** [79.37, 81.37] | **89.92** | **80.13** |
| S3-LR | 31.97 [30.66, 33.24] | 54.18 [52.68, 55.67] | 31.97 | 54.18 |
| S3-Set | 67.41 [66.17, 68.58] | 68.17 [66.83, 69.61] | 69.36 | 68.77 |
| AEROBLADE | 58.07 [56.70, 59.25] | 59.61 [57.90, 61.26] | 58.07 | 59.61 |
| F-mean(A2) | 87.63 [86.82, 88.38] | 74.46 [73.23, 75.61] | 91.58 | 75.10 |
| F-mean (A, temperature) | 38.87 | 56.63 | 37.65 | 56.33 |
| F-stack (A) | 35.88 | 55.70 | 33.68 | 54.97 |

S1, S3-LR and AEROBLADE do not depend on the S2 seed; S3-Set differs between seeds
only through the seed of its own training.

---

## 3. The validation thresholds do not hold on the panel's real images (I-064)

Every threshold was fixed at 5% FPR on the 491 validation reals (I-062); on validation it
sits at exactly 4.89%. On the panel it holds only where the reals look like the training
reals. Share of each view's **real** images called fake at that threshold, P-A seed 0
(source: `real/analysis/p_a_sd1_seed0/metrics.csv`, column `fpr_at_5pct_threshold`):

| real pool | S1 | S2 | S3-Set | F-mean(A2) | F-stack |
|---|---:|---:|---:|---:|---:|
| E7 COCO reals (486) | 6.2 | 5.1 | 5.8 | 4.1 | 5.1 |
| E7 luma reals (486) | 21.0 | 8.2 | 20.6 | 12.1 | 7.4 |
| E3 reals (477) | 60.2 | 68.8 | 53.0 | 74.2 | 34.6 |
| E4 WildRF reals (743) | 65.1 | 83.0 | 49.0 | 75.9 | 26.6 |
| E5 Chameleon reals (1,864) | 67.4 | 83.9 | 52.2 | 79.3 | 24.9 |
| T-ctrl reals (200, diagnostic) | 53.5 | 42.5 | 12.0 | 44.0 | 13.5 |

So **M1 is TPR at the validation threshold, not TPR at 5% FPR on the panel.** On E3,
E4, E5 and E8 (E8 has no reals to show it) a high M1 is largely a shifted threshold.
That holds for S2 and F-mean(A2) alike, and it is why the gate's M2 (threshold-free)
is the more trustworthy half here. It does not change the verdict: M2 fails by a wider
margin than M1. In `4ch_only` and `full_bank`, F-mean(A2)'s real FPR on E3/E4/E5
(58–81%) exceeds S2's (33–78%) in every one of the six cases, which is where its M1 "gain" comes from. All four runs:
`real/analysis/<run id>/metrics.csv`.

This is the FIRE calibration failure again, now measured: a threshold fitted on one
real distribution does not transfer to web, social-media and in-the-wild reals. It is
the problem the P3 reliability gate has to solve.

---

## 4. Per slice (P-A seed 0)

Source: `real/analysis/p_a_sd1_seed0/metrics.csv`. TPR@5%FPR / AUC, %. Gate: M1+M2 = in
both macros, M1 = M1 only (no reals), — = reported, outside the gate. Fakes / reals per
slice; a slice's reals are its view's whole real class (I-061).

| slice | gate | fakes / reals | S1 | S2 | S3-LR | S3-Set | AEROBLADE | F-mean(A2) | F-stack |
|---|---|---|---|---|---|---|---|---|---|
| E7 sdxl-vae | M1+M2 | 491 / 486 | 50.9 / 87.8 | **99.6 / 99.8** | 8.1 / 48.1 | 59.3 / 85.7 | 8.1 / 56.0 | 84.7 / 98.0 | 10.2 / 52.2 |
| E7 SD3.5-large | M1+M2 | 487 / 486 | 35.7 / 82.2 | **80.1 / 96.9** | 5.3 / 46.3 | 41.7 / 84.2 | 12.3 / 59.8 | 63.2 / 96.0 | 6.0 / 49.7 |
| E7 FLUX.1 **luma** (§8) | M1+M2 | 492 / 486 | 40.2 / 65.7 | 34.1 / 80.3 | 3.3 / 32.4 | **53.5** / 76.5 | 32.1 / 63.5 | 52.6 / **83.5** | 3.7 / 33.5 |
| E3 GPT-Image2 | M1+M2 | 500 / 477 | 51.4 / 43.5 | 67.0 / 49.2 | 42.6 / 55.6 | 74.0 / 59.7 | **83.4 / 62.3** | 80.6 / 49.7 | 44.4 / 55.5 |
| E3 Nano-Banana2 | M1+M2 | 500 / 477 | 63.0 / 51.2 | **92.2 / 72.3** | 54.0 / 60.5 | 57.6 / 53.6 | 78.4 / 59.9 | 90.0 / 55.7 | 55.8 / 61.1 |
| E4 WildRF matched | M1+M2 | 749 / 743 | 93.3 / 78.4 | **99.9 / 95.5** | 59.4 / 71.9 | 51.9 / 52.2 | 48.7 / 47.3 | 95.2 / 75.1 | 61.9 / 73.1 |
| E5 Chameleon pipeline | M1+M2 | 1,884 / 1,864 | 74.0 / 56.1 | 91.3 / **68.5** | 36.5 / 64.3 | 78.2 / 65.2 | 77.7 / **68.5** | 91.2 / 63.2 | 43.4 / 64.8 |
| E8 Flux | M1 | 250 / 0 | 72.4 | **98.0** | 44.8 | 64.8 | 84.4 | 95.2 | 52.0 |
| E8 GoT | M1 | 250 / 0 | 93.6 | **100.0** | 39.6 | 92.4 | 82.8 | 99.6 | 46.8 |
| E8 Infinity | M1 | 250 / 0 | 94.8 | **100.0** | 41.2 | 55.2 | 67.6 | 99.2 | 50.0 |
| E8 NOVA | M1 | 250 / 0 | 97.6 | **100.0** | 25.2 | 86.4 | 51.2 | **100.0** | 27.6 |
| E8 OmniGen | M1 | 250 / 0 | 94.4 | **100.0** | 23.6 | 94.0 | 70.0 | **100.0** | 28.8 |
| E7 FLUX.1 standard | — | 492 / 486 | 63.4 / 91.8 | 58.9 / 93.5 | 2.0 / 30.4 | 40.7 / 84.0 | 31.9 / 76.3 | 71.5 / 96.0 | 2.0 / 32.4 |
| E7 sd-vae-ft-ema (in-bank) | — | 489 / 486 | 72.4 / 94.1 | 100.0 / 100.0 | 94.3 / 98.6 | 43.4 / 79.3 | 6.5 / 48.8 | 88.5 / 98.3 | 97.1 / 99.0 |
| E7 sd-vae-ft-mse (in-bank) | — | 481 / 486 | 64.2 / 91.8 | 100.0 / 100.0 | 94.6 / 98.8 | 47.2 / 80.1 | 6.4 / 51.3 | 81.9 / 97.9 | 97.1 / 99.1 |
| E7 SD2.1 (in-bank) | — | 482 / 486 | 65.4 / 92.2 | 100.0 / 100.0 | 94.0 / 98.6 | 48.1 / 81.9 | 7.5 / 53.8 | 86.1 / 98.1 | 96.5 / 98.9 |
| T-ctrl diagnostic | — | 200 / 200 | 84.5 / 74.4 | 94.0 / 92.2 | 28.5 / 55.8 | 81.0 / 91.8 | 42.5 / 66.4 | 95.5 / 89.2 | 30.0 / 57.7 |

**Trivial-rule baselines, printed beside their numbers (D-044):** E7 FLUX.1 —
`4:2:0 → fake` 100.00% (lift +49.85), quality +14.35, so the **FLUX.1 pair is S2 93.5 /
80.3 AUC and F-mean(A2) 96.0 / 83.5 (standard / luma)**, and only the luma number is
claimable. E4 matched: quality separability +11.39 (~61% reachable with no generative
artefact). E3 Nano-Banana2: residual +6.47 after matching.

**Caveats that travel with these rows.** E4 holds 24 byte-identical groups (48 of 1,492
rows), used as distributed. Counted once each, S2 is 99.9 / 95.5 and F-mean(A2) 95.1 /
75.0, unchanged in substance (`metrics_deduplicated.csv`). E3's real class lost 23 of
500 images to the I-047 crop rule and no fakes (Amendment 1). E3 GPT-Image2 sits at
chance for every head (AUC 43–62).

Seed 1 tracks seed 0: S2's AUC changes by at most 3.2 points on any gate slice (E3 GPT-Image2, 49.2 → 46.0). Per-slice
tables for all four runs are in `real/analysis/<run id>/metrics.csv`.

**In-bank sanity check.** On E7's in-bank VAEs S2 is at 100.0 / 100.0 and S3-LR at
94–95% TPR, so the heads work in distribution. S3-LR collapses only off it, as the
validation table predicted (§6).

---

## 5. Complementarity (RQ1, P-A seed 0)

Source: `real/analysis/p_a_sd1_seed0/complementarity.json`. Errors are taken at the
5%-FPR validation thresholds, so on views with reals they include the false positives of
§3. The pooled row is every row of the 12 gate slices.

| slice | pair | double fault | Jaccard of misses | Yule's Q | McNemar (test, p; A-only / B-only right) |
|---|---|---:|---:|---:|---|
| pooled gate slices | S1 · S2 | 0.229 | 0.47 | 0.77 | χ², 2.9e-44 (983 / 1,708) |
| | S1 · S3-Set | 0.176 | 0.30 | 0.28 | χ², 1.6e-05 (1,984 / 2,266) |
| | S2 · S3-Set | 0.165 | 0.31 | 0.42 | χ², 5.2e-13 (2,096 / 1,653) |
| E7 sdxl-vae | S2 · S3-Set | 0.003 | 0.01 | −0.43 | χ², 8.2e-37 (225 / 24) |
| E7 SD3.5 | S2 · S3-Set | 0.057 | 0.15 | 0.31 | χ², 8.6e-26 (257 / 67) |
| E7 FLUX.1 luma | S2 · S3-Set | 0.154 | 0.28 | 0.27 | χ², 0.086 (178 / 213) |
| E5 Chameleon | S2 · S3-Set | 0.256 | 0.45 | 0.65 | χ², 5.2e-23 (425 / 767) |
| E4 WildRF | S2 · S3-Set | 0.210 | 0.30 | 0.07 | χ², 8.7e-05 (411 / 305) |

**F-oracle (an upper bound, never a method).** Over S1/S2/S3-Set, pooled oracle accuracy
is 0.883 against S2's 0.676: headroom +0.207, and +0.273 over all five heads. Per slice
the headroom is near 0 on E8 and E7 sdxl-vae (≤ 0.026). It is 0.18–0.25 on E3, E4, E5
and FLUX.1-luma.

**Reading.** The semantic heads (S1, S2) and the decoder-bank head (S3-Set) do fail on
different images: Jaccard of misses is 0.28–0.45 for S2 · S3-Set, and Yule's Q is low or
negative on several slices. The oracle headroom is real. But equal-weight fusion does
not capture it, because S1 and S3-Set are much weaker than S2 on E4/E5/E3. Where S3-Set
is right and S2 is wrong (FLUX.1-luma: 213 vs 178; E5: 767 vs 425), the gain is diluted
by S1. The oracle's headroom on views with reals is also inflated by §3: on those slices
a "right" stream is often the one that happened not to fire on a real.

---

## 6. Exploratory — not part of the gate

**(a) Lineage transfer: single-lineage reconstruction evidence flips sign on other
decoders.** P-A's S3-LR, trained on SD1 reconstructions only, is perfect on SD1
validation (AUC 1.000) and **below chance on every other decoder**. Validation AUC by
decoder (source: `validation/p2_heads_p_a_sd1_seed0.json`,
`per_group_val_auc_full_validation`):

| decoder | S1 | S2 | S3-LR | S3-Set | AEROBLADE | F-mean(A2) |
|---|---:|---:|---:|---:|---:|---:|
| sd15* / sd21* / sd_ft_ema* | 0.965 / 0.950 / 0.932 | 1.000 | **1.000** | 0.980 / 0.985 / 0.984 | 0.59 / 0.59 / 0.53 | 0.996 / 0.995 / 0.994 |
| sdxl / sdxl_fp16fix | 0.905 / 0.921 | 0.999 / 0.998 | **0.163 / 0.167** | 0.992 / 0.989 | 0.661 / 0.682 | 0.996 / 0.997 |
| dcae / movq_k22 / realesrgan_x4 | 0.921 / 0.957 / 0.999 | 0.997 / 0.999 / 1.000 | **0.436 / 0.424 / 0.683** | 0.993 / 0.999 / 0.994 | 0.503 / 0.601 / 0.933 | 0.998 / 1.000 / 1.000 |
| sd35 / flux | 0.830 / 0.818 | 0.970 / 0.945 | **0.407 / 0.222** | 0.923 / 0.900 | 0.717 / 0.955 | 0.967 / 0.952 |
| qwen_image / flux2_klein | 0.859 / 0.541 | 0.874 / 0.600 | **0.378 / 0.296** | 0.839 / 0.952 | 0.839 / 0.979 | 0.913 / 0.771 |

(* = in protocol.) The panel confirms it: S3-LR is at 46–48% AUC on E7 sdxl-vae and
SD3.5 and 30–32% on FLUX.1, against 98.6–98.8% on the in-bank VAEs (§4). The
permutation-invariant S3-Set, trained on the same features, stays at 0.84–0.999 on the
same decoders. **This supports the decoder-bank design (contribution C1):** a
reconstruction cue learned against one VAE is lineage-specific and can invert. It is a
limitation of single-VAE reconstruction detectors such as CO-SPY (CVPR 2025), which
condition on one autoencoder. This is a P2 observation on our data, not a test of CO-SPY
itself.

**(b) Reconstruction fidelity vs S2 transfer.** Across the 12 bank decoders, mean
reconstruction PSNR on the 491 validation reals (S3's `psnr_db`) against P-A S2's
validation AUC on that decoder's fakes: **Spearman ρ = −0.785 (p = 0.0025)** for seed 0
and −0.788 (p = 0.0023) for seed 1. Over the 9 unseen decoders only, ρ = −0.817
(p = 0.0072). Decoders that reconstruct more faithfully (FLUX.2-klein 30.6 dB, FLUX.1
29.0, Qwen 28.0) are the ones S2 transfers to worst. Twelve points, exploratory, and not
corrected for multiplicity. Source: `validation/p2_heads_<run id>.json`, `exploratory`.

---

## 7. S2's in-training validation AUC is not evidence of quality (I-059)

S2's in-training validation AUC saturated: best 0.99998 (P-A seed 0, step 4,500). The
same-lineage validation used for checkpoint selection cannot rank checkpoints, so
**"best validation AUC" did not select for transfer, and the in-training curve is not
evidence of S2's quality.** The panel is the only evidence, and on it S2 is the strongest
head.

**Proposed for P3, to be pre-registered, not implemented:** a non-saturating
model-selection signal, chosen before training. Candidates, in order of preference:

1. **held-out-lineage validation within the training protocol.** For multi-lineage
   protocols, select on a lineage the run does not train on, rotating. P-A has one
   lineage, so for P-A this degenerates to (2) or (3);
2. **validation log-loss** (or Brier score), which keeps moving after AUC saturates;
3. **TPR at 1% FPR on the validation reals**, the operating point a deployment uses.
   It is noisier with ~490 reals (§4 of the pre-registration), so it needs a larger
   real validation pool.

---

## 8. Not run, and what remains

| item | status | reason / what is needed |
|---|---|---|
| §7 E3 per content type (texture / structure / hybrid) | **DONE (2026-10-06)**, exploratory, post-verdict | Section X, X2b: `reports/exploratory_x/x2/x2b_per_slice.csv` |
| §7 DIV2K lossless diagnostic | **DONE (2026-10-06)** | §8a below. *Correction:* this row earlier said DIV2K was not in the panel. It is: the panel's `t_ctrl_diagnostic` view is 200 DIV2K crops and 200 reconstructions of DIV2K crops, already scored once by every run. No new scoring was needed |
| §7 `gen_matched` vs raw ablation (S1, S2 on P-A) | **DEFERRED to P3** (user decision 2026-10-06) | "budget permitting"; needs a GPU S2 training run |
| §9 ViT-L/16 sensitivity | P4 by design | declared outside the P2 gate |
| I-042 quota accounting | OPEN | measure on the next GPU session: meter, commit run with the editor closed, meter |

### 8a. DIV2K lossless diagnostic — the JPEG-reliance check (§7)

**Data.** `t_ctrl_diagnostic` holds 200 DIV2K crops — 2K photography stored as
lossless PNG, **never JPEG** — and 200 reconstructions of DIV2K crops by the bank's
decoders, also lossless PNG. It is withheld from every training list (D-051) and was
scored once in Session E.

**Method.** `scripts/p2_div2k_diagnostic.py` reads the committed scores, so nothing is
re-scored, and writes `reports/p2_div2k/div2k_diagnostic.{csv,tex,json}`.
- **Thresholds:** 5% FPR, fixed on the validation reals (COCO, JPEG history).
- **CIs:** 95% bootstrap, 1,000 resamples, the same draw for every score.
- **Gap:** paired, DIV2K-real FPR minus COCO-real FPR.

| run | score | FPR on DIV2K reals (never JPEG) | FPR on E7 COCO reals | gap, points | AUC DIV2K real vs reconstruction |
|---|---|---|---|---|---|
| P-A s0 | S1 | 0.535 [0.470, 0.610] | 0.062 [0.041, 0.084] | 47.3 [40.2, 54.9] | 0.744 [0.693, 0.792] |
| P-A s0 | S2 | 0.425 [0.360, 0.495] | 0.051 [0.033, 0.072] | 37.4 [30.3, 44.6] | 0.922 [0.894, 0.949] |
| P-A s0 | S3-Set | 0.120 [0.075, 0.165] | 0.058 [0.037, 0.078] | 6.2 [1.4, 11.1] | 0.918 [0.891, 0.943] |
| P-A s0 | F-mean(A2) | 0.440 [0.370, 0.505] | 0.041 [0.025, 0.058] | 39.9 [32.6, 47.2] | 0.892 [0.861, 0.922] |
| P-A s1 | S1 | 0.535 [0.470, 0.610] | 0.062 [0.041, 0.084] | 47.3 [40.2, 54.9] | 0.744 [0.693, 0.792] |
| P-A s1 | S2 | 0.310 [0.250, 0.375] | 0.045 [0.027, 0.064] | 26.5 [19.9, 33.2] | 0.919 [0.892, 0.946] |
| P-A s1 | S3-Set | 0.125 [0.080, 0.170] | 0.053 [0.035, 0.072] | 7.2 [2.6, 12.3] | 0.917 [0.891, 0.944] |
| P-A s1 | F-mean(A2) | 0.435 [0.365, 0.505] | 0.051 [0.033, 0.072] | 38.4 [31.0, 45.4] | 0.896 [0.863, 0.924] |
| 4ch_only | S1 | 0.525 [0.455, 0.595] | 0.045 [0.029, 0.066] | 48.0 [41.1, 54.9] | 0.750 [0.701, 0.798] |
| 4ch_only | S2 | 0.435 [0.365, 0.505] | 0.041 [0.025, 0.060] | 39.4 [32.1, 46.7] | 0.914 [0.887, 0.942] |
| 4ch_only | S3-Set | 0.095 [0.055, 0.140] | 0.039 [0.023, 0.056] | 5.6 [1.3, 10.1] | 0.939 [0.917, 0.959] |
| 4ch_only | F-mean(A2) | 0.470 [0.400, 0.535] | 0.033 [0.019, 0.049] | 43.7 [36.3, 50.8] | 0.926 [0.902, 0.949] |
| full_bank | S1 | 0.675 [0.610, 0.740] | 0.062 [0.041, 0.084] | 61.3 [54.3, 68.2] | 0.710 [0.659, 0.760] |
| full_bank | S2 | 0.535 [0.460, 0.610] | 0.041 [0.025, 0.060] | 49.4 [41.8, 56.4] | 0.962 [0.944, 0.977] |
| full_bank | S3-Set | 0.220 [0.165, 0.275] | 0.043 [0.027, 0.062] | 17.7 [11.9, 23.3] | 0.936 [0.910, 0.960] |
| full_bank | F-mean(A2) | 0.680 [0.615, 0.745] | 0.043 [0.027, 0.064] | 63.7 [56.9, 70.5] | 0.966 [0.951, 0.980] |

S3-LR, AEROBLADE and F-stack are in the CSV.

**Reading.**
- **The semantic streams call clean photographs fake.** At a threshold that flags ~5%
  of COCO reals, S1 flags 52–68% of never-compressed DIV2K photographs and S2 31–54%.
  The gap CIs exclude 0 everywhere.
- **The decoder-bank head is far less affected**: S3-Set flags 9.5–22%.
- **The ranking survives.** With no JPEG trace on either side, S2 (0.91–0.96 AUC) and
  S3-Set (0.92–0.94) still separate DIV2K reals from DIV2K reconstructions. So the
  failure is the operating point, not the ranking — the same calibration problem as
  I-064.
- **Caveat.** DIV2K differs from COCO in content and resolution as well as
  compression. A high DIV2K FPR is consistent with JPEG reliance but does not isolate
  it. X1's format control (re-encoding with identical JPEG settings) is the direct
  test.

---

## 9. Provenance

| artefact | sha256 / commit |
|---|---|
| pre-registration (with Amendment 2), as used by the analysis | `go_no_go.json → preregistration_sha256` |
| panel list / luma list digests | `bff608c3…c51116` / `190279cc…ec9c2303` (= Amendment 1), checked by `p2_score.py` |
| scores, per run | `real/scores/scores_<run id>_summary.json → scores_sha256`, `inputs_sha256` |
| heads reports | `validation/p2_heads_<run id>.json` (copied verbatim from the fit output) |
| code | `5e76fed` (`SESSION_E_REAL_STARTED.json → commit`) |

---

## X1 — EXPLORATORY: is frozen DINOv3 blind to VAE-only images?

**Exploratory, outside the gate: no P2 number and no part of the verdict changes.**
X1 asks the *Simplicity Prevails* question (arXiv 2602.01738, Table 9) in our pipeline,
at ViT-B/16. This section does not quote the paper's numbers; compare against its
Table 9 directly.

**Design** (`scripts/x1_gen_only_probe.py`, CPU; results `reports/p2_x1/x1_results.{json,csv}`):

- **Features and head.** Frozen DINOv3-B/16 features from the existing S1 caches (none
  were missing), and a logistic regression with S1's settings (`fit_probe`, same C grid).
- **Training.** Train-split real COCO vs train-split SD1.5 full generations
  (`gen_matched`) only: 3,101 images, **no reconstruction**.
- **Choosing C.** C (0.1) was chosen on a seeded 20% hold-out of the train split
  (813 images), drawn by COCO group, so the val split was used only for testing.
- **No leakage.** No image id and no COCO group is in both training and test
  (`tests/test_x1_split.py`).
- **Tests.** (a) the 500 val `gen_matched` vs the 491 val reals. (b) Each val real vs
  its own reconstruction by each of the 12 bank decoders (491 pairs per decoder).
  (c) E7 DDA-COCO subsets on the panel against E7's 486 reals.
- **Metrics.** Threshold at 5% FPR fixed on the 491 val reals (4.89% there, by
  construction). 95% bootstrap CIs, 1,000 resamples, seed 2026. In (b) the bootstrap
  resamples real/reconstruction pairs; elsewhere reals and fakes are resampled
  independently.
- **S2 side by side** on the same images, each with its own threshold fixed on the same
  val reals.

**(a, b) Val split. AUC [95% CI] | detection rate at 5% FPR (%)**

| test (fakes vs val reals) | gen-only probe | | S2 P-A seed 0 | | S2 full_bank seed 0 | |
|---|---|---:|---|---:|---|---:|
| **gen_matched (SD1.5 full generations)** | **0.999** [0.999, 1.000] | 99.8 | 1.000 [1.000, 1.000] | 100.0 | 0.999 [0.999, 1.000] | 99.8 |
| sd15 reconstructions | 0.893 [0.876, 0.909] | 45.8 | 1.000 [1.000, 1.000] | 100.0 | 1.000 [1.000, 1.000] | 100.0 |
| sd21 | 0.813 [0.794, 0.831] | 33.6 | 1.000 [1.000, 1.000] | 100.0 | 0.999 [0.999, 1.000] | 100.0 |
| sd_ft_ema | 0.819 [0.800, 0.838] | 36.9 | 1.000 [1.000, 1.000] | 100.0 | 1.000 [0.999, 1.000] | 100.0 |
| sdxl | 0.796 [0.778, 0.816] | 26.3 | 0.998 [0.997, 0.999] | 99.6 | 0.999 [0.999, 1.000] | 99.8 |
| sdxl_fp16fix | 0.796 [0.778, 0.817] | 26.1 | 0.997 [0.995, 0.999] | 99.2 | 0.999 [0.998, 1.000] | 99.8 |
| sd35 | 0.721 [0.704, 0.742] | 16.5 | 0.974 [0.966, 0.981] | 81.3 | 0.999 [0.998, 1.000] | 99.8 |
| flux | 0.708 [0.692, 0.727] | 15.7 | 0.929 [0.916, 0.942] | 56.6 | 0.995 [0.992, 0.997] | 97.8 |
| dcae | 0.783 [0.766, 0.803] | 23.6 | 0.997 [0.996, 0.999] | 99.0 | 0.998 [0.997, 0.999] | 99.2 |
| realesrgan_x4 | 0.996 [0.993, 0.998] | 97.4 | 1.000 [1.000, 1.000] | 100.0 | 1.000 [1.000, 1.000] | 100.0 |
| flux2_klein | **0.546** [0.539, 0.554] | **5.9** | 0.598 [0.586, 0.610] | 7.1 | 0.982 [0.976, 0.988] | 90.2 |
| qwen_image | 0.854 [0.836, 0.871] | 32.2 | 0.855 [0.837, 0.874] | 27.1 | 0.999 [0.998, 1.000] | 99.4 |
| movq_k22 | 0.887 [0.871, 0.905] | 42.2 | 0.999 [0.999, 1.000] | 100.0 | 1.000 [0.999, 1.000] | 99.8 |

**(c) E7 DDA-COCO on the panel** (existing images and scores; fakes / reals; in-bank
status is P-A's):

| E7 subset | n | P-A | gen-only probe | | S2 P-A seed 0 | | S2 full_bank seed 0 | |
|---|---|---|---|---:|---|---:|---|---:|
| FLUX.1 † | 492 / 486 | unseen | 0.881 [0.859, 0.902] | 47.0 | 0.935 [0.919, 0.949] | 58.9 | 0.988 [0.982, 0.993] | 94.3 |
| sd-vae-ft-ema | 489 / 486 | in-bank | 0.809 [0.780, 0.835] | 38.4 | 1.000 [1.000, 1.000] | 100.0 | 0.999 [0.998, 1.000] | 99.4 |
| sd-vae-ft-mse | 481 / 486 | in-bank | 0.772 [0.743, 0.801] | 30.6 | 1.000 [1.000, 1.000] | 100.0 | 0.998 [0.997, 0.999] | 99.2 |
| stable-diffusion-2-1 | 482 / 486 | in-bank | 0.784 [0.755, 0.813] | 31.1 | 1.000 [1.000, 1.000] | 100.0 | 0.998 [0.997, 0.999] | 99.6 |
| sdxl-vae | 491 / 486 | unseen | 0.732 [0.701, 0.763] | 22.0 | 0.998 [0.997, 0.999] | 99.6 | 0.997 [0.995, 0.999] | 98.8 |
| stable-diffusion-3.5-large | 487 / 486 | unseen | 0.689 [0.658, 0.722] | 18.1 | 0.969 [0.960, 0.977] | 80.1 | 0.992 [0.989, 0.995] | 96.1 |

† Standard (not luma) FLUX.1: a rule reading only chroma subsampling scores 100.00%
here (§6d, D-044), so every FLUX.1 number in this table is an upper bound. X1 did not
score the luma variant.

**Figure.** `x1_auc_by_decoder.png` / `.svg` shows AUC per decoder for the probe vs both
S2 runs, with 95% CIs and a chance line at 0.5. It is committed under
`reports/p2_x1/figures/` (CLAUDE.md §2b, 2026-10-06). The table above is the same data
(`x1_results.csv`).

**S1 for reference — extracted, not recomputed, and NOT the clean test.** S1 trained on
reconstructions (and `gen_matched`), so its numbers do not answer X1's question.

- **Validation AUC per decoder.** From
  `validation/p2_heads_<run id>.json → per_group_val_auc_full_validation`: 491 reals vs
  100–130 fakes per decoder, a different and smaller fake set than X1's.
  - P-A seed 0: sd15 0.965, sd21 0.950, sd_ft_ema 0.932, sdxl 0.905,
    sdxl_fp16fix 0.921, sd35 0.830, flux 0.818, dcae 0.921, realesrgan_x4 0.999,
    flux2_klein 0.541, qwen_image 0.859, movq_k22 0.957.
  - full_bank: flux2_klein 0.667, qwen_image 0.937; the other ten are within 0.04 of
    P-A's.
- **E7 on the panel, AUC / TPR@5%** (`real/analysis/<run id>/metrics.csv`).
  - P-A seed 0: FLUX.1 91.8 / 63.4, sd-vae-ft-ema 94.1 / 72.4, sd-vae-ft-mse
    91.8 / 64.2, sd2-1 92.2 / 65.4, sdxl-vae 87.8 / 50.9, SD3.5 82.2 / 35.7.
  - full_bank: 80.2–91.1 AUC.

**Reading (exploratory).**

1. **Frozen DINOv3-B is not blind to VAE-only images in our pipeline, but it is
   weak on them.**
   - The probe trained on full generations separates them almost perfectly
     (AUC 0.999).
   - On reconstructions of the same real images, its AUC falls to 0.71–0.89 for ten of
     the twelve decoders, and it catches only **16–46%** of them at the 5% FPR
     threshold.
   - The image-level signal it learned is mostly something full generation adds and a
     VAE round-trip does not.
2. **The two extremes.**
   - **FLUX.2-klein is nearly invisible** to it: 0.546, 5.9%. This is also the
     highest-fidelity decoder (30.6 dB mean PSNR on the val reals, P2 §6b).
   - **Real-ESRGAN ×4** is easy (0.996, 97.4%), but it is a super-resolution upscaler,
     not a VAE, so it says nothing about VAE blindness.
3. **Fine-tuning closes most of the gap; seeing the decoder closes the rest.**
   - S2 detects most reconstructions, including decoders outside its training
     protocol (P-A: SDXL 0.998, DC-AE 0.997).
   - FLUX.2-klein stays near chance for P-A S2 (0.598), and Qwen-Image stays at 0.855.
   - Only `full_bank`, which trained on those decoders' reconstructions, reaches
     0.98–0.999.
   - This agrees with P2 §6(b): reconstruction fidelity predicts transfer.

**Caveats.**

- **S2's val numbers are optimistic.** S2's checkpoint was selected on its own
  validation AUC (I-059), so its numbers on these val images carry that selection. The
  probe's carry none.
- **Format.** Reals and reconstructions are PNG, and `gen_matched` is JPEG,
  format-matched to COCO. Test (a) could carry a residual format cue; tests (b) and (c)
  do not.
- **Threshold.** It is fixed on the same 491 reals that are the negatives in (a) and
  (b), as specified, so the FPR there is 4.89% by construction.

**X1 format control (PENDING-KAGGLE).** Every X1 image — train and test, real and
fake — will be re-encoded as JPEG q95, 4:4:4, and the probe re-fitted on the same
split, to test whether the probe reads file format.
- The images exist only on Kaggle (§6a(a)), so this runs as a Kaggle session:
  `env/kaggle_run_x1_format.py`, dry-run PASSED.
- The result will go here, side by side with the table above, via
  `scripts/x1_format_control.py analyse`.

---

## X — EXPLORATORY (post-verdict) analyses X1–X3

**Exploratory, after the verdict: they do not change it.** No P2 model was retrained,
and no P2 output was rewritten. CPU only.
`scripts/p2_exploratory_x.py`; every table is in
`reports/exploratory_x/<x1|x2|x3a|x3b|x3c>/` as CSV plus a LaTeX booktabs `.tex`.

**Rules for every table:**
- **Same images.** Within a run, every row uses the same images and slices: one scores
  file, rows joined by id, Session E's slice definitions.
- **AUC** is the primary metric, beside AP.
- **TPR@5%FPR "slice-calibrated":** the threshold is set on that slice's own reals,
  because of I-064. The real-image FPR at the validation threshold is a separate column
  in the CSVs.
- **CIs:** 95% bootstrap, 1,000 resamples, seed 2026, using the Session E scheme. The
  slice threshold is re-fitted in every replicate. Every gain is paired and compared
  with the best single module (S1, S2, S3-Set), re-maximised per replicate.
- **Macro averages** run over each run's unseen panel slices *with reals*: P-A 7,
  4ch_only 6, full_bank 4. E8 has no reals, so it can't enter a slice-calibrated macro.

**Sanity checks:**
- F-mean(A2) recomputed from the S1/S2/S3-Set columns matches the scored column
  (max gap ≤ 1e-4).
- The validation thresholds of S1/S2/S3-Set are reproduced from the frozen heads before
  the pairs' validation thresholds are derived.
- X3b at k = 12 reproduces the P2 full_bank heads exactly: S3-LR 0.9727, S3-Set 0.9844
  validation AUC.
- Point estimates are identical between a 20-resample smoke run and the 1,000-resample
  run.

**Figures** (committed under `reports/exploratory_x/figures/`, CLAUDE.md §2b):
- `x3a_min_lpips_by_decoder.{png,svg}`: AUC per decoder for 1 decoder vs the SD1
  family vs all VAE members.
- `x3b_auc_vs_k.{png,svg}`: held-out-decoder validation AUC and panel macro AUC against
  k.

### X2 — module ablation

**X2a. Macro over unseen slices** (`x2/x2a_macro.csv`). Gain in points vs the best single
module, paired 95% CI.

| run | S2 AUC | S2 TPR@5% (slice-cal) | best pair AUC (gain) | F-mean(A2) AUC (gain) | F-stack (A) AUC (gain) | F-oracle AUC — **upper bound, not a method** |
|---|---|---|---|---|---|---|
| P-A s0 | 0.804 [0.793, 0.813] | 0.476 [0.454, 0.496] | S2+S3-Set 0.735 (−6.89 [−8.02, −5.81]) | 0.745 (−5.89 [−6.92, −4.85]) | 0.557 (−24.65) | 0.998 |
| P-A s1 | 0.801 [0.791, 0.812] | 0.472 [0.445, 0.495] | S2+S3-Set 0.745 (−5.64 [−6.84, −4.50]) | 0.751 (−5.03 [−6.11, −3.99]) | 0.550 (−25.16) | 0.997 |
| 4ch_only | 0.770 [0.757, 0.782] | 0.390 [0.355, 0.415] | S2+S3-Set 0.709 (−6.14 [−7.30, −4.97]) | 0.711 (−5.88 [−6.97, −4.83]) | 0.724 (−4.55 [−5.62, −3.52]) | 0.994 |
| full_bank | 0.774 [0.760, 0.788] | 0.361 [0.312, 0.385] | S2+S3-Set 0.738 (−3.64 [−5.19, −2.05]) | 0.731 (−4.26 [−5.88, −2.73]) | 0.745 (−2.85 [−4.48, −1.18]) | 0.986 |

S1 is at 0.615–0.664 and S3-Set at 0.604–0.688 in every run. **No pair and no fusion
beats S2 in any run; every gain CI lies below 0.** The same holds for TPR (e.g. P-A s0:
F-mean(A2) −13.48 [−17.85, −11.07]).

The F-oracle row is near 1.0 everywhere because it lets the label pick the most
favourable module per image. It is too loose to be informative and is shown only
because it was requested. X2c gives the meaningful headroom.

**X2c. Complementarity at slice-calibrated 5% thresholds**, the 2×2 tables summed over
the unseen slices (`x2/x2c_complementarity.csv`, per-slice rows included):

| run | pair | double fault | Jaccard of misses | McNemar p | only A right / only B right | oracle accuracy headroom |
|---|---|---|---|---|---|---|
| P-A s0 | S1 / S2 | 0.277 | 0.579 | 5.9e-165 | 401 / 1,638 | 0.040 |
| P-A s0 | S1 / S3-Set | 0.370 | 0.699 | 2.8e-08 | 919 / 695 | 0.069 |
| P-A s0 | S2 / S3-Set | 0.274 | 0.543 | 7.1e-201 | 1,896 / 435 | 0.043 |
| full_bank | S1 / S2 | 0.318 | 0.615 | 7.7e-55 | 420 / 1,011 | 0.058 |
| full_bank | S1 / S3-Set | 0.421 | 0.768 | 2.5e-34 | 641 / 271 | 0.038 |
| full_bank | S2 / S3-Set | 0.339 | 0.619 | 1.5e-135 | 1,231 / 270 | 0.038 |

P-A s1 and 4ch_only are in the same range (Jaccard 0.555–0.736, headroom 0.037–0.072).

**X2d. Who wins where** (`x2/x2d_who_wins.csv`; best single module by AUC, margin over
the better of the other two, paired CI):

- **S2** wins nearly every slice. Under P-A s0 it takes E4 by +17.15 [14.97, 19.48]
  and every E7 subset by 1.7–12.7 points; FLUX.1's margin of 1.70 [−0.06, 3.48] includes
  0. Under full_bank its E7 margins reach 15.2.
- **S3-Set wins on E3 GPT-Image2 under P-A**: 0.597 vs S2 0.492, +10.58 [5.91, 15.42].
  AEROBLADE is higher still there.
- **S3-Set also wins on E5 Chameleon** under 4ch_only (+4.52 [2.48, 6.56]) and full_bank
  (+0.31 [−2.14, 2.56], a tie).
- Equal-weight pairs or F-stack are the best non-oracle score on a few slices
  (FLUX.1-luma, T-ctrl). Their margins are within the CIs.

**X2b. Per-slice AUC**, all runs (`x2/x2b_per_slice_auc.{csv,tex}`). Under P-A s0, E3
by content type:

| E3 content type | S1 | S2 | S3-Set | F-mean(A2) |
|---|---|---|---|---|
| Hybrid | 0.442 | 0.479 | 0.522 | 0.442 |
| Structure | 0.424 | 0.476 | 0.423 | 0.396 |
| Texture | 0.575 | 0.840 | 0.786 | 0.767 |

**The winner depends on E3's content type** (P-A s0, `x2/x2b_per_slice.csv`):
- **Structure images:** S1, S2 and S3-Set are all below chance (0.42–0.48), while
  S3-LR (0.749), AEROBLADE (0.716) and F-stack (0.745) detect them.
- **Hybrid images:** everything fails or inverts (0.34–0.52).
- **Texture images:** S2 leads (0.840).

This is the §7 content-type report, produced here as exploratory.

### X1 — generation-only DINOv3 probe, in this format (`x1/x1_auc.{csv,tex}`)

| fakes vs reals | gen-only probe | S2 P-A s0 | S2 P-A s1 | S2 4ch_only | S2 full_bank |
|---|---|---|---|---|---|
| SD1.5 full generations | 0.999 | 1.000 | 1.000 | 1.000 | 0.999 |
| reconstructions, 10 decoders* | 0.708–0.893 | 0.855–1.000 | 0.832–1.000 | 0.875–1.000 | 0.995–1.000 |
| flux2_klein reconstructions | 0.546 | 0.598 | 0.611 | 0.595 | 0.982 |
| realesrgan_x4 (upscaler) | 0.996 | 1.000 | 1.000 | 1.000 | 1.000 |
| E7 subsets (panel) | 0.689–0.881 | 0.969–1.000 | 0.968–1.000 | 0.969–1.000 | 0.988–0.999 |

\* All bank decoders except flux2_klein and realesrgan_x4. Full rows, AP and
slice-calibrated TPR with CIs are in `x1/x1_results.csv`. The E7 FLUX.1 row is an upper
bound (chroma shortcut).

### X3 — decoder count

**X3a. Training-free min-LPIPS** (`x3a/x3a_auc.{csv,tex}`). * = the decoder is one of
the variant's members.

| test | sd15 only | SD1 family | all 11 VAE members |
|---|---|---|---|
| val reals vs sd15 / sd21 / sd_ft_ema reconstructions | 0.993* / 0.871 / 0.834 | 0.981* / 0.992* / 0.989* | 0.588* / 0.594* / 0.534* |
| … vs sdxl / sd35 / dcae / movq_k22 | 0.758 / 0.588 / 0.686 / 0.693 | 0.768 / 0.588 / 0.667 / 0.680 | 0.661* / 0.717* / 0.503* / 0.601* |
| … vs flux / flux2_klein / qwen_image | 0.576 / 0.594 / 0.615 | 0.578 / 0.598 / 0.616 | **0.955* / 0.979* / 0.839*** |
| E7 sd-vae-ft-ema / sd-vae-ft-mse / sd2-1 (panel) | 0.756 / 0.776 / 0.787 | 0.863 / 0.886 / 0.895 | 0.488 / 0.513 / 0.538 |
| E7 sdxl-vae / SD3.5 / FLUX.1-luma | 0.696 / 0.563 / 0.516 | 0.710 / 0.571 / 0.524 | 0.560 / 0.598 / 0.635 |
| E3 GPT-Image2 / Nano-Banana2 / E4 / E5 | 0.562 / 0.596 / 0.675 / 0.664 | 0.551 / 0.588 / 0.661 / 0.666 | 0.623 / 0.599 / 0.473 / 0.685 |

**X3b. Learned heads vs the number of decoders k**, training images fixed to
full_bank's (`x3b/x3b_decoder_count.{csv,tex}`). "Near-first" order shown; "far-first"
is in the CSV.

| k | S3-LR val AUC (all) | S3-LR val, held-out decoders | S3-LR panel, full_bank unseen | S3-LR panel, P-A gate slices | S3-Set panel, full_bank unseen | S3-Set panel, P-A gate slices |
|---|---|---|---|---|---|---|
| 1 (sd15) | 0.810 | 0.775 | 0.648 [0.630, 0.666] | 0.640 [0.625, 0.655] | 0.650 [0.631, 0.667] | 0.646 [0.631, 0.661] |
| 2 | 0.851 | 0.815 | 0.675 [0.657, 0.691] | 0.670 [0.656, 0.685] | 0.643 [0.625, 0.660] | 0.654 [0.640, 0.668] |
| 4 | 0.900 | 0.858 | 0.654 [0.635, 0.671] | 0.695 [0.682, 0.709] | 0.631 [0.613, 0.650] | 0.696 [0.684, 0.710] |
| 6 | 0.941 | 0.924 | 0.626 [0.608, 0.644] | 0.686 [0.674, 0.700] | 0.626 [0.608, 0.645] | 0.692 [0.679, 0.705] |
| 8 (all lineages) | 0.950 | 0.934 | 0.628 [0.610, 0.646] | 0.692 [0.680, 0.706] | 0.625 [0.607, 0.644] | 0.685 [0.672, 0.699] |
| 12 | 0.973 | n/a | 0.638 [0.619, 0.656] | 0.720 [0.707, 0.733] | 0.604 [0.586, 0.623] | 0.705 [0.691, 0.718] |

- **Held-out decoders** means decoders whose *features* the head does not use. Their
  reconstructions are still among the full_bank training fakes.
- **Order matters at small k.** S3-Set's held-out-decoder AUC at k = 2 is 0.805
  near-first vs 0.734 far-first (adding Real-ESRGAN).

**X3c. S2 of each run on the same panel slices** (`x3c/x3c_s2_auc.{csv,tex}`, AUC; * =
the run trained on that E7 VAE's lineage):

| slice | P-A s0 | P-A s1 | 4ch_only | full_bank |
|---|---|---|---|---|
| E3 GPT-Image2 | 0.492 | 0.460 | 0.522 | **0.663** |
| E3 Nano-Banana2 | 0.723 | 0.694 | 0.722 | **0.825** |
| E4 WildRF | **0.955** | **0.955** | 0.950 | 0.927 |
| E5 Chameleon | 0.685 | 0.714 | 0.670 | 0.681 |
| E7 SD3.5 | 0.969 | 0.968 | 0.969 | 0.992* |
| E7 FLUX.1-luma | 0.803 | 0.819 | 0.787 | 0.949* |

full_bank saw every bank lineage, so its E7 numbers are not an unseen test. The fair
unseen test of training breadth is P4 leave-one-lineage-out. E3's generators (GPT-Image2,
Nano-Banana2) are outside every bank, so its column *is* unseen for all four runs.

**Not run — GPU estimate for S2 trained on sd15 only (k = 1, the first P4 bank-size
point):**
- **Training:** Session D measured 1.15–1.38 GPU-h per S2 run on one T4, early-stopped
  at 4,500–5,500 steps of an 8,000 ceiling, about 1.07 s per step
  (`reports/kaggle_20261005/p2_s2_<run id>.json`, `gpu_hours`, `measured_step_seconds`).
  Without early stopping the ceiling is about 2.4 h.
- **Scoring:** about 8 min.
- **Overall:** about **1.5 GPU-h expected, 2.6 h worst case**. The quota charged depends
  on I-042, which is still open.
- **Prerequisites:** a new `sd15_only` protocol in `configs/p2_runs.yaml` and a Session
  D-style runbook. Both are additions that need approval.

### Plain-language summary (exploratory; includes results that do not support our hypothesis)

1. **S2 alone is the best detector.** Its unseen-slice AUC is 0.77–0.80. No equal-weight
   pair, F-mean(A2) or F-stack beats it in any of the four runs: every gain is negative
   with a CI below 0. *Against the fusion hypothesis.*
2. **At a real 5% false-alarm rate, set on each slice's own reals, S2 catches only
   36–48% of unseen fakes.** The ~88% M1 figure was mostly threshold shift (I-064).
3. **The modules' errors overlap more than P2 §5 suggested.** At slice-calibrated
   thresholds, the Jaccard of misses is 0.54–0.77 and a perfect per-image chooser gains
   only 4–7 accuracy points. The earlier larger headroom was partly the threshold
   artefact. *Weakens the complementarity argument.*
4. **The decoder-bank head (S3-Set) does beat S2 on some unseen generators:** E3
   GPT-Image2 (+10.6 points under P-A) and E5 Chameleon (4ch_only). So
   generator-side evidence is not redundant everywhere, but no fixed fusion exploits it.
5. **On E3 the winner depends on content.** On structure images S1, S2 and S3-Set
   are below chance; only S3-LR and AEROBLADE see them (0.75, 0.72). Texture images are S2's (0.84). Hybrid images defeat everything
   (0.34–0.52). This is the clearest sign of complementarity in the panel, and it is by
   content, not by generator.
6. **Frozen DINOv3 only partly sees VAE-only fakes** (X1: AUC 0.55–0.89). Fine-tuning
   (S2) closes most of that gap. FLUX.2-klein is missed by everything except the S2 that
   trained on it.
7. **Training-free, more decoders is not better.** min-LPIPS with sd15 alone detects
   sd15 reconstructions (0.99), and the SD1 family detects SD1 reconstructions
   (0.98–0.99). The minimum over all 11 VAEs drops them to 0.53–0.59
   and E4 to 0.47. The all-member minimum helps only on high-fidelity decoders (FLUX
   0.96, FLUX.2 0.98). A naive bank is worse than one well-chosen decoder. *Against
   "more decoders" in its simplest form.*
8. **With a learned head, more decoders help on reconstructions:** validation AUC
   0.81 → 0.97, held-out decoders 0.77 → 0.93, P-A gate slices 0.64 → 0.72. *Supports
   C1.*
9. **But more decoders do not help on unseen whole generators** (E3/E4/E5). S3-LR stays
   at 0.63–0.68 and S3-Set falls from 0.65 to 0.60. *Against "more decoders generalise
   to unseen generators".*
10. **Training on more lineages makes S2 better on unseen E3 generators** (GPT-Image2
    0.49 → 0.66, Nano-Banana2 0.72 → 0.83), slightly worse on E4 (0.955 → 0.927) and
    equal on E5. Breadth of training data, not of probe features, is where the gain
    appears. P4's leave-one-lineage-out runs are the fair test.
