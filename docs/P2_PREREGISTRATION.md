# P2 pre-registration — the analysis plan, fixed before any test score exists

**Project:** Decoder-Bank Probing (DBP) · 24i-8017, MS Data Science, FAST-NUCES
**Status:** committed before B7. The commit hash of *this file* is recorded in the P2
report, and `scripts/p2_analyze.py` refuses to run if this file is absent.

This document exists so that the go / no-go verdict cannot be chosen after the numbers
are seen. Everything below — the slices, the metrics, the baseline, the criterion — is
decided here, while the test panel is unscored. `p2_analyze.py` implements this
document and has no flag that changes any of it.

---

## 1. What is being decided

**RQ1 (complementarity).** Do generator-side evidence and semantic evidence fail on
*different* images, on unseen generator families, content types and degradations? And
if they do, is the overlap small enough to exploit?

The gate decides whether P3 builds the reliability gate on top of a fusion, or whether
the thesis pivots to a deployment-oriented study of the best single stream plus
calibration. **Both outcomes are publishable and both are reported unchanged.** A
NO-GO is a finding about the streams, not a failure of the work.

## 2. Protocols

| role | run id | training fakes |
|---|---|---|
| **PRIMARY — the gate is decided here** | `p_a_sd1` | the SD1 lineage only: `sd15`, `sd21`, `sd_ft_ema`, plus SD1.5 `gen_matched` |
| secondary, reported not gated | `4ch_only` | all 4-channel lineages (D-034; the only SEF-comparable configuration) |
| secondary, reported not gated | `full_bank` | all 12 distinct members |

All three train on identical data within a protocol and differ only in which lineages
the fakes may come from, so a difference between the three streams is attributable to
the representation (§6.1 of the P2 plan).

## 3. Unseen slices under P-A

Resolved to counts here, so the macro-average cannot be silently re-weighted later.
Counts are panel sizes after the approved caps, not population sizes.

| slice | panel n | has reals? | enters M1 | enters M2 |
|---|---|---|---|---|
| E7 `sdxl-vae` | 500 | shared real class (500) | yes | yes |
| E7 `stable-diffusion-3.5-large` | 500 | shared real class | yes | yes |
| E7 `FLUX.1`, **luma-only variant** | 500 | shared real class | yes | yes |
| E8 `Flux` | 250 | **no** | yes | — |
| E8 `GoT` | 250 | no | yes | — |
| E8 `Infinity` | 250 | no | yes | — |
| E8 `NOVA` | 250 | no | yes | — |
| E8 `OmniGen` | 250 | no | yes | — |
| E3 matched, by generator (GPT Image 2, Nano-Banana2) | ≤1,500 total | yes | yes | yes |
| E4 WildRF matched | ≤1,500 | yes | yes | yes |
| E5 Chameleon **pipeline-matched** | 2,000 + 2,000 | yes | yes | yes |

E7's in-bank subsets under P-A — `sd-vae-ft-ema`, `sd-vae-ft-mse`,
`stable-diffusion-2-1` — are reported but **do not enter M1 or M2**. The mapping from
an E7 subset name to a bank member is asserted in `p2_score.py`, not inferred: a subset
that failed to map would otherwise be marked unseen by default and inflate every
generalisation claim.

E8 has no real class, so **AUC on it alone is undefined** (D-035) and it contributes to
M1 only, at thresholds fitted on validation.

## 4. Metrics

For every stream, every fusion and every slice:

- **AUC** (slices with reals only);
- **AP**;
- **TPR at 1% and 5% FPR**, at thresholds fixed on validation reals;
- **balanced accuracy** at the validation 5% FPR threshold;
- **ECE**, 15 equal-width bins [R14];
- 95% **stratified bootstrap** CIs, 1,000 resamples, resampled within slice [R17].

**TPR@1%FPR is pre-declared noisy.** Validation holds roughly 480 reals, so a 1% FPR
threshold rests on about five false positives and moves substantially with one image.
It is reported for completeness. **TPR@5%FPR is the gate metric**, and no conclusion
rests on the 1% column.

## 5. Complementarity (RQ1)

Per pair of streams, per slice and pooled, at validation-fitted thresholds:

- **double-fault rate** — the share both get wrong. This is the ceiling on what any
  fusion can fix, and it is the number the RQ1 argument rests on [R16];
- **disagreement rate**;
- **Yule's Q** [R16], reported as *undefined* rather than 0 where one stream is perfect
  on a slice;
- **McNemar** on the paired errors [R18] — exact binomial below 25 discordant pairs,
  continuity-corrected chi-square above, and the test used is named beside the p-value;
- **Spearman** correlation of the scores;
- **oracle headroom** — F-oracle minus the best single stream. F-oracle uses the labels
  and is therefore **not a method**; it is an upper bound and is never reported as a
  result.

## 6. The go / no-go criterion

- **M1** = macro-average TPR@5%FPR over the unseen slices of §3.
- **M2** = macro-average AUC over the unseen slices of §3 that have reals.

> **GO** if `F-stack − best single stream ≥ 2.0 points` on **M1 or M2**, *and* the 95%
> **paired** bootstrap CI of that difference excludes 0.
> Otherwise **NO-GO**.

Three clauses that close the obvious loopholes:

1. **The best single stream is fixed on validation**, by validation AUC, before any
   panel score exists. It is not the stream that happens to do worst on the panel.
2. **The bootstrap is paired**: F-stack and the single stream are resampled through the
   *same* indices, stratified by slice, so the interval is on their difference rather
   than two overlapping intervals on their levels. 1,000 resamples.
3. **Tie rule.** If M1 and M2 disagree in sign, GO still requires the criterion on
   either, as stated. The disagreement is recorded in `go_no_go.json` and stated in the
   report; it is not resolved after the fact.

`F-stack` is the logistic stacker of §6 of the plan, fitted on validation only.
`dbp.eval.fusion.FittedFusion.score` raises if it is ever applied to the split it was
fitted on.

## 7. Mandatory secondary reports (no gate attached)

- per content type on E3 (texture / structure / hybrid);
- in-bank vs out-of-bank on E7, per protocol;
- the DIV2K lossless diagnostic, as the JPEG-reliance check;
- the training ablation `gen_matched` vs raw generations, S1 and S2 on P-A, budget
  permitting;
- **every trivial-rule baseline printed beside its number** (D-044): E7 FLUX.1's
  `4:2:0 → fake` at 100.00% (lift +49.85) and quality at +14.35; E4 matched's quality
  separability at +11.39; E3's Nano-Banana2 residual at +6.47.

## 8. E7 FLUX.1 — the chroma control

On that subset a rule reading only chroma subsampling is a perfect classifier, and the
set was distributed that way: it says nothing about FLUX's VAE (§6d). Re-encoding does
not fix it, because chroma subsampling is destructive.

So FLUX.1 is reported as **a pair of numbers, standard / luma-only**, always beside the
trivial baseline. The luma-only variant converts both classes to greyscale-in-RGB
before scoring, after which chroma carries no information; the models have seen
greyscale during training through D-041's synchronized augmentation, at p ≈ 0.075.
**The luma-only number is the one that enters M1 and M2.** An unchecked FLUX.1 number
is not claimable.

## 9. Declared sensitivity analysis for P4 — not part of this gate

**Repeat the go / no-go with S1 on frozen DINOv3 ViT-L/16.**

P2 runs ViT-B/16 for both S1 and S2, because no committed measurement licenses ViT-L
(Phase A, A3: P1's backbone table is empty and P0's LoRA entry is a dry run with no
optimizer). That choice has a known direction of risk: **a weaker semantic stream could
inflate the apparent fusion gain**, because F-stack would be beating a lower bar. If
P2 returns GO, this sensitivity analysis is how that GO is stress-tested; it is
declared here so it cannot later look like a post-hoc rescue of a NO-GO.

It is explicitly **outside** the P2 gate. Nothing in §6 changes because of it.

## 10. What would invalidate this pre-registration

Stated so the failure is visible rather than absorbed:

- the panel being scored more than once per frozen configuration, other than after a
  crash recorded in the log (B7);
- any hyperparameter, threshold, temperature or stacker coefficient chosen by looking
  at panel scores;
- an S2 run trained for fewer than 2,000 steps — the approved floor, below which the
  stream comparison is not valid and the run stops and asks;
- a change to the split, whose seed (2026) every calibration in P2 depends on.

---

**References** as numbered in the P2 plan: [R14] Guo et al. 2017 · [R16] Kuncheva &
Whitaker 2003 · [R17] Efron & Tibshirani 1993 · [R18] Dietterich 1998.

---

## Amendment 1 — 2026-10-03 — the evaluation-crop eligibility rule (I-047)

**No panel score exists.** This amendment is written and committed before any has been
produced: the S1 panel feature pass failed partway through with
`AugmentError: image 422x317 is smaller than the 320px centre crop`, and no scoring
script has been run against the panel at any point. The gate criterion in §6 is
unchanged — this amendment changes which rows the macro averages are taken over, and
nothing else.

### The defect

The panel was drawn without checking that every selected image survives the
pre-registered 320 px centre crop (D-045). **375 of the 12,150 rows have a shorter side
below 320**, the smallest being 150 px. Such a row cannot be centre-cropped to the
evaluation crop, so it cannot be scored by S1, S2 or S3 at all. The panel sampler now
enforces `shorter side >= eval_crop` for every future draw
(`dbp.eval.panel.draw(min_side=EVAL_CROP)`), with a test that fails against the old
behaviour.

### The rule

A panel row is **eligible** if the shorter side its own manifest records is at least the
evaluation crop, 320 px. The test is on pixels alone and **never on the label**, so it
cannot become a way of removing inconvenient rows from one class. Ineligible rows are
removed; **there is no backfill**, because drawing replacements would be a second,
unseeded draw and the panel's reproducibility rests on the first one. Rows whose
manifest records no size are kept and counted separately — the T-ctrl diagnostic panel
is the only such case, and P1 built it at 384.

The alternatives were considered and rejected: up-scaling the short side to 320, or
padding to it, would keep every row but stamp a resampling or border signature on
exactly the undersized subset — and in E3 that subset is **reals only**. That is a
D-042-class shortcut, manufactured by us rather than inherited from the distributor, in
the direction that would most flatter the result. Lowering `eval_crop` to 256 was
rejected because S2 trains on 320 crops, so evaluating at 256 is a train/test mismatch.

### What it removed

| view | dropped | fake | real |
|---|---:|---|---|
| e3_matched | 23 | 0 / 1000 | **23 / 500** |
| e4_wildrf_matched | 8 | 2 / 751 | 6 / 749 |
| e5_chameleon_pipeline | 252 | 116 / 2000 | 136 / 2000 |
| e7_dda_coco | 92 | 78 / 3000 | 14 / 500 |
| e8_evalgen | 0 | — | — |
| t_ctrl_diagnostic | 0 | — | — |
| **total** | **375** | | |

**The E3 class shift is declared, not corrected.** 23 reals were removed and no fakes,
so E3's real class shrinks by 4.6% while its fake class is untouched. This is a change
in class balance arising from a label-blind pixel rule — E3's fakes are generated at
high resolution and its reals are web images — and it is **not** a signal. It is stated
here, and must be stated wherever an E3 number is reported, alongside the duplicate
caveat required for E4.

### New panel

| | before | after |
|---|---:|---:|
| panel rows | 12,150 | **11,775** |
| luma companion | 1,000 | **978** (FLUX.1 492, real 486) |

Digests, recomputed over the surviving rows — the per-row `sha256` values are unchanged,
only rows were removed:

| | digest |
|---|---|
| panel, before | `2e1c192024ec52e42c4e1761afa6fb2e37c0f7a6983f5ff75d1174305550b769` |
| panel, **after** | `bff608c3d8e00ea834532cd541849b4f21de803c8f8e27a55ca5e70424c51116` |
| luma, before | `ac948b11adbded1e99aa01d81ed4f4f1dcb829f89cd059f612f70af7ed0eb881` |
| luma, **after** | `190279cc889da9c4d879c4d7ffd18afae42949c58896a983a94f1bd6ec9c2303` |

The panel **directory is untouched**: all 13,150 files remain in `dbp-p2-panel`. Only
the lists and the digests change, so nothing is re-downloaded and no pixel is rewritten.

### New slice counts entering M1 and M2

E7's subsets are the per-VAE slices; §8's substitution is unaffected and still replaces
`e7_dda_coco/FLUX.1` with `e7_dda_coco_luma/FLUX.1`.

| slice | before | after |
|---|---:|---:|
| e7_dda_coco/FLUX.1 | 500 | 492 |
| e7_dda_coco/sd-vae-ft-ema | 500 | 489 |
| e7_dda_coco/sd-vae-ft-mse | 500 | 481 |
| e7_dda_coco/sdxl-vae | 500 | 491 |
| e7_dda_coco/stable-diffusion-2-1 | 500 | 482 |
| e7_dda_coco/stable-diffusion-3.5-large | 500 | 487 |
| e7_dda_coco/real | 500 | 486 |
| e3_matched (both generators) | 1,500 | 1,477 |
| e4_wildrf_matched | 1,500 | 1,492 |
| e5_chameleon_pipeline | 4,000 | 3,748 |
| e8_evalgen | 1,250 | 1,250 |
| t_ctrl_diagnostic (no gate, §7) | 400 | 400 |

§6's criterion is unchanged: GO if `F-stack − best single stream >= 2.0 points` on M1 or
M2 with the 95% paired bootstrap CI excluding 0. The three loophole-closing clauses are
unchanged. The noise warning on TPR@1%FPR stands and is, if anything, slightly stronger:
the E7 real slice is now 486 rather than 500.

---

## Amendment 2 — 2026-10-05 — the governing fusion, the best-single rule, the seed rule, and four analysis fixes (I-056, I-060, I-061, I-062, D-068)

**No panel metric exists.** This amendment is written and committed before any panel row
is scored by a head or a fusion, and before any panel label is joined to any score. The
only panel-side artefacts are the S2 score files written by the Session D runbook
(`outputs/p2/s2/<run id>/scores/{panel,luma}.csv`, columns `image_id, score`, no label);
they have been checked for row counts (11,775 and 978) and sha256 only, and nothing has
been computed from them. `p2_score.py` has never been run against the panel.

### What was seen before this amendment was written

Disclosed in full, because it is what motivated the amendment. **All of it is
validation data; none of it is panel data.** The fusion check (user instruction
2026-10-05) fitted the heads and both fusions as §6 specified, on each protocol's
validation set (D-068, below), and reported:

| run | validation rows (real / fake) | AUC S1 | S2 | S3-LR | S3-Set | AEROBLADE | F-stack standardised weight |
|---|---|---|---|---|---|---|---|
| `p_a_sd1_seed0` | 932 (491 / 441) | 0.95899 | 0.99997 | **1.00000** | 0.97783 | 0.63033 | **S3-LR 97.1%**, S2 2.7%, S1 0.2% |
| `p_a_sd1_seed1` | 932 (491 / 441) | 0.95899 | 0.99990 | **1.00000** | 0.98061 | 0.63033 | **S3-LR 98.5%**, S2 1.3%, S1 0.2% |
| `4ch_only_seed0` | 1,550 (491 / 1,059) | 0.95027 | 0.99973 | 0.99733 | 0.99350 | 0.66060 | S2 56.5%, S3-Set 23.0%, S3-LR 20.2% |
| `full_bank_seed0` | 2,000 (491 / 1,509) | 0.89837 | 0.99757 | 0.97268 | 0.98435 | 0.72473 | S2 52.6%, S3-Set 28.9%, S3-LR 10.7%, S1 7.9% |

and, on the **full** 2,000-row validation set, the P-A S3-LR head's AUC on each decoder
**outside** the protocol: SDXL 0.163, SDXL-fp16fix 0.167, FLUX.1 0.222, FLUX.2-klein
0.296, Qwen-Image 0.378, SD3.5 0.407, MoVQ 0.424, DC-AE 0.436, Real-ESRGAN 0.683, against
1.000 on all three in-protocol decoders. **These out-of-protocol validation numbers
were seen before this amendment was written**, as were the per-decoder validation AUCs
of the other four heads (S3-Set 0.84–0.999 on the same decoders). Source:
`reports/p2_heads_<run id>.json`, keys `fusion_check` and
`per_group_val_auc_full_validation`, produced by `scripts/p2_fit_heads.py` at commit
`b1d0f8e`.

### Why §6 cannot decide P-A as written (I-060)

P-A's validation set holds one lineage, by design. On it S3-LR separates real from fake
perfectly, so the logistic stacker has no finite solution and puts 97–98% of its weight
on S3-LR, and "the best single stream by validation AUC" is also S3-LR. §6 would then
compare F-stack with the stream F-stack is made of, and the outcome would be a NO-GO by
construction, carrying no information about RQ1. S2 is also at 0.9999 on the same set,
so any rule that excludes a head for "perfectly separating validation" would be
arbitrary. Every learned weight and every validation-based choice is undetermined on
saturated data; the fix is to stop learning them there, not to choose a different set
to learn them on (fitting on all-lineage validation would leak the unseen decoders
into P-A, and E7's unseen subsets are reconstructions by those same VAEs).

### The governing analysis (replaces §6's fusion and best-single clauses for the verdict)

**1. Fusion members, by design.** The governing fusion is **F-mean(A2)** (`f_mean_rz`)
over **S1** (linear probe), **S2** (LoRA) and **S3-Set**. S3-Set is the thesis's
designated S3 head — permutation-invariant, the basis of the P3 plug-in experiment
(P2 prompt B5); S3-LR and AEROBLADE were always ablation and baseline heads. The members
are fixed by that design role, not by any validation number, and there is no exclusion
rule.

**2. Normalisation on validation reals only.** For each member stream, on that run's
validation reals (the same 491 images in every protocol):

```
z(x) = clip( (x - median_real) / (1.4826 * MAD_real), -5, +5 )
```

`MAD_real` is the median absolute deviation of the validation reals' scores; 1.4826 is
the normal-consistency constant, so one unit is one standard deviation for Gaussian
reals and the clip sits at five. If `MAD_real = 0` the scale falls back to the
(population) standard deviation of the validation reals; if that is also 0 the fit
raises. `F-mean(A2) = (z_S1 + z_S2 + z_S3-Set) / 3`. **No fake label, learned weight or
temperature enters the governing fusion** — stacking and temperature scaling have no
finite solution on perfectly separated data. The same three medians and scales are
applied unchanged to every panel and luma row.

**3. Thresholds, as pre-registered (§4): on validation reals only (I-062).** For every
head and every fusion, the threshold at FPR α ∈ {1%, 5%} is **the lowest
validation-real score s such that the share of validation reals scoring ≥ s is ≤ α**;
a score ≥ the threshold is called fake. With 491 reals that is the 24th-highest real
score at 5% (FPR 4.89%) and the 4th-highest at 1% (FPR 0.81%), ties moving it up.
No fake score enters any threshold. *Why this is a fix:* the committed
`calibrate_at_fpr` picks the threshold with the best validation TPR under the FPR
budget. Where a head separates validation perfectly, that is the lowest validation fake,
far above the 5% real quantile: P-A's S2 "5% FPR" threshold ran at **0.20%** FPR on the
validation reals (seed 1: 0.61%) and S3-LR's at **0.00%**. The single heads would then be
judged at much stricter operating points than F-mean on M1, which flatters the fusion.

**4. The best single stream.** Not chosen on validation: with S2 and S3-LR both at
≈1.000 there, the choice is arbitrary. **For each metric separately, best single = the
maximum over all five single heads (S1, S2, S3-LR, S3-Set, AEROBLADE) of that head's
panel macro**, and the maximum is **recomputed inside every bootstrap resample**. This
is the conservative direction: the fusion is compared with whichever head did best on
the panel, so selection noise favours the single head, never the fusion.

**5. The gate statistic and rule.** For metric M ∈ {M1, M2}:
`Δ_M = M(F-mean(A2)) − max over the five single heads of M(head)`, in points.
**GO if Δ ≥ 2.0 points on M1 or M2, and the 95% paired bootstrap CI of that Δ
excludes 0.** The margin, the confidence level, the 1,000 resamples and the tie rule
(§6, clause 3) are unchanged.

**6. The seed rule.** P-A was trained with two seeds. **The P-A verdict is GO only if
both `p_a_sd1_seed0` and `p_a_sd1_seed1` pass**; each seed is reported separately with
its own Δ and CI. **P-A remains the governing protocol.** `4ch_only` and `full_bank`
are run through the identical procedure (unseen slices are those whose generator the
protocol did not train on, read from `RunConfig.subset_status`) and reported as
secondary results; they have no gate role.

### What is still run and reported unchanged, as secondary (Analysis A)

The original §6 analysis is run and reported in full beside the governing one, with
**no** role in the verdict: **F-stack** (the logistic stacker over all five heads,
temperature-scaled, fitted on the protocol's validation set) minus **the best single
stream fixed on validation AUC**, with the same slices, thresholds (point 3) and
macro bootstrap as below. The original temperature-scaled F-mean over all five heads is
reported too. So the pre-registered analysis is not suppressed; it is demoted, for the
reason given above, before any panel number exists.

### Fixes that apply to every analysis

**D-068 — the protocol decides what every fit may see.** S1 trains on the S1 list's
training reals and the training fakes of the protocol's lineages (plus SD1.5
`gen_matched`). S3-LR and S3-Set train on the 1,000 S3 training reals and the
protocol's own 1,000-fake pick, re-drawn with the list builder's seeded
`balanced_take` (the three protocols' picks reproduce the probe list's 2,861 training
fakes and 1,000 training reals exactly). The committed fitting script trained S1 and S3
on every lineage whatever the protocol, while S2 respected it.

**The validation set** of a run is the probe list's 2,000 validation rows restricted to
the protocol: every validation real (491) plus the validation fakes whose lineage the
protocol trains on (and `gen_matched`). That is **932 rows for P-A, 1,550 for
`4ch_only` and 2,000 for `full_bank`**. Every validation-fitted quantity — S1's `C`, the
set head's epoch, every threshold, the F-mean(A2) normalisation, and Analysis A's
temperatures and stacker — is fitted on it and on nothing else.

**I-056 — every join is by image id.** Streams, labels and metadata are joined by
`image_id` through `dbp.eval.join`, never by position. A duplicate id, a missing id, an
unexpected id, a row count other than the expected (11,775 panel, 978 luma, 6,883 S2
validation) or a label that differs between two sources raises, naming the first ten
ids. A missing score is an error, never a NaN that silently leaves a fusion or a
metric. Every fusion records the streams it used, asserted equal to its configuration.

**I-061 — slices and the bootstrap.**

*(a) Slices with a shared real class.* E7 and E3 carry their reals as a class of the
view, not of any generator subset. Each generator slice is therefore scored as **that
generator's fakes against the view's whole real class**:

| slice family | fakes | reals |
|---|---|---|
| `e7_dda_coco/<source_vae>` | that VAE's fakes | E7's 486 reals |
| `e7_dda_coco_luma/FLUX.1` (§8) | 492 luma FLUX.1 | the 486 luma reals |
| `e3_matched/<source_label>` | that generator's fakes | E3's 477 reals |
| `e4_wildrf_matched`, `e5_chameleon_pipeline` | the view's fakes | the view's reals |
| `e8_evalgen/<generator>` | that generator's fakes | none (M1 only, §3) |

Without this, E7's and E3's slices hold no reals, their AUC is undefined, and they
would silently leave M2. Under P-A, M1 is the mean over the 12 unseen slices of §3
(E7 SDXL, SD3.5, FLUX.1-luma; E3 × 2; E4; E5; E8 × 5) and M2 over the 7 of them with
reals. **A slice the macro needs but cannot compute is an error, never skipped.**

*(b) The bootstrap resamples the macro.* The committed code bootstrapped a *pooled*
TPR/AUC over all gate rows while the point estimate is a macro over slices. Now each
replicate draws, with replacement and with one generator seeded 2026, **the fakes of
every slice within that slice and the reals of every view within that view's real
pool**; a view's resampled reals are shared by all of its slices, and a luma row is the
same resampling unit as its parent image. The **same** draw is applied to every head and
fusion (paired), M1 and M2 are recomputed as macros, the best single is re-maximised
(point 4), and the CI is the 2.5th–97.5th percentile of Δ over the 1,000 replicates.

*(c) Complementarity* (§5) is computed per slice on the same slices, at the point-3
thresholds, for every pair of single heads, and adds the **Jaccard index of the misses**
(|both miss| / |either misses|; undefined, not 0, when neither misses) beside McNemar.
A non-finite score raises; a stream is never dropped from the comparison. F-oracle rows
(correct if any member head is correct) are reported per slice for the F-mean(A2)
members and for all five heads, as upper bounds only.

### Exploratory, outside the gate

Reported in P2 and labelled exploratory: the per-decoder validation AUC table of every
head (the first part of the lineage-transfer result), and the Spearman correlation
across the 12 bank decoders between each decoder's mean reconstruction PSNR on the 491
validation reals (S3's `psnr_db` feature) and P-A S2's validation AUC on that decoder's
fakes. Neither touches the panel and neither enters the verdict.

### Unchanged

§2 (protocols), §3 (the unseen slices and the E7 in-bank exclusions), §4's metric list,
§5's measures, §7, §8 (FLUX.1 enters through its luma variant), §9 and §10 (one scoring
of the panel per frozen configuration; nothing chosen by looking at panel scores; the
split seed 2026) all stand. The panel is scored and analysed **once**, for all four
runs, after this amendment is committed; nothing in it may change after panel results
are seen.
