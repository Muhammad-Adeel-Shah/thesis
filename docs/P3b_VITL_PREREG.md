# P3b — ViT-L follow-up: pre-registration (FROZEN)

**Status:** **frozen 2026-11-11**, by the commit tagged `p3b-vitl-prereg-frozen`.
- **When:** written before any ViT-L training and before any ViT-L score existed.
- **Approval:** the pass rules Q1–Q6 are approved by the user (review of `8800801`, with
  changes folded in).
- **Section 6** holds the Kaggle smoke's MEASURED numbers.

After the freeze, nothing in sections 1–5 changes.

**Why:** P4 compared our S2-D (DINOv3 **ViT-B/16**) with DDA (DINOv2 **ViT-L/14**). DDA's
macro AUC over the 4 unseen slices was 0.784 against S2-D's 0.755 / 0.760, a paired
difference of −2.9 [−6.0, +0.1] points for seed 0 (`reports/p4_results/tables/t9c_paired.md`).
The two backbones differ in size by about 3.5x. P3b removes that difference: the same
S2-D recipe on a ViT-L backbone, compared with DDA on the same slices.

---

## 1. The model: S2-D, with one change

**Unchanged** (the S2-D recipe of P3, as committed). The values are read from
`scripts/p2_train_s2.py`, `src/dbp/streams/s2_artifact.py` and S2-D seed 0's training
report (`reports/kaggle_20261023/p3_s2d_session1/p2_s2_s2d_p35_seed0.json`):

| item | value |
|---|---|
| decoder bank | all 12 distinct members (`configs/decoder_bank.yaml`), no lineage held out |
| training data | the P1 corpus and its pairs, gen_matched at 25% of fakes, plus the 3,600 Open Images train reals and their reconstructions by all 12 members; the 400 held-out Open Images reals never enter training (`p3_s2d_inputs.py`) |
| real pool | COCO and Open Images train reals, pooled and sampled uniformly by real (decision of 2026-10-21) |
| degradation | the real-side chain at p = 0.35 (`reports/p3_degradation_choice.json`, sha256 `6cda0151…e8d5`) |
| batch | 16: 8 reals, 6 reconstruction fakes, 2 gen_matched fakes (`batch_plan`) |
| crop | 320 |
| LoRA | r 8, alpha 8, dropout 0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| head | one linear layer on `[CLS ‖ mean patch]` (`build_head`) |
| optimiser | AdamW, lr 1e-4, weight decay 0.01, 300 warm-up steps, cosine |
| schedule | the 6,000-step schedule stopped at **N\* = 5,000** (`reports/p3_nstar.json`, sha256 `de30996d…b678`); validation on 512 pairs every 500 steps; no early stopping, no selection inside the run |
| precision | fp16 autocast with a GradScaler, as S2-D |
| gradient checkpointing | **on**, as S2-D (the trainer has always enabled it) |
| seeds | **0 and 1** |
| the trainer | **byte-identical**: `scripts/p2_train_s2.py`, sha256 `6b9c30c4…b7b6` (the guard in `tests/test_p3_s2d.py`, user decision 2026-10-21); the backbone is chosen by its existing `--backbone` option |

**The only change:** the backbone. DINOv3 ViT-B/16 (`dinov3-vitb16`) becomes **DINOv3
ViT-L/16**: `facebook/dinov3-vitl16-pretrain-lvd1689m` @
`ea8dc2863c51be0a264bab82070e3e8836b02d51` (`dinov3-vitl16` in
`src/dbp/backbones/loader.py`; `model.safetensors` 1,212,559,808 B, sha256
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`, Hub API 2026-11-11).
- **Same family, larger:** its config differs from ViT-B/16's only in `hidden_size`
  (1024), `intermediate_size` (4096), `num_attention_heads` (16) and `num_hidden_layers`
  (24). The patch size (16), the 4 register tokens and the preprocessor (mean, std,
  resizing) are identical.
- **So:** the head's input becomes 2 x 1024, and LoRA adapts 24 layers instead of 12.
  Nothing else follows from the change.

**No tuning on Panel B.** Nothing in this document is chosen from a Panel B score.
- N\* stays 5,000; it is not reselected for ViT-L.
- A run that **crashes** (the session dies) is resumed from its checkpoint. A run that
  crashes twice is reported as failed, never re-seeded.
- A run whose loss is NaN or diverges is **not** resumed: it follows the failure path in
  section 5.

## 2. Scoring

**Panel B is scored once per frozen ViT-L model.** The path is P3's final scoring
(`p3_final_score.py --backbone dinov3-vitl16`), with the committed lists and the deployed
protocol (centre crop 320, head logit):
- **clean:** Panel B (9,405), its luma companion (700) and the 891 threshold reals;
- **robust:** Panel B's 8,655 test rows under the two H5 decision conditions only, `jpeg50`
  and `resize050` (`dbp.eval.perturb`, unchanged).

Once only, with summaries, carry-forward and refusals as in P3 and P4. A fail-fast
projection from the smoke comes before any long job (user rule of 2026-11-09).

## 3. Comparators

All are committed, and nothing is re-scored:
- **DDA** (`reports/p4_baseline_scores/dda_native/`; under the conditions, `dda_jpeg50`
  and `dda_resize050`);
- **CO-SPY** (`cospy_native`);
- **S2-D ViT-B** seeds 0 and 1 (`reports/p3_final_scores/s2d_p35_seed{0,1}/`). ViT-L seed
  s is paired with ViT-B seed s.

DDA's and CO-SPY's scores are the P4 scores, reused and never re-scored.

**Limitation (stated wherever Q1 is reported):** the comparison with DDA is between models
of the same size, not the same backbone. Ours is DINOv3 ViT-L/16; DDA's is DINOv2
ViT-L/14. The pretraining (data, objective, version) and the patch size differ, so a
difference between the two is not attributable to the method alone.

## 4. Questions and pass rules (fixed before training)

**Statistics:**
- **Shared with P3 and P4:** slices, pools, the macro over the 4 unseen slices (E3
  GPT-Image-2, E3 Nano-Banana-2, E4, E5) and the k = 0 union threshold.
- **Bootstrap:** P3's `bootstrap`, unchanged: 2,000 resamples, seed 2026, real pools drawn
  once per resample and shared by every model in a comparison, percentile **95%**.
- **Signs:** every difference is ViT-L minus the comparator.
- **The primary question is Q1** (ViT-L vs DDA). Q2–Q6 are secondary. **All six are
  reported, whatever the outcome.**
- **Seed rule:** a question passes only if it passes for seed 0 **and** for seed 1.
  - If the two seeds disagree, the verdict is **"seed-dependent"**, which counts as not
    passed.
  - If both fail, the verdict is "not passed".
- **Reading a failed "matches":** failing Q1's or Q2's "matches" rule is **not** evidence
  that ViT-L is worse than the comparator; it means the margin could not be shown. The
  thesis says so wherever such a verdict is reported. Only a CI that lies wholly below 0
  shows a deficit.

| # | question | statistic | pass rule (fixed now) |
|---|---|---|---|
| Q1 | ViT-L vs DDA | paired macro AUC, ViT-L − DDA | **matches DDA** if the 95% CI's lower bound > −2 points on both seeds (P3 H2's margin); **beats DDA** if the lower bound > 0 on both seeds |
| Q2 | ViT-L vs CO-SPY | paired macro AUC, ViT-L − CO-SPY | the same two rules as Q1 |
| Q3 | does size help | paired macro AUC, ViT-L − ViT-B (same seed) | **gain** if the point ≥ +2 points and the CI's lower bound > 0, on both seeds (P3 H4's rule) |
| Q4 | false positives at k = 0 | achieved FPR on the E4 and E5 test reals at the model's own union threshold | **pass** if the macro (mean of E4 and E5) ≤ 15% and neither pool > 25%, on both seeds (P3 H3's absolute rule) |
| Q5 | k-shot calibration at k = 50 | P3's H3-k methods (Q) and (S) on each view's calibration pool | **pass**, per method, if the achieved test FPR ≤ 7.5% in ≥ 80% of draws on each of E4 and E5, on both seeds (P3 H3-k's rule) |
| Q6 | robustness | macro AUC under `jpeg50` and `resize050` minus clean (test rows) | **pass** if both are within 5 points of clean, on both seeds (P3 H5's rule) |

**Descriptive** (no rule; the same items as P4, the same bootstrap: B = 2,000, seed 2026,
the same slices; DDA's and CO-SPY's P4 scores reused, not re-scored):
- **robustness, paired:** ViT-L − DDA macro AUC under `jpeg50` and under `resize050`,
  against P4's `dda_jpeg50` and `dda_resize050`;
- **false positives, paired:** ΔFPR, ViT-L − DDA and ViT-L − CO-SPY, each model at its own
  k = 0 union threshold; also ViT-L − ViT-B;
- **E7:** AUC per VAE (DDA's home benchmark, stated as such);
- **E8:** TPR at the k = 0 threshold;
- every slice's AUC and AP;
- the training validation curve.

**Exploratory:** anything else, labelled so.

**Analysis code.** A thin driver (`scripts/p3b_analyze.py`) will call `p3_analyze.py`'s
and `p4_analyze.py`'s functions unchanged. It will be committed before any ViT-L Panel B
score exists, and it must reproduce the committed P3/P4 values of the comparators exactly.

## 5. Failure path (pre-declared)

**Training fails: stop and report.** A seed's run fails if any of these holds:
- the loss is NaN, or the run diverges: any recorded loss or validation AUC is not finite,
  or the training report's status is not `OK`;
- its validation AUC at step 5,000 is **more than 5 points below** S2-D ViT-B's for the
  same seed. This is the trainer's existing P3 validation set (512 pairs). ViT-B's values,
  MEASURED:
  - seed 0: **0.99453** (`reports/kaggle_20261023/p3_s2d_session1/p2_s2_s2d_p35_seed0.json`);
  - seed 1: **0.99423** (`reports/kaggle_20261024/p3_s2d_session2/p2_s2_s2d_p35_seed1.json`);
  - so a ViT-L value below 0.94453 (seed 0) or 0.94423 (seed 1) fails.

What happens then:
- the failure is reported with the training report and curve;
- there is **no re-tuning** (no change of LR, steps, schedule, crop, batch or LoRA);
- there is **no Panel B scoring** of that seed;
- the questions are reported as not answered, with the reason.

The check runs on the training reports, before any Panel B score.

**Out of memory on a T4.** If a T4 runs out of memory even with gradient checkpointing (the
smoke shows 2.136 GiB of 14.562; section 6):
- the **only** fallback is the university PC: 1x RTX 4000 Ada, 20 GB;
- it runs the byte-identical trainer, and its hardware is recorded;
- there is **no gradient accumulation and no trainer change**.

**The projected time is unaffordable.** If it does not fit the quota (section 6), the user
decides before any training.

## 6. Smoke and time plan (filled in from the Kaggle smoke before the freeze)

**The smoke** (`env/kaggle_run_p3b_vitl_smoke.py`):
- **What runs:** the unchanged trainer's own `--smoke 50` on ViT-L, run in-process by
  `scripts/p3b_smoke_probe.py`, which measures it from outside:
  - optimiser-step timing after 5 warm-up steps;
  - peak GPU memory;
  - for the measurement only, transformers' gradient-checkpointing switch made a no-op.
- **The two smokes:** gradient checkpointing on (the recipe), then off (would ViT-L fit
  without it).

**Run:** Kaggle T4 × 2, `dbp-code-8800801`, 2026-11-11, `muhammadadeelshah`. Archive
`F:\thesis-2\kaggle_raw\p3b_vitl_smoke_reports.zip` (25,041,554 B, sha256
`aefacb4e…2f19`); the smoke records are committed under `reports/kaggle_20261111/p3b_smoke/`.
- **Both smokes:** SMOKE-OK, exit 0, 50 optimiser steps, no out-of-memory, the trainer's
  sha256 equal to the guard (`6b9c30c4…b7b6`). The run is the recipe: ViT-L @ `ea8dc286…`,
  crop 320, batch 16 (8 / 6 / 2), LoRA on q/k/v/o, 1,572,864 of 304,702,464 parameters
  trainable (0.5162%).
- **Inputs identical to S2-D's:** the merged files' sha256, the pools (5,526 reals, Open
  Images share 0.651466) and the Open Images pairs' content sha256 equal S2-D session 1's;
  46,800 staged files hash-verified, 0 mismatched; 0 held-out rows anywhere.

| item | value |
|---|---|
| peak GPU memory, recipe (checkpointing on) | **2.136 GiB allocated / 2.676 GiB reserved** of 14.562 GiB (MEASURED) |
| seconds per step after warm-up, recipe | **1.7126 s/step** (max 1.8008), 2.14x ViT-B's 0.80041 (MEASURED) |
| measurement smoke (checkpointing off) | 1.3636 s/step, 7.243 GiB allocated / 7.721 GiB reserved (MEASURED) |
| projected GPU-h per 5,000-step run | **2.49** = 1.7126 s × 5,000 × ViT-B's overhead factor 1.048 (validation, checkpoints; the factor is ESTIMATED for ViT-L) |
| projected two seeds | 4.99 GPU-h; **about 2.5–2.8 h wall** in parallel on T4 × 2 (ESTIMATED: the smoke ran one job at a time, and two jobs share 4 CPU cores) |
| gradient checkpointing needed | **no**: ViT-L fits a T4 without it (7.2 of 14.6 GiB). The recipe keeps it **on**, because the trainer stays byte-identical; turning it off would save about 20% of step time and is not proposed |
| Panel B scoring per model | **about 0.46–0.76 h** (ESTIMATED): S2-D ViT-B seed 0 MEASURED 779.5 s for clean (203.1 s), `jpeg50` (231.3 s) and `resize050` (345.1 s) (`reports/p3_final/s2d_p35_seed0_{clean,robust}.json`), scaled by 2.14 (the step-time ratio) up to 3.5 (the parameter ratio) |

In fixed-steps mode the trainer records `allowance_hours` 2.25 but never caps steps on it
(`fixed_budget`), so a 2.49 GPU-h run trains all 5,000 steps.

**Time plan.** Quota is about 1 h per session hour.

| session | what | wall / quota (ESTIMATED from the MEASURED smoke) |
|---|---|---|
| training | seeds 0 and 1 in parallel on T4 × 2: 2.49 GPU-h each, plus about 0.15 h for setup and staging | about 2.7–3.0 h |
| scoring | both models in parallel: clean + `jpeg50` + `resize050` | about 0.7–1.0 h |
| **total** | | **about 3.4–4.0 h** |

**The earlier estimate is superseded.**
- **What it was:** the pre-smoke estimate, corrected by the review (2026-11-11) to
  1.165 h x 2.5–3.5 = 2.9–4.1 h per seed, about 5–7 h of quota in total.
- **What was measured:** ViT-L/ViT-B step time is 2.14, below that assumed 2.5–3.5. That
  gives 2.49 GPU-h per seed, about 3.4–4.0 h in total.
- **Margin:** the review's 5–7 h stays the planning margin, if the quota is short.
