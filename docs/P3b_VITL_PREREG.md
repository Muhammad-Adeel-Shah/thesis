# P3b — ViT-L follow-up: pre-registration (DRAFT, not frozen)

**Status:** draft of 2026-11-11, written **before any ViT-L training and before any ViT-L
score exists**. It is frozen by a commit tagged `p3b-vitl-prereg-frozen` once the user
has approved it and the Kaggle smoke (section 6) has been read. After the freeze, nothing
in sections 1–5 changes.

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
- A run that fails (NaN loss or a crash) is resumed from its checkpoint. A run that fails
  twice is reported as failed, never re-seeded.

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

## 4. Questions and pass rules (fixed before training)

**Statistics:**
- **Shared with P3 and P4:** slices, pools, the macro over the 4 unseen slices (E3
  GPT-Image-2, E3 Nano-Banana-2, E4, E5) and the k = 0 union threshold.
- **Bootstrap:** P3's `bootstrap`, unchanged: 2,000 resamples, seed 2026, real pools drawn
  once per resample and shared by every model in a comparison, percentile **95%**.
- **Signs:** every difference is ViT-L minus the comparator.
- **Judging:** a rule judged "on both seeds" passes only if it passes for seed 0 and for
  seed 1.

| # | question | statistic | pass rule (fixed now) |
|---|---|---|---|
| Q1 | ViT-L vs DDA | paired macro AUC, ViT-L − DDA | **matches DDA** if the 95% CI's lower bound > −2 points on both seeds (P3 H2's margin); **beats DDA** if the lower bound > 0 on both seeds |
| Q2 | ViT-L vs CO-SPY | paired macro AUC, ViT-L − CO-SPY | the same two rules as Q1 |
| Q3 | does size help | paired macro AUC, ViT-L − ViT-B (same seed) | **gain** if the point ≥ +2 points and the CI's lower bound > 0, on both seeds (P3 H4's rule) |
| Q4 | false positives at k = 0 | achieved FPR on the E4 and E5 test reals at the model's own union threshold | **pass** if the macro (mean of E4 and E5) ≤ 15% and neither pool > 25%, on both seeds (P3 H3's absolute rule) |
| Q5 | k-shot calibration at k = 50 | P3's H3-k methods (Q) and (S) on each view's calibration pool | **pass**, per method, if the achieved test FPR ≤ 7.5% in ≥ 80% of draws on each of E4 and E5, on both seeds (P3 H3-k's rule) |
| Q6 | robustness | macro AUC under `jpeg50` and `resize050` minus clean (test rows) | **pass** if both are within 5 points of clean, on both seeds (P3 H5's rule) |

**Descriptive** (no rule):
- every slice's AUC and AP;
- E7 per VAE (DDA's home benchmark, stated) and E8 TPR;
- the paired FPR differences against ViT-B and DDA;
- ViT-L − DDA under each H5 condition;
- the training validation curve.

**Exploratory:** anything else, labelled so.

**Analysis code.** A thin driver (`scripts/p3b_analyze.py`) will call `p3_analyze.py`'s
and `p4_analyze.py`'s functions unchanged. It will be committed before any ViT-L Panel B
score exists, and it must reproduce the committed P3/P4 values of the comparators exactly.

## 5. What would change the plan

- **The recipe smoke does not fit a T4.** If the recipe smoke (gradient checkpointing on)
  runs out of memory, the plan stops and the user decides; the recipe is not changed
  silently.
- **The projected time is unaffordable.** If it does not fit the quota (section 6), the
  user decides before any training.

## 6. Smoke and time plan (filled in from the Kaggle smoke before the freeze)

**The smoke** (`env/kaggle_run_p3b_vitl_smoke.py`):
- **What runs:** the unchanged trainer's own `--smoke 50` on ViT-L, run in-process by
  `scripts/p3b_smoke_probe.py`, which measures it from outside:
  - optimiser-step timing after 5 warm-up steps;
  - peak GPU memory;
  - for the measurement only, transformers' gradient-checkpointing switch made a no-op.
- **The two smokes:** gradient checkpointing on (the recipe), then off (would ViT-L fit
  without it).

| item | value |
|---|---|
| peak GPU memory, recipe (checkpointing on) | PENDING-GPU |
| seconds per step after warm-up | PENDING-GPU (ViT-B: 0.800 s/step, 1.165 GPU-h per run, MEASURED) |
| projected GPU-h per 5,000-step run | PENDING-GPU |
| gradient checkpointing needed | PENDING-GPU (a measurement smoke runs with it off) |
| Panel B scoring per model | ESTIMATED from P3's 0.019–0.031 s/image (ViT-B) x the smoke's ViT-L/ViT-B step-time ratio, over 10,996 + 2 x 8,655 images |
