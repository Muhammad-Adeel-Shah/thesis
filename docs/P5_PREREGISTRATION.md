# P5 pre-registration — DRAFT (not frozen)

**Status: DRAFT, 2026-11-24.**
- **It is frozen,** tagged `p5-prereg-frozen`, audited and pushed right after the panel
  review, and **before any P5 model trains and before any panel clip is scored**.
- **Values:**
  - Every value marked *(user, date)* was set by the user.
  - Every value marked **PROPOSED** is the agent's proposal and needs the user's decision
    before the freeze.
  - Every `⟨…⟩` is a placeholder for a number that comes from the panel session or the
    A10G smoke. Nothing else may be a placeholder.
- **Changes after the freeze:** only as dated amendments, committed before any panel score
  exists. Nothing changes after a panel result has been seen.
- **The plan behind it:** `docs/P5_PLAN.md` (approvals A1–A8 and every decision block since).

**Project:** Decoder-Bank Probing (DBP) · 24i-8017, MS Data Science, FAST-NUCES.
**Context:** P3 established S2 (DINOv3 ViT-B/16 + LoRA, trained on a decoder bank) for images.
P5 asks the same questions for video, with a bank of video VAEs. The pilot (EXPLORATORY,
2026-11-21) showed the image S2-D per frame at clip AUC 0.882–0.910 on raw clips, and
0.6425–0.7543 after one libx264 CRF 23 re-encode
(`reports/kaggle_20261121/p5_pilot/p5_pilot_summary.json`). So nothing is "already caught",
and the video model must be trained through re-encodes.

---

## 1. What is being decided

- **RQ-V1 (generalisation):** does a frame/temporal S2 trained on a bank of video-VAE
  reconstructions detect videos from video-VAE lineages it never saw?
- **RQ-V2 (the bank):** does the bank beat a single VAE on generators neither saw?
- **RQ-V3 (time):** does a temporal head on the same backbone add to per-frame scoring?
- **RQ-V4 (deployability):** does the operating point hold on unfamiliar real videos, at
  k = 0 and with k target reals?

All outcomes are reported unchanged. A failed hypothesis is a finding.

## 2. Data

### 2.1 Training (Pexels; AWS only)

- **Source:** `zengxianyu/open-sora-pexels-subset` @ `c595493c834ec2c47ceee5d18ecca002aa646540`,
  treated as the Pexels License (research, private, no redistribution; user, 2026-11-21).
- **The training pool:** shards 4:35 of the pre-declared shard plan.
  - It shares no video id with the pilot pool.
  - Eligible clips are ranked by `sha256("2026:" + video_id)`, and the first 2,200 are kept,
    one clip per video.
- **Splits** (`dbp.data.video.p5_splits`, fixed 2026-11-22 before any training clip was
  reconstructed): ids ordered by `sha256("2026:p5-split:" + id)`. The first 200 are
  **threshold-held-out**, the next 200 **validation**, and the remaining 1,800 **train**.
- **Clips:**
  - 25 consecutive native frames from a seeded start (`window_start`);
  - centre crop 320, never resized, no temporal resampling;
  - the pre-declared exclusions (`EXCLUSIONS`);
  - PNG, lossless.
- **The threshold-held-out reals are never reconstructed** *(user, 2026-11-22)*. They are
  scored on Kaggle only.
  - The Kaggle session rebuilds exactly these 200 clips from the same shards, by
    `p5_splits`, with the same protocol.
  - It refuses unless every `clip_digest` equals the AWS manifest's.

### 2.2 The video decoder bank (`configs/video_bank.yaml`, diffusers 0.42.0)

| key | lineage | precision for the bank (rule §2.3) |
|---|---|---|
| `wan21` | Wan2.1 VAE | **fp16**: pilot passed; A10G min 62.337 dB, re-run delta 0 (A10G smoke, `reports/aws_20261124/session1/reports/p5_train_recon/`) |
| `hunyuan` | HunyuanVideo VAE | **the decision rule below**; **fp16** if kept: A10G min 60.536 dB, re-run delta 0, run on NVIDIA A10G (A10G smoke, `reports/aws_20261124/session1/reports/p5_train_recon/`) |
| `cogvx` | CogVideoX VAE | **fp32** *(user, 2026-11-22: determinism)* |
| `ltx090` | LTX-Video 0.9 VAE | **fp16**: pilot passed; A10G min 59.453 dB, re-run delta 0 (A10G smoke, `reports/aws_20261124/session1/reports/p5_train_recon/`) |
| `mochi` | Mochi VAE | **fp32** (its fp16 check failed: pilot 45.09 dB, A10G min 47.796 dB) |
| `sd15_pf` | SD1.5 VAE per frame (the image lineage) | **fp16**: pilot passed; A10G min 60.972 dB, re-run delta 0 (A10G smoke, `reports/aws_20261124/session1/reports/p5_train_recon/`) |
| `wan22_5b` | reserve (descriptive only, V1-c) | — |
| `allegro` | **excluded** (diffusers implements it only with tiling, which the protocol forbids) | — |

- **The HunyuanVideo rule** *(user, 2026-11-21/22/24; pre-declared)*:
  - It stays in the bank if it runs on the A10G: deterministic, fits, correct frame count.
  - The user then decides it against the reconstruction share ($45), with the measured
    cost. The user may re-allocate from the reserve.
  - **If it leaves the bank:**
    - F1's three folds are wan21, ltx090, cogvx (§6.6);
    - the HunyuanVideo and AccVideo panel slices move to the out-of-bank (unseen) stratum.
  - **The decision:** ⟨kept / dropped, date⟩.
- **Lineage caveat** (pilot): Qwen-Image's VAE shares Wan2.1's encoder (all 84 encoder
  tensors equal after bf16 rounding), but **the decoders differ** (0 of 106 tensors equal in
  fp32).
  - This is recorded as a caveat. There is no "seen decoder" flag *(user, 2026-11-22)*.

### 2.3 The reconstruction protocol (`reconstruct.py`, unchanged in spirit from the images)

- `latent_dist.mode()`; no scaling or shift; clamp, round, uint8; PNG.
- No spatial or temporal tiling, and no frame-wise blending. A native causal cache is
  allowed.
- Deterministic: `seed_everything(2026, deterministic=True)`, cuDNN deterministic, TF32 off.
  Every member is re-run in the smoke with a delta of 0.
- **Half precision** *(user, 2026-11-22/24)*: fp16 only for a member whose fp16 check passed
  in the pilot AND again in this run's A10G smoke, deterministically.
  - The check is ≥ 55 dB against its own fp32 at every clip, finite, and a matching re-run.
  - HunyuanVideo's check is the A10G smoke's alone, recorded as run on an A10G.
  - mochi and cogvx are fp32.
  - A member is never built in two precisions.
- **What is built:** 2,000 clips (train + validation) × the bank's members.

### 2.4 The training pairs and their degradation *(user, 2026-11-22/23/24)*

- **A pair** is a real clip and its reconstruction by one member. Both share the decode,
  window and crop path, and are stored lossless.
- **The format shortcut is removed:** each reconstruction is re-encoded with libx264 at the
  CRF whose bits-per-pixel best matches its real source clip, seeded, then decoded (P5_PLAN
  §3.4).
- **The re-encode degradation:**
  - It applies to **exactly p = 0.80** of the training pairs, **identically to both clips
    of the pair**.
  - Codec: **H.264 with p = 0.7, H.265 with p = 0.3**.
  - CRF: an integer, **uniform over 18–35**, preset medium, yuv420p, native fps.
  - Drawn once per pair, seeded.
- **The other degradations** of P3's S2-D chain keep **p = 0.35**.
- **x265 is not byte-deterministic** (3 outputs in 10 encodes, 2026-11-23).
  - So every H.265-degraded clip is **encoded once, written and hashed**, and the hashes are
    the record *(user, 2026-11-24, option a)*. H.264 stays byte-checked.
  - **A limitation:** "x265 output is not byte-deterministic".
- **PROPOSED, how it is stored:**
  - The re-encoded bitstreams (mp4, about 50 KB per clip) are stored, not the decoded
    frames, and are decoded at training time.
  - Decoding a stored bitstream is exact, so the hashes fix the training data.
- **The thesis states** that V5 tests robustness *inside* the trained range: CRF 23/28/35
  for H.264 and 28/35 for H.265 are all within 18–35.
- **The provenance of the recipe:** it was fixed from training-domain pilot data only
  (Pexels reals and their reconstructions), never from test data.

### 2.5 The test panel (Kaggle only; `scripts/p5_fetch_panel.py`, P5_PLAN §3.3)

- **AIGVDBench** (`AIGVDBench/AIGVDBench` @ `bf48acaa4920990af3dd2a511ea88e63354df305`) is
  the primary view.
- **AEGIS hard test** (`Clarifiedfish/AEGIS` @ `d2ad8bd3…`) is secondary: clean only and
  descriptive (its licence forbids modification, so it has no V5 views).
- **The layout,** measured from the zip directories:
  - every open-source fake has a same-named real;
  - the 11 closed-source zips share 2,000 numeric names, none of them a real.
- **The universe:** the official `Split/test.jsonl` ∩ Real.zip (⟨n⟩ names, ⟨g⟩ source
  videos).
- **Grouping:** by source video (stricter than by name). **A name, with its real and every
  fake made from it, is in exactly one pool**, and no source video spans two pools *(user,
  2026-11-23)*.
- **The pools.** Whole groups, ordered by `sha256("2026:p5-panel:" + source)`:
  - **k-shot calibration** 300 → ⟨n⟩;
  - **closed-source reals** 300 → ⟨n⟩;
  - **open-source pool** 900 → ⟨n⟩.
- **Open-source slices** (20 zips): per zip, the first 150 names in
  `sha256("2026:p5-panel:" + zip + ":" + name)` order whose real **and** that zip's fake are
  both eligible.
  - These are **content-matched pairs**, reported as the content-matched view.
  - Counts per slice: ⟨…⟩.
- **Closed-source slices** (11 zips): one shared order, `sha256("2026:p5-panel:closed:" +
  name)`; each zip keeps its first 150 eligible fakes, against the closed-source real pool.
  - Counts per slice: ⟨…⟩.
- **Eligibility:**
  - 8-bit, not rotated, short side ≥ 320;
  - a seeded 17-frame window (scoring uses frames 1–16), centre crop 320, never resized;
  - not a still.
  - A slice stops after 50 consecutive ineligible fakes.
  - The exclusions are counted per zip: ⟨…⟩.
- **Duplicates:**
  - sha256, plus pHash ≤ 6 bits on keyframes 1/8/16, within and across views;
  - also against every training, validation, threshold and pilot clip;
  - plus the I2V conditioning-leak check (each panel fake's frame 1 against every
    training-real frame).
  - Matches are removed and counted per view: ⟨…⟩.
  - A real and its own fake are counted as `own_pair`, not as duplicates.
- **The metadata gate** (fps, width, height, duration, bitrate, reported frames, codec,
  profile, pix_fmt):
  - a direction-free single-feature AUC per slice, its fakes against its reals.
  - The flags are ⟨…⟩, and the user's rebalancing decision, made before the freeze, is
    ⟨…⟩. Nothing is dropped automatically.
  - Wherever a slice is reported, its flags are stated with it.
- **The views:**
  - **The primary view is FORMAT-MATCHED:** reals and fakes are re-encoded identically.
    **PROPOSED:** libx264 CRF 23, preset medium, yuv420p, native fps, the pilot's setting.
  - **Raw** is secondary.
  - **The content-matched view** is the open-source slices' paired reals.
- **The panel strata** (P5_PLAN §3.2):
  - **in-bank VAE:** the generator's VAE is hash- or tensor-identical to a bank member;
  - **out-of-bank VAE;**
  - **closed / unknown.**
  - "**Unseen**" means out-of-bank and closed.
  - The tensor checks still open (AccVideo's Hunyuan VAE, LTX's version, the SD-family
    generators) are resolved before the freeze: ⟨…⟩.
- **The panel digest:** ⟨sha256⟩, committed before any P5 model trains.
- **Scored once per frozen model.** A re-run is allowed only after a crash, logged with its
  reason.
- **Licences (§6c):**
  - research only throughout;
  - AEGIS: no modification;
  - the HunyuanVideo outputs stay private.
  - **No AIGVDBench number is reported before its `docs/CITATIONS.md` entry is filled.**

## 3. Models

| id | what | seeds |
|---|---|---|
| `S2V-F` | **frame model:** the S2 recipe per frame (DINOv3 ViT-B/16 + LoRA r = 8, batch 16, LR 1e-4, wd 0.01, warm-up 300, crop 320), on the §2.4 pairs. **`N*` = 5,000 steps, inherited from P3** (no new selection) | 0, 1 |
| `S2V-T` | **temporal model:** `S2V-F`'s frozen backbone. Per-frame features `f_t` and deltas `f_t − f_{t−1}`, t = 1..16, feed a 2-layer transformer head of width 256, trained on cached features | 0, 1 |
| `S2V-1` | the same two stages, trained on `wan21` reconstructions only (V2) | 0, 1 |
| `LOLO-v` | the same two stages, with lineage v's reconstructions held out; `sd15_pf` is always in. **F1 has fired:** 3 folds (§6.6) | 0 |
| frozen comparators | image `S2-D` ViT-B s0/s1 (P3), per frame; DDA per frame (P4); D3 (training-free); DeCoF (`AIGVDBench/Weights/DeCoF.tar`, `weights_only=True`, sha256 recorded; dropped if unavailable, pre-declared) | — |

**PROPOSED (not set by the user yet):**
- **`S2V-F` frame sampling:** each step draws a pair and one frame index t ∈ 1..24, the same
  t for the real and the reconstruction, seeded. Frame 0 of a causal VAE is never used.
- **`S2V-T` training:**
  - features are cached once per pair, from the pair's stored degraded version;
  - AdamW, LR 3e-4, wd 0.01, batch 64, 3,000 steps, seeded;
  - no selection on any panel data.
- **Precision:** P3's S2 recipe (AMP as in P3) on the A10G.
  - Versions are recorded on both sides.
  - The frozen model hashes are checked before scoring, on the T4.
- **Clip scores:**
  - **Frame models** (`S2V-F`, image S2-D, DDA per frame): the mean of the frame logits
    over frames 1–16 of the clip's 17-frame window.
  - **`S2V-T`:** its clip logit.

## 4. Metrics

- **AUC** (primary, threshold-free) and **AP**, clip level, per slice. A slice is one
  generator's fakes against its real set:
  - **open-source slices:** their paired reals (content-matched);
  - **closed-source slices:** the closed-source real pool.
- **The k = 0 threshold:** 5% FPR on the 200 threshold-held-out Pexels reals, scored on
  Kaggle (P5_PLAN §5).
- **Achieved FPR and TPR** at that threshold; slice-calibrated TPR@5%FPR (descriptive); ECE
  (15 bins).
- **Macro averages** are over the **unseen** slices (out-of-bank + closed) of AIGVDBench.
- **Confidence intervals:**
  - bootstrap, **2,000 resamples, seed 2026**, paired for every comparison;
  - fakes resampled within slice, reals within their real set;
  - **97.5%** for V1 and V2 (Bonferroni over 2), and 95% otherwise.

## 5. Fixed before the panel is scored

- The recipe of §2.4 and the models of §3 (including every PROPOSED value, once set).
- The bank membership and precisions of §2.2 and §2.3.
- The panel, its digest, the gate decisions and the strata.
- `N*` = 5,000, inherited. **No value in this document is chosen from panel data.**

## 6. Hypotheses and decision rules (the thresholds are P3's)

### 6.1 PRIMARY V1 — generalisation to unseen video-VAE lineages (H1 analogue)

- **V1-a.** `G_V` = the mean over the LOLO folds of the held-out-lineage AUC (`S2V-T` of
  fold v). This is the validation reals vs their reconstructions by the held-out member.
  - **Success:** `G_V ≥ 0.90`, the 97.5% CI lower bound `≥ 0.85`, and no fold below `0.75`.
- **V1-b.** For each fold, the AUC on the panel's **in-bank** slices whose VAE is the held-out
  lineage, where a slice has ≥ 100 fakes.
  - **Success:** each such slice ≥ **0.85**.
  - A fold with no qualifying slice is reported as "no panel slice".
- **V1-c (descriptive):** `wan22_5b` reconstructions of the validation reals.

### 6.2 PRIMARY V2 — the bank vs a single VAE on unseen generators (H1-c / H4 analogue)

- **Success:** `S2V-T` minus `S2V-1`, macro AUC over the unseen slices, ≥ **+2 points**, with
  the 97.5% paired CI excluding 0, **on both seeds**.

### 6.3 SECONDARY V3 — time helps

- **Success:** `S2V-T` minus `S2V-F` (mean frame logit), macro AUC over the unseen slices,
  ≥ **+2 points**, 95% CI excluding 0, on both seeds.
- It is also reported per fps stratum.

### 6.4 SECONDARY V4 — false alarms at k = 0, and k-shot (H3 / H3-k analogue)

- **k = 0:**
  - **PROPOSED real pools:** AIGVDBench test reals (open-source paired reals and
    closed-source reals, each clip once) and the AEGIS reals (youtube, dvf), descriptive.
  - **Success:** AIGVDBench's achieved FPR ≤ **15%**, and no AIGVDBench real set above
    **25%**.
- **k-shot** (if F2 has not fired):
  - k ∈ {0, 10, 50, 100}, drawn from the **300-name calibration pool**, 1,000 draws, seed
    2026.
  - Methods **(Q)** and **(S)** exactly as P3 §6.3, with the same binomial j values.
  - **Success, per method:** at k = 50, achieved FPR ≤ **7.5%** in ≥ **80%** of draws,
    judged on AIGVDBench. AEGIS is descriptive.

### 6.5 SECONDARY V5 — robustness (H5 analogue; AIGVDBench only)

- **PROPOSED:** each condition replaces the format-matching encode and is applied
  identically to reals and fakes.
- **Success:** H.264 CRF 28, and resize-and-restore 0.5 (then the format-matching encode),
  each within **5 AUC points** of the format-matched clean view.
- **Descriptive:** H.264 CRF 23/35, H.265 CRF 28/35 (encoded once and hashed, §2.4), and an
  fps change (every 2nd frame).
- **If F3 fires:** H.264 only (CRF 28; 23 and 35 descriptive).

### 6.6 Fallback status (pre-declared order; each trigger logged with its time)

- **F1 FIRED** at 2026-11-23 23:59 PKT: the training data were not complete. **LOLO has 3
  folds:**
  - with HunyuanVideo: **wan21, hunyuan, ltx090**;
  - without it: **wan21, ltx090, cogvx** *(user, 2026-11-23)*.
- **F2** (drop k-shot): if scoring has not started by Wed 25 Nov, end of day. ⟨fired / not⟩
- **F3** (robustness H.264 only): if clean scoring is not done by Thu 26 Nov, 12:00.
  ⟨fired / not⟩
- **F4** (baselines D3 + S2-D per frame): if F3 fires, or if DeCoF or DDA fails its
  pre-flight. ⟨fired / not⟩

### 6.7 DESCRIPTIVE V6 — baselines

- **The video bank adds value:** `S2V-T` minus image `S2-D` per frame ≥ **+2 points**, 95% CI
  excluding 0, on both seeds.
- DDA per frame, D3 and DeCoF: paired, descriptive.
- If DeCoF was trained on AIGVDBench's train split, it is reported as an in-domain
  reference, not as a cross-generator baseline.

## 7. What counts as success

| claim in the thesis | requires |
|---|---|
| "a bank-trained video detector **generalises** to unseen video-VAE lineages" | V1-a, plus V1-b on every fold that has a qualifying slice |
| "**the bank beats one VAE** on generators neither saw" | V2 |
| "**time adds** to per-frame scoring" | V3 |
| "**deployable** at k = 0 / with k reals" | V4 |
| "**robust** to ordinary re-encoding" | V5 (inside the trained range, stated) |

**Failure paths.**
- If V1 fails, the generalisation claim is withdrawn, and the per-fold table shows where.
- If V4 k = 0 fails but k = 50 passes, the recommendation is "S2V-T plus k-shot
  calibration", stated as the result.

## 8. What would invalidate this pre-registration

- The panel scored more than once per frozen model (except after a logged crash).
- Any value in §5 chosen while panel scores existed.
- A panel clip or source video found in the training, validation, threshold or pilot data
  after the freeze. This is recorded, and the affected rows are reported separately.
- Test data leaving Kaggle.

---

**Values set by the user** (dates in P5_PLAN):
- A5: clips of 25 frames, crop 320, pairs lossless.
- The recipe: p = 0.80; H.264 0.7 / H.265 0.3; CRF 18–35; other degradations p = 0.35.
- Half precision rules; mochi and cogvx fp32.
- The HunyuanVideo rule; F1's folds.
- The panel principles of 2026-11-23.
- libx265, option (a).
- The thresholds are P3's.

**PROPOSED, to be set before the freeze:**
- the storage of the re-encodes (§2.4);
- `S2V-F` frame sampling and `S2V-T` training (§3);
- the format-matched setting (§2.5);
- V4's real pools (§6.4);
- the composition of the V5 conditions (§6.5).

**Placeholders (`⟨…⟩`), from the panel session and the A10G smoke:**
- the universe and pool counts; the slice counts and exclusions;
- the gate flags and the rebalancing decision; the dedup counts; the strata checks; the
  digest;
- the fp16 results per member; the HunyuanVideo decision;
- F2–F4 status.
