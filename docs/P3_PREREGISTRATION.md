# P3 pre-registration — FROZEN

**Status: FROZEN, 2026-10-08.**
- **What is frozen:** the text of draft commit `b562216ea1793e530ba9f55e68ca8e1a5b555634`,
  unchanged except for this status block.
- **When:** frozen before any P3 model was trained and before Panel B was drawn.
- **Values:** every value was set by the user on 2026-10-08.
- **Resolved before freezing:** E3's unstated real-image source (§2, the
  source-statement principle) and the BANK-k composition (§3).
- **Changes from here:** only as dated amendments, committed before Panel B is scored.
  Nothing may change after any Panel B result is seen.
- **The plan behind it:** `docs/P3_PLAN.md`.

**Project:** Decoder-Bank Probing (DBP) · 24i-8017, MS Data Science, FAST-NUCES.
**Context:** P2 returned NO-GO (`p2-complete`). P3 is the pre-registered NO-GO path: the
best single stream (S2) plus calibration, studied for deployment.

---

## 1. What is being decided

- **RQ-G (generalisation):** does S2 trained on a decoder bank detect images from
  decoder lineages and generators it never saw?
- **RQ-D (deployability):** can its operating point be made to hold on real images from
  unfamiliar sources, without target-domain fakes? I-064: at the validation threshold,
  S2 called 69–84% of E3/E4/E5 panel reals fake.

All outcomes are reported unchanged. A failed hypothesis is a finding.

## 2. Data

- **Development:** train and validation splits (P2's, seed 2026), plus the **P2 panel**.
  Every choice in §5 that is "tuned" is tuned here.
- **Test: Panel B**, drawn and frozen as `docs/P3_PLAN.md` §1 specifies, with:
  - disjointness by `sha256`, pHash ≤ 6 bits and source image (E7: COCO id) against the
    P2 panel, every training image and the new real sources;
  - crop eligibility ≥ 320;
  - a digest committed before any P3 model is trained.
- **Panel B is scored once per frozen model.** A re-run is allowed only after a crash,
  recorded in the log with its reason.
- **Panel B contains no MNW.** When MNW's download route resolves, it becomes **Panel C**:
  drawn with the same disjointness rules, frozen with its own digest, scored **once**,
  and analysed by **this frozen analysis unchanged**. Panel C results are reported
  separately and do not alter any Panel B verdict.
- **E8** is in Panel B as a **secondary, fakes-only view**: TPR at the k = 0 threshold,
  descriptive. It is not in any macro and has no decision rule.
- **E3 is small:** at most 226 fakes per generator and at most 176 test reals. Its CIs
  are wide, it is reported with its n, and no conclusion rests on E3 alone.
- **Calibration pools:** each Panel B view's reals are split, before scoring, into a
  **test** pool and a **calibration** pool (E3 50; E4 100; **E5 500**; E7 100). E5's test
  pool stays at 2,000 reals. Calibration reals are used only by §6.3 and never as test
  data.
- **A new real source must not share a platform with any Panel B view's real images.**
  - **How it is checked:** from each test dataset's paper or README, recorded as a table
    in the log.
  - **Excluded:** Unsplash, and any set built from it, because it is E5's (Chameleon's)
    real source.
  - **The new real source:** Open Images V7 only (Flickr).
- **The source-statement principle** (user decision, 2026-10-08):
  - **A view whose real-image source is stated nowhere is excluded from every FPR
    decision rule.** That is E3: its Hugging Face README, its paper and its code README
    name no source (log P3-A part 3).
  - **A view with a stated source is kept**, and any unnamed extra platforms are a
    recorded limitation. That is E5: "platforms like Unsplash".
  - E3's FPR remains **descriptive**, including in H3's three-step table.
- **Deduplication of the new real source.** The Open Images draw is sha256- and
  pHash-checked (≤ 6 bits) against **all** real images of E3, E4 and E5 — the full
  datasets, not only the Panel B rows — as well as Panel B, the P2 panel, the training
  corpus and COCO.
  - **Matches** are removed from the draw, and the match counts are logged per view.
  - **Descriptive only, not a gate:** how many E3 reals pHash-match COCO train2017 +
    val2017.

## 3. Models

| id | what | seeds |
|---|---|---|
| `S2-B` | the P3 baseline: S2 on all 8 lineages, the P3 recipe (fixed `N*` steps), no new reals, no degradation | 0, 1 |
| `S2-D` | **the deployable model**: S2-B + Open Images V7 reals (**3,600 train + 400 held out**, with their bank reconstructions) + the degradation chains | 0, 1 |
| `S2-B+deg`, `S2-B+reals` | single-factor ablations | 0 |
| `LOLO-f`, f = 1..8 | S2 with lineage f's members held out (fold 1 also holds out `gen_matched`) | 0 |
| `BANK-k-o` | S2 on k lineages, order o (k = 1, 2, 4, 6, 8; o = near, far; k = 1 and 8 shared). **k = 1 is sd15 alone** — one decoder, the prior-work setting. For **k ≥ 2 whole lineages are added, and SD1 is complete from k = 2** (sd15, sd21 ≡ sd_ft_mse, sd_ft_ema). **k = 8 is the full 12-decoder bank = `S2-B`**, unchanged. Decoder counts, verified against `configs/decoder_bank.yaml`: near-first 1 / 5 / 8 / 10 / 12; far-first 1 / 4 / 6 / 9 / 12. Every table and plot shows the decoder count beside k | 0 |
| `BANK-1L` | **descriptive only**: the whole SD1 lineage (3 decoders), same recipe, `N*` steps, `gen_matched` kept. It splits the k = 1 → 2 jump into "more decoders of the same family" (sd15 → SD1) and "a new family" (SD1 → SD1 + a 2nd lineage). No hypothesis is attached | 0 |
| P2 references | P2 `full_bank_seed0` S2 and `p_a_sd1_seed0` S2, unchanged | — |

**`N*` (model selection without saturation, I-059):**
- every LOLO fold trains for `N_max` = **6,000** steps and evaluates its held-out
  lineage every 500 steps;
- `N*` is the step that maximises the mean held-out-lineage AUC across the 8 folds;
- fold f's own numbers are reported at `N*(-f)`, computed from the other 7 folds;
- every non-LOLO model trains for exactly `N*` steps, with no further selection.

## 4. Metrics

- **AUC** (primary, threshold-free) and **AP**, per slice. A slice is one generator's
  fakes against the view's whole real class (P2 Amendment 2, I-061).
- **Achieved FPR on test reals** at the deployed threshold:
  - **k = 0**, for every model, fixed at 5% FPR by the reals-only rule (I-062) on **the
    union of the 491 COCO validation reals and the 400 held-out Open Images reals**. This
    was decided on 2026-10-08, before Panel B existed. The **COCO-only** threshold (P2's
    rule) is reported beside it;
  - and the k-shot thresholds of §6.3.
- **TPR at the deployed threshold.** Also **slice-calibrated TPR@5%FPR** (threshold on
  the slice's own test reals; descriptive).
- **ECE**, 15 equal-width bins, on sigmoid(score) after the §6.3 shift.
- **Macro averages.**
  - **AUC and AP** are averaged over the four **unseen slices**: E3 GPT-Image2, E3
    Nano-Banana2, E4, E5.
  - **Every FPR decision rule** (H3, H3-k) is taken over the real pools **E4 and E5**,
    each counted once. E3 is excluded by the source-statement principle (§2); its FPR is
    reported descriptively, and its two slices share one real pool. The "no real pool
    above 25%" rule is likewise taken per pool, over E4 and E5.
  - **MNW is Panel C only; E8 is in no macro (§2).** E7 is in-bank for full-bank models,
    so it is excluded from these macros and used by H1-b and §6.5.
- **Confidence intervals:**
  - 95% bootstrap, **2,000 resamples, seed 2026**;
  - fakes resampled within slice, reals within the view's real pool;
  - **paired** for every comparison;
  - for the two primary hypotheses, **97.5%** intervals (Bonferroni over 2).

## 5. Fixed before Panel B is scored (tuned on development only)

- Degradation chain strengths and probabilities (`P3_PLAN.md` §4b).
- `N*`.
- The choice and number of new real sources.
- The k-shot methods of §6.3.
- `N_max`.

Each is recorded in the log with the development numbers that chose it.

## 6. Hypotheses and decision rules

### 6.1 PRIMARY H1 — generalisation to unseen decoder lineages (RQ-G)

- **H1-a.** `G` = the mean over the 8 LOLO folds of the held-out-lineage AUC. This is
  validation reals vs their reconstructions by the held-out members, at `N*(-f)`.
  **Success:** `G ≥ 0.90`, the 97.5% CI lower bound `≥ 0.85`, and no fold below `0.75`.
- **H1-b (Panel B, true unseen VAEs).** For folds 1–3, AUC on Panel B's E7 subsets of
  the held-out lineage:
  - fold 1: sd-vae-ft-ema, sd-vae-ft-mse, sd2-1;
  - fold 2: sdxl-vae;
  - fold 3: SD3.5 and FLUX.1 **luma**.

  Reported with CIs. **Success:** each subset ≥ 0.85. FLUX.1 standard is reported beside
  it as an upper bound (§6d).
- **H1-c (SECONDARY): one decoder vs the bank, on unseen lineages.**
  - **Comparison:** for folds f = 2..8, AUC on lineage f's held-out validation
    reconstructions, LOLO-f minus `BANK-1` (sd15 only, same recipe, `N*` steps).
    Neither model trained on lineage f, and both score the same images (paired).
  - **Success:** the mean of (LOLO-f − `BANK-1`) over folds 2–8 is ≥ **+2 AUC points**,
    and its 95% paired bootstrap CI excludes 0. Every fold is reported.

### 6.2 PRIMARY H3 — false alarms at deployment, no target data (RQ-D)

At k = 0 (the union threshold of §4), on the Panel B test reals of the real pools **E4
and E5**, each counted once. E3 is excluded from this decision rule by the
source-statement principle (§2):
- **Success (both seeds):**
  - `S2-D`'s macro achieved FPR over E4 and E5 ≤ **15%**;
  - neither E4 nor E5 above **25%**;
  - **and** a paired reduction against the P2 `full_bank` S2, scored at the same k = 0
    union rule, of ≥ **50% relative**, with its 97.5% CI excluding 0.
- **Reported descriptively:**
  - whether the macro achieved FPR also reaches ≤ 10%;
  - a **three-step table** per real pool (**E3 included, descriptive**) and macro: P2
    `full_bank` at the COCO-only threshold → P2 `full_bank` at the union threshold (the
    **threshold effect**) → `S2-D` at the union threshold (the **model effect**).

### 6.3 SECONDARY H3-k — few-shot calibration (CPU)

**Draws.** k reals from each view's calibration pool, 1,000 random draws per k, seed 2026.

| pool | size | k values | notes |
|---|---|---|---|
| E5 | 500 | 0, 10, 50, 100 | genuine draws at every k |
| E4 | 100 | 0, 10, 50, 100 | **k = 100 is a single draw** (the whole pool). k = 50 draws come from a pool of 100, so they overlap, and the share of draws understates the variability (a stated limitation) |
| E3 | 50 | 0, 10, **25**, 50 | **k = 50 is a single draw**; k = 25 is added, descriptive. E3 is descriptive throughout |
| E7 | 100 | 0, 10, 50, 100 | descriptive; k = 100 is a single draw |

A **single-draw** cell is reported as one achieved FPR with a **bootstrap CI over the test
reals**, never as a share of draws.

**Methods.**
- **(Q) Conformal-style quantile.** The threshold is the j-th highest of the k reals,
  with j the largest value such that `P(Binomial(k, 0.075) ≤ j − 1) ≤ 0.10`, i.e.
  P(FPR > 7.5%) ≤ 10% by the binomial bound on exchangeable scores. Computed exactly with
  `scipy.stats.binom`:
  - **k = 50 → j = 1** (P = 0.0203; j = 2 would give 0.1025);
  - **k = 100 → j = 4** (P = 0.0526);
  - **k = 10 and k = 25:** no j meets the bound (j = 1 gives 0.4586 and 0.1424), so
    j = 1 is used and the bound is reported as **not met**.

  (Q) controls FPR by construction, so **its finding is the TPR it achieves**.
- **(S) Logit shift.** Scores are shifted by median(k reals) − median(validation reals),
  then the k = 0 threshold is applied.

**Reported for every k:** mean achieved test FPR, the share of draws with FPR ≤ 7.5% (or
the single-draw value with its CI), TPR and ECE.

**Success, per method:** at k = 50, achieved test FPR ≤ 7.5% in ≥ 80% of draws, **judged on
E4 and E5**. E3 and E7 are descriptive; no conclusion rests on E3 alone (§2).

### 6.4 SECONDARY H2 — the deployable model does not lose ranking

Macro AUC over the four unseen slices (as pre-decided), `S2-D` minus P2 `full_bank` S2.
**Non-inferior** if the 95% CI's lower bound is > **−2 points**, on both seeds.

**Also reported:** the same test without the two E3 slices (E4 and E5 only). A shared
platform would inflate `S2-D`'s E3 AUC. If the two verdicts differ, both are stated as
they are; neither replaces the other.

### 6.5 SECONDARY H4 — bank size

`BANK-8` minus `BANK-1` on Panel B macro AUC over unseen views.
- **Success:** ≥ **+2 points** with a 95% CI excluding 0.
- **Reported descriptively:** the full curve for both orders, with CIs, and the
  held-out-lineage validation AUC of the lineages outside each run. X3c's mixed result
  is the reason this is a test, not an assumption: full_bank beat P-A on E3 but not on
  E4.

### 6.6 SECONDARY H5 — robustness (descriptive with a threshold)

`S2-D` on Panel B under JPEG q50 and resize-and-restore 0.5.
- **Success:** each within **5 AUC points** of clean.
- All 12 conditions are reported as curves for `S2-D`, P2 `full_bank` and P2 P-A.

### 6.7 EXPLORATORY H6 — content-aware gate S2 + S3 (only if budget allows)

- **Gate:** chooses or weights S2 vs S3 by a content score that is trained or fixed on
  the **development** panel.
- **Success:** gate minus `S2-D` ≥ 2 points of macro AUC, CI excluding 0 (the P2 rule).
- It does not enter any primary claim.

## 7. What counts as success

| claim in the thesis | requires |
|---|---|
| "S2 on a decoder bank **generalises** to unseen decoder lineages" | H1-a, plus H1-b on ≥ 2 of 3 folds |
| "S2-D is **deployable without target data**" | H3, plus H2 (non-inferior ranking) |
| "**a few target-domain reals suffice** to restore the operating point" | H3-k at k = 50 |
| "**the bank beats one decoder on lineages neither saw**" | H1-c |
| "**more decoders help**" | H4 |

**Failure paths.**
- If **H3 fails but H3-k passes**, the deployment recommendation becomes "S2-D plus k-shot
  calibration". That is stated as the result, not as a rescue.
- If **H1 fails**, the generalisation claim is withdrawn, and the per-fold table shows
  which lineages fail.

## 8. What would invalidate this pre-registration

- Panel B scored more than once per frozen model (except after a logged crash).
- Any value in §5 chosen while Panel B scores existed.
- Any Panel B image, or source image, found in training or development data after the
  freeze. This is recorded, and the affected rows are reported separately, not
  silently dropped.
- A change to the P2 split (seed 2026), on which every validation number depends.

---

**Values set by the user, 2026-10-08:**
- `N_max` = 6,000;
- H1-a: 0.90 / 0.85 (97.5% lower bound) / 0.75;
- H1-b: 0.85 per subset;
- H1-c: +2 points, 95% CI;
- H2: −2 (non-inferiority);
- H3: 15% macro / 25% per real pool / 50% relative reduction (97.5% CI), over the real
  pools **E4 and E5** each once. E3 is descriptive (the source-statement principle);
  ≤ 10% and the three-step table are reported descriptively;
- H2: the four unseen slices as pre-decided, plus a report without E3 (both verdicts
  stated if they differ);
- BANK-k: SD1 complete from k = 2; decoder counts near 1/5/8/10/12, far 1/4/6/9/12;
  `BANK-1L` descriptive;
- H3-k: FPR ≤ 7.5% in ≥ 80% of draws at k = 50, per method, judged on E4 and E5;
- the new real source: Open Images V7 only, 3,600 train + 400 held out;
- calibration pools: E3 50, E4 100, E5 500, E7 100;
- H4: +2 points;
- H5: 5 points;
- E8: secondary, fakes-only;
- MNW: Panel C;
- seeds: two for S2-B and S2-D, one otherwise.
