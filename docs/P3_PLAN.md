# P3 plan — Phase A (plan and audit only; no training code)

**Status:** approved 2026-10-08. The pre-registration is **FROZEN**
(`docs/P3_PREREGISTRATION.md`, frozen from draft `b562216`). Phase B (data) may start; no
P3 model is trained before Panel B's digest is committed.

**Starting point:** P2 closed with **NO-GO** (tag `p2-complete`). Following the P2
pre-registration's NO-GO path, P3 is the deployment-oriented study of **the best single
stream, S2 trained on the decoder bank, plus calibration**.

**Goal:** make S2 trained on the 12-decoder bank deployable, and test whether it
generalises. The main obstacle is false alarms on unfamiliar real images (I-064, the
DIV2K diagnostic).

Every number below names its source. "Derived" means computed from measured inputs, with
the arithmetic shown. "NOT-MEASURED" means no command has produced it yet.

---

## 0. What P2 left us (the evidence this plan is built on)

| finding | value | source |
|---|---|---|
| best single detector on unseen slices | S2; macro AUC 0.770–0.804 across the four runs | `reports/exploratory_x/x2/x2a_macro.csv` |
| no fusion beats S2 | every pair and fusion gain CI < 0 | same |
| false alarms at the validation 5% threshold | S2 flags 69–84% of E3/E4/E5 panel reals (P-A s0) | `reports/p2_session_e/real/analysis/p_a_sd1_seed0/metrics.csv` |
| false alarms on never-JPEG photographs | S2 flags 31–54% of DIV2K reals vs 4–6% of COCO reals | `reports/p2_div2k/div2k_diagnostic.csv` |
| the ranking survives | S2 DIV2K AUC 0.91–0.96 | same |
| TPR at a true 5% FPR (slice-calibrated) | S2 0.36–0.48 | `x2a_macro.csv` |
| training breadth helps S2 on unseen generators | full_bank vs P-A on E3 GPT-Image2: 0.663 vs 0.492 AUC | `reports/exploratory_x/x3c/x3c_s2_auc.csv` |
| in-training validation AUC saturates | best 0.99998, cannot rank checkpoints (I-059) | `reports/kaggle_20261005/p2_s2_p_a_sd1_seed0.json` |
| frozen DINOv3 is weak on VAE-only images | gen-only probe AUC 0.55–0.89 on reconstructions | `reports/p2_x1/x1_results.json` |

The deployment problem is calibration first and ranking second.

---

## 1. Panel B — a fresh, once-only test panel

**Rule.** Every P3 claim is tested **once**, on Panel B. The P2 panel becomes a
**development** panel: everything tuned in P3 is tuned on it, or on validation, and never
on Panel B.

### 1.1 Disjointness, all enforced by the draw script, with hard failures

- **No shared image.** No `sha256`, and no pHash within 6 bits
  (`dbp.data.phash.DUPLICATE_THRESHOLD_BITS = 6`), against:
  - the P2 panel (12,150 drawn rows, including the 375 removed by I-047);
  - the P2 luma companion;
  - every training and validation image of P1–P3 (T-real, `gen_matched`, T-ctrl/DIV2K,
    the bank reconstructions);
  - the new real sources of §4a.
- **No shared source image.** A source image is identified as follows:
  - **E7:** its COCO id. All six VAE subsets and the reals are built from the same COCO
    val2017 images, so one COCO id can appear as a real and in several subsets. In the
    P2 panel, 745 of 2,634 COCO ids appear in more than one subset (computed this
    session from `p2_panel_sets/p2_panel_e7_dda_coco.csv`). **A COCO id used anywhere in
    the P2 panel is excluded from Panel B, and each COCO id appears at most once in
    Panel B.**
  - **E4, E5, E3:** the distributed file path.
- **Crop eligibility.** Shorter side ≥ 320 (the I-047 rule), label-blind, no backfill.
- **Within-panel duplicates** are removed as in P1: byte-identical first, then pHash.

### 1.2 Pools (unused after the P2 draw; before eligibility and deduplication)

| set | licence (data card) | pool (P1 eval summary) | used by P2 (draw) | unused, before eligibility |
|---|---|---|---|---|
| E3 MLLM, matched view | apache-2.0 | 726 real, 726 GPT-Image2, 726 Nano-Banana2 | 500 real, 500 + 500 fake | **226 / 226 / 226** |
| E4 WildRF, matched view | NOASSERTION, research use only | 1,251 real, 1,252 fake | 749 real, 751 fake | **502 / 501** |
| E5 Chameleon, pipeline-matched | academic research only | 14,863 real, 11,170 fake | 2,000 / 2,000 | **12,863 / 9,170** |
| E7 DDA-COCO | apache-2.0 | 5,000 real, 29,969 fake (6 VAEs) | 2,634 distinct COCO ids | **about 2,366 COCO ids** (if every subset uses the same 5,000 val2017 ids — consistent with 29,969 ≈ 6 × 5,000; to be verified in the draw) |
| E8 EvalGEN (optional, fakes only) | apache-2.0 | 5,000 fake | 1,250 | **3,750** |
| E6 MNW | evaluation-only, citation required | download route unresolved | — | **BLOCKED** (data card: "blocked on the download route") |

Pool sources: `reports/kaggle_20260928/p1_eval_*_summary.json` (`by_label`); P2 draw
counts from pre-registration Amendment 1; licences from `docs/data_cards/`.

### 1.3 Proposed Panel B composition

Each view's reals are split into a **test** pool and a **calibration** pool for §4c. The
two never overlap, and the calibration pool is never scored as test data.

| view | test fakes | test reals | calibration reals | notes |
|---|---|---|---|---|
| E3 matched (2 generators) | all eligible unused (≤ 226 each) | eligible unused minus calibration (≤ 176) | 50 | small; k-shot capped at 50 for E3 |
| E4 matched | ≤ 501 | ≤ 402 | 100 | duplicates removed; P2's 24 byte-identical groups are not reused |
| E5 pipeline-matched | 2,000 | 2,000 | **500** | seeded draw; the calibration pool is large enough for genuine k-shot draws (change 3) |
| E7 (6 VAE subsets + reals) | 300 per subset | 300 | 100 | one COCO id once; FLUX.1 also as a luma companion (§6d shortcut) |
| E8 (secondary, fakes only) | 250 per generator | — | — | **decided:** TPR at the k = 0 threshold, descriptive; not in any macro |
| E6 MNW | **not in Panel B** | | | **decided:** Panel B is frozen without MNW. MNW becomes a separate once-only **Panel C**, analysed with the same frozen analysis, when its download route resolves |

That is **about 9,750 rows with E8**, plus the FLUX.1 luma companion. "≤" counts are
upper bounds until the eligibility filter runs.

**E3 is small.** At most 226 fakes per generator and at most 176 test reals, so its
CIs will be wide. It is reported with its n beside every number, and no conclusion rests
on E3 alone.

**Important limitation.** For a full-bank model, all six E7 subsets are **in-bank**: the
decoders are bank members. So E7 tests unseen-ness only for the leave-one-lineage-out
folds (§2) and serves robustness (§5). The generalisation claims for the deployable
model rest on **E3, E4, E5** for AUC, and on **E4, E5** for the FPR decision rules (E3 is descriptive there: the source-statement principle). MNW enters only through Panel C, and E8 is descriptive.

**E3 limitation (user decision, 2026-10-11, option 1; no amendment).**
- **The rule kept:** E3's source identity stays the frozen rule — the file path, plus
  pHash ≤ 6.
- **Why it is a limitation:** E3's generation protocols (image-to-text-to-image, editing)
  start from a real image, so same-number real and fake files may share content.
- **What pHash shows:** no pixel link; same-number pairs in the P2 panel have median
  Hamming distance 32 (GPT-Image2) and 30 (Nano-Banana2), the same as random pairs (log
  P3-B).
- **Consequence:** some Panel B E3 rows may share content with P2-panel E3 rows.
- **How it is reported:** every Panel B E3 row carries `e3_number_in_p2` (true if its
  number appears in the P2 panel under any label), counted per label. Descriptive only:
  there is no split analysis, because too few rows are unflagged.

**Cost.** A Kaggle **CPU** session (no GPU quota): metadata, pHash and file copies, in
the same way the P2 panel was materialised. Run time NOT-MEASURED. Panel B is **frozen
with a digest and committed before any P3 model is trained**, so no training choice can
see it. The new real sources (§4a) are deduplicated against it.

---

## 2. Leave-one-lineage-out (LOLO) for S2 — 8 runs

There are 8 lineages (`configs/decoder_bank.yaml`); an alias is never split from its
twin:

| fold | held-out lineage | members held out |
|---|---|---|
| 1 | kl_4ch_sd1 | sd15, sd21 ≡ sd_ft_mse (byte-identical, P1-K), sd_ft_ema **+ `gen_matched`** (SD1.5 generations would leak the lineage) |
| 2 | kl_4ch_sdxl | sdxl, sdxl_fp16fix |
| 3 | kl_16ch | sd35, flux |
| 4 | dc_ae_f32 | dcae |
| 5 | gan_sr | realesrgan_x4 |
| 6 | kl_32ch_flux2 | flux2_klein |
| 7 | kl_16ch_qwen | qwen_image |
| 8 | vq_movq_4ch | movq_k22 |

**Each fold reports:**
1. **AUC on the held-out lineage's reconstructions**: the 491 validation reals vs their
   reconstructions by the held-out members (paired; 491 pairs per member, as in X1);
2. **Panel B** macro AUC and slice-calibrated TPR, scored once at the end;
3. **E7's held-out-lineage subsets on Panel B** (folds 1–3), the true unseen-VAE test.

**Model selection without saturation (I-059).** Every fold trains for a fixed `N_max`
steps and evaluates its held-out lineage every 500 steps:
- **Fold numbers:** fold `f` reports at step `N*(-f)`, the step that maximises the mean
  held-out-lineage AUC over the *other seven* folds. This is nested, so no fold picks
  its own step.
- **Deployable model:** it trains for `N*`, the step that maximises the mean over all
  eight folds. It is never selected on anything it was trained on.

In-protocol validation log-loss and TPR@1%FPR are recorded beside it as secondary
signals.

**H1-c — one decoder vs the bank on unseen lineages** (added 2026-10-08, change A).
- **Comparison:** for folds 2–8, LOLO-f vs `BANK-1` (sd15 only, same recipe, `N*`
  steps), on lineage f's held-out validation reconstructions. Neither model trained on
  lineage f.
- **Pairing:** the same 491 reals and their reconstructions are scored by both, so the
  comparison is paired.
- **Cost:** `BANK-1` is scored on those reconstructions at no extra training cost, about
  6,400 images × 0.028 s ≈ 3 min.

---

## 3. Bank-size curve for S2

- **Points:** k = 1, 2, 4, 6, 8.
  - **k = 1 is sd15 alone**: one decoder, the prior-work setting.
  - **k ≥ 2 adds whole lineages, and SD1 is complete from k = 2** (sd15, sd21 ≡
    sd_ft_mse, sd_ft_ema).
  - **k = 8 is the full 12-decoder bank**, which is `S2-B`, unchanged.
  - **Decoder counts** (verified this session against `configs/decoder_bank.yaml` and
    the committed `X3B_ORDERS`):

    | k | 1 | 2 | 4 | 6 | 8 |
    |---|---|---|---|---|---|
    | near-first | 1 | 5 | 8 | 10 | 12 |
    | far-first | 1 | 4 | 6 | 9 | 12 |

    At the same k the two orders have different decoder counts. Between k = 2 and 6 they
    share none (5/8/10 vs 4/6/9), so near-vs-far is compared **at equal k** and,
    descriptively, **against decoder count** (AUC plotted vs decoders). An exact
    equal-count pair exists only at k = 1 and k = 8.
  - **`BANK-1L`** (one descriptive run): the whole SD1 lineage (3 decoders), seed 0,
    same recipe, `N*` steps, `gen_matched` kept. It splits the k = 1 → 2 jump into "more
    decoders of the same family" (sd15 → SD1) and "a new family" (SD1 → SD1 + a 2nd
    lineage). No hypothesis is attached; H4 stays `BANK-8` − `BANK-1`.
- **Orders:** the two pre-fixed orders of X3b (`X3B_ORDERS` in
  `scripts/p2_exploratory_x.py`, committed before any X3b number existed), at the
  lineage level:
  - near-first: SD1, SDXL, MoVQ, 16ch, Qwen, FLUX.2, DC-AE, GAN-SR;
  - far-first: SD1, GAN-SR, DC-AE, FLUX.2, Qwen, 16ch, MoVQ, SDXL.
- **Shared runs:** k = 1 and k = 8 are shared by both orders, and k = 8 is the
  deployable baseline. That makes **7 new runs**: k = 1, plus k = 2, 4, 6 × two orders.
- **Equal compute:** the same `N*` steps, batch and real images in every run; only the
  lineages the fakes may come from change. `gen_matched` is kept in every run.
- **Reported:** Panel B macro AUC per k, with a CI, and the held-out-lineage validation
  AUC of the lineages outside the run.

---

## 4. False alarms and calibration (I-064 — the main deployment problem)

### 4a. More diverse real training images

**Decided (2026-10-08, revised by Change 1): Open Images V7 only, 4,000 images.**
- **Split:** 3,600 train and **400 held out as validation reals** (10%).
- **Pairs:** reconstructed through all 12 bank decoders, so it enters training as aligned
  real/reconstruction pairs.
- **Licence filter:** only images whose per-image metadata lists **CC BY 2.0** are kept.
- **Attribution:** a file under `docs/` (image id, author, original URL, licence) is
  committed with the draw.
- **Dropped:** Unsplash Lite (Change 1: it is E5's real source) and RAISE (user
  decision: no).

**Licence check (metadata only; no image fetched).** Raw page fetch on 2026-10-08, 07:54
UTC; the quote is verbatim from the fetched bytes.

| source | host contacted (metadata only) | page sha256 | verbatim licence text | what it means for P3 |
|---|---|---|---|---|
| **Open Images V7** | `storage.googleapis.com/openimages/web/factsfigures_v7.html` | `03f731ed…4fa4fb0` | "The annotations are licensed by Google LLC under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license. Note: while we tried to identify images that are licensed under a Creative Commons Attribution license, we make no representations or warranties regarding the license status of each image and you should verify the license for each image yourself." | Per-image licence filter and attribution file, as above. Splits on the same page: train 9,011,219 / validation 41,620 / test 125,436 images |

**The platform rule** (pre-registration §2). A new real source must not share a platform
with any Panel B view's real images. Checked from each test dataset's paper or README, in
the IMPLEMENTATION_LOG P3-A (part 3) table:
- **E4** is Reddit / Twitter / Facebook;
- **E5** is Unsplash ("platforms like Unsplash");
- **E3's real source is not stated** in its README, its paper or its code README. By
  the source-statement principle (pre-registration §2), E3 is excluded from every FPR
  decision rule, and its FPR is descriptive.

**Host REVERSED (user decision, 2026-10-15): the CVDF S3 copies.**
- **Why:** Flickr refused every route to the originals (Kaggle IPs; a home IP at 4
  connections and at 1 request per 6 s). Flickr does not allow bulk downloads of the
  originals, and that is respected.
- **The source:** the copies in the public bucket `open-images-dataset` (no credentials;
  host `open-images-dataset.s3.amazonaws.com` and its regional endpoint). The CVDF README
  says they are "rescaled to have at most 1024 pixels on their longest side".
- **What does not change:** the same Open Images V7 validation split, CC BY 2.0 rows,
  seed-2026 walk and attribution file.
- **Rationale:**
  - the test reals are web-processed, not camera originals (E5 Chameleon's reals are
    all exactly 500 px wide; E4 is social-media images);
  - S2-D's degradation chains supply the varied compression.
- **Rules kept:**
  - shorter side ≥ 384, no resizing, native 384 crop to PNG;
  - greyscale, CMYK, 16-bit, palette and alpha rejected;
  - dedup (sha256 + pHash ≤ 6) against COCO, Panel B, P2, the training corpus and all
    E3/E4/E5 reals;
  - the first 4,000 in seed order; a seeded 3,600 / 400 split.
- **Checksum:** `OriginalMD5` describes the Flickr original, so each copy's sha256 is
  recorded instead.
- **Where it runs:** one Kaggle CPU session, up to 8 connections. This closes the local
  route (ii) deviation.
- **Not used:** the ~105 Flickr originals already fetched (one processing history per
  source). They stay in `kaggle_raw`.
- **Pre-registration:** it says only "Open Images V7 only (Flickr)", the platform, so no
  amendment is needed (see the log).

**Fetch rules (user decision, 2026-10-11; host rule (a) and checksum rule (b) superseded
on 2026-10-15 as above).**
- **Host:** Flickr originals, not the S3 copies. The CVDF copies are rescaled to at most
  1024 px on the longest side, which is one resize and re-encode on every image; only the
  original uploads keep a genuine camera and JPEG history.
- **When:** after the Panel B draw and digest, because the dedup needs Panel B.
- **Rules:**
  - **(a) Hosts and requests.** `OriginalURL` from the metadata. The approved hosts are
    `farm*.staticflickr.com`, plus `live.staticflickr.com` if a request redirects there.
    Any other host is listed and waits for approval. At most 8 parallel requests, with
    retries and backoff.
  - **(b) Checksum.** Every file is checked against `OriginalMD5` (base64 MD5). A mismatch
    means a re-encode was served: it is rejected and counted.
  - **(c) Seeded over-draw.** All 41,620 CC BY 2.0 rows are put in a seed-2026 order. The
    walk keeps the first 4,000 that pass download, checksum, crop eligibility (≥ 320) and
    dedup (sha256, pHash ≤ 6) against Panel B, the P2 panel, the training corpus, COCO, and
    all reals of E3, E4 and E5. Every rejection is logged with its reason, and a seeded
    split then gives 3,600 train / 400 held out.
  - **(d) No resizing.** Crop at native resolution to the pipeline's COCO size before
    reconstruction, and store the crops as PNG. The original width/height distribution
    and the count of files with EXIF camera make/model are logged (counts only).
  - **(e) Disk.** Download into a temp directory; only the crops and the manifest go to
    the output. Download size, output size and run time are stated before the run.
  - **(f) Private.** The dataset is PRIVATE on `muhammadadeelshah`. The attribution file
    (image id, author, original URL, licence) is committed to `docs/`.

**Hosts the image fetch will need (to approve; not contacted for images).** The Open
Images image host named on its official download page. It is read from the metadata
files at the next step, shown to you with the count left after the CC BY 2.0 filter, and
added to the allowlist (§6) only with your approval.

**Deduplication** (user decision, 2026-10-08). The Open Images draw is sha256- and
pHash-checked (≤ 6 bits) against **all** real images of E3, E4 and E5 — the full
datasets, from their P1 eval manifests — and against Panel B, the P2 panel, the training
corpus and COCO. Matches are removed and logged per view.

**Descriptive only, not a gate:** the number of E3 reals that pHash-match COCO train2017
+ val2017. This runs on the same Kaggle CPU session as the dedup.

**Excluded as training sources, because they are test or development data, or share
their platform:**
- MNW, WildRF, Chameleon, the E3 and E8 sources;
- **Unsplash, and any set built from Unsplash** (E5's real source);
- COCO val2017 (E7's source images);
- DIV2K (T-ctrl).

Every new real is pHash-deduplicated (≤ 6 bits) and sha256-checked against Panel B, the
P2 panel, the training corpus and COCO; COCO is on Flickr, and so is Open Images. The
number removed is reported.

**Cost (derived).** The P1 measured reconstruction cost is 1.177 GPU-h per 1,000 reals
for the whole bank (T4, crop 384; `IMPLEMENTATION_LOG.md` P1-K, `median_ms_per_image =
4237.0`). 4,000 reals × 1.177 / 1,000 = **4.7 GPU-h**. Storage is about 4,000 × 13 PNGs
at 384²; its size is NOT-MEASURED. All datasets are **PRIVATE** on `muhammadadeelshah`.

**The deployed k = 0 threshold for S2-B and S2-D** (decided 2026-10-08, change B,
before Panel B exists):
- **Rule:** 5% FPR on the union of the 491 COCO validation reals and the **400 held-out
  Open Images reals**, by the reals-only rule of I-062.
- **Reported beside it:** the COCO-only threshold, i.e. P2's rule.
- **H3's three-step table:** P2 full_bank at COCO-only → P2 full_bank at the union (the
  threshold effect) → S2-D at the union (the model effect).

### 4b. Real-side degradation chains, applied identically to each real/reconstruction pair

This extends D-041's synchronised augmentation. One chain is drawn per pair and applied
to both members, so the degradation can never become a label cue:

| op | range | note |
|---|---|---|
| JPEG | quality 50–95, chroma 4:2:0 or 4:4:4 | |
| WebP | quality 50–95 | |
| resize-and-restore | scale 0.5–1.0, down then back up to the original size | the corpus crops are 384 px. A plain 0.5× downscale would leave 192 px, below the 320 crop, so the image is restored to its size; the reconstruction protocol's "never resize" (§7) governs decoder inputs, not augmentation |
| social-media style | resize-and-restore, then JPEG 70–85 | |
| none | p = 0.3 | so clean images stay in-distribution |

Strengths and probabilities are tuned on the **development** panel only. The CPU cost of
encoding on Kaggle's data loader is **NOT-MEASURED**; it is measured in the first smoke
run, with +25% contingency in the budget.

**Fixed 2026-10-17, before any candidate trains (user decisions; log P3-C of that date).**
- **The chain probability p** is the only tuned value. The table's "none | p = 0.3" means
  a chain is applied with **p = 0.7**, the plan default. The candidates are **p = 0.7 and
  p = 0.35** (half), each trained as `S2-B+deg` seed 0 on the P3 recipe. The selected
  candidate's run is the pre-registered `S2-B+deg` ablation.
- **Not tuned, fixed now:** when a chain applies, one of the four operations is picked
  with **equal probability (1/4 each)**; every parameter is **uniform over its stated
  range** (JPEG and WebP quality 50–95, resize scale 0.5–1.0, social-media JPEG 70–85);
  **JPEG chroma 4:2:0 vs 4:4:4 at 50/50**. One draw per pair, applied to both members.
- **Objective and guard:** as the log entry of 2026-10-17 states (dev macro achieved FPR
  over the E4 and E5 dev real pools at the COCO-only k = 0 threshold; dev macro AUC over
  the four unseen slices no more than 1 point below `S2-B` seed 0).

**Changed 2026-10-19, still before any candidate trains (user decisions; log P3-C of that
date).** Everything else above is unchanged: 4 operations at 1/4 each, candidates p = 0.7 /
0.35, the same rule and guard.
- **The social-media operation's JPEG is 4:2:0 only** (the platforms subsample chroma).
  This resolves the first build's ambiguity. The plain JPEG operation stays at 4:2:0 vs
  4:4:4, 50/50.
- **Re-share variant:** with probability **0.5** (fixed, not tuned) the social-media
  operation is applied **twice in a row**: resize-and-restore 0.5–1.0, then JPEG 70–85 at
  4:2:0, twice, the second round with independently drawn scale and quality. It is
  synchronised like every chain: one draw per pair, both members, both rounds.

### TrueFake — a secondary, descriptive social-media test (rule fixed 2026-10-20, before Panel B is scored)

User approval of 2026-10-20, with the user's additions. **No Panel B score and no TrueFake
image exists when this is written.** Data card: `docs/data_cards/truefake.md`.

- **Status:** secondary and **descriptive**. There is no decision rule and no hypothesis.
  It is scored **once, at the end, with Panel B**, on the frozen models.
- **Sample (frozen):** `reports/truefake/truefake_sample.csv` (sha256
  `a8954c2f63e81d95173cbec99e3f7c1ee2dd3d49527fac5ff7075de5e1ac8fd6`).
  - **4,000 ids, each in all 4 versions** (pre-share, Facebook, Telegram, Twitter): 16,000
    files, 7.985 GiB.
  - Seed 2026, ordered by sha256("2026:<id>") within each stratum.
  - Fakes: 250 per generator, spread evenly over its prompt categories (diffusion) or ψ
    settings (StyleGAN).
  - Reals: 1,000 FFHQ and 1,000 FORLAB.
- **Disjointness, before scoring:** the sampled reals (all four versions) are checked by
  sha256 and pHash (≤ 6 bits) against our training reals (COCO T-real, Open Images).
  Matches are dropped (no backfill) and counted per pool.
- **Models (4):** `S2-B` seed 0, **`S2-B+deg`** (the tuned winner, seed 0), `S2-D` seed 0,
  and the P2 `full_bank` S2 reference.
  - `S2-B` → `S2-B+deg` → `S2-D` separates how much social-media robustness comes from the
    degradation training and how much from the new reals.
  - Cost: 4 × 16,000 × 0.064 s (the slowest MEASURED S2 scoring rate) = **1.14 GPU-h**,
    within the 1.5 GPU-h cap.
- **Scoring:** the deployed protocol (centre crop 320, the model's final weights), every
  version scored independently.
- **Metric:**
  - **AUC per generator slice** (its fakes against a real pool), for each of pre-share,
    Facebook, Telegram and Twitter;
  - **Δ = AUC(platform) − AUC(pre-share)**, with a **paired bootstrap**: 2,000 resamples,
    seed 2026, the same ids in both versions, reals within their pool, fakes within their
    slice, 95% CI.
  - **Macros reported separately:** diffusion (lineages in our bank) and **StyleGAN 1–3
    (an unseen generator family)**.
- **Real pools, reported separately, both descriptive only:**
  - **FFHQ:** Flickr faces, the same platform as our training reals. AUC only, **never an
    FPR claim**.
  - **FORLAB:** camera originals **obtained from Flickr** ("over 33.000 Flickr images",
    Iuliani et al.), so also the same platform as our training reals. The achieved FPR at
    the k = 0 threshold is reported **descriptively only, never as a claim**.
  - *Correction of 2026-10-20:* the approved proposal treated FORLAB as the
    platform-disjoint camera pool; its source paper shows it is Flickr-sourced.
- **Caveats stated wherever a number appears:**
  - **both TrueFake real pools are Flickr**, the platform of our training reals, so
    **neither supports an FPR claim**; FORLAB's FPR is reported descriptively only
    (user, 2026-10-20);
  - StyleGAN fakes are faces, so **StyleGAN vs FFHQ is the content-matched comparison**,
    and any comparison against FORLAB also reflects content;
  - the platforms downscale (720/1200 px wide, ≤ 800 px tall), so the fixed 320 centre
    crop covers a different extent of the scene pre- and post-share;
  - FFHQ is non-commercial and FORLAB's terms are NOT-VERIFIED, so the numbers support no
    deployment claim (CLAUDE.md §6c).
- **Fetching:**
  - only in a **Kaggle CPU session**, after the user approves its hand-over, which states
    the expected download time;
  - member by member from `TrueFake.zip` by byte range, each member's CRC-32 checked
    against the archive index;
  - **if Google Drive refuses or rate-limits (403, 429, or a quota page), the session
    stops and reports; no workaround.**
- **Privacy:** **TrueFake images stay private.** They never go into the thesis, a report, a
  figure or a commit; aggregate numbers only. Any Kaggle dataset holding them is private.

### 4c. Few-shot threshold calibration — CPU only

For each Panel B view, k target-domain reals are drawn from that view's **calibration
pool** (§1.3), with 1,000 random draws per k.

| pool | size | k values |
|---|---|---|
| E5 | 500 | 0, 10, 50, 100 — genuine draws |
| E4 | 100 | 0, 10, 50, 100 — k = 100 is a single draw; the k = 50 draws overlap (a stated limitation: the share of draws understates the variability) |
| E3 | 50 | 0, 10, 25, 50 — k = 50 is a single draw; **k = 25 added**, descriptive |
| E7 | 100 | 0, 10, 50, 100 — descriptive |

A single-draw cell is reported with a bootstrap CI over the test reals, never as a share
of draws.

**Methods (decided 2026-10-08, change C):**
- **k = 0:** the deployed threshold, fixed on the validation reals (change B: the 491
  COCO validation reals plus the 400 held-out Open Images reals).
- **(Q) Conformal-style quantile.** The threshold is the j-th highest of the k reals,
  with j the largest value such that, by the binomial bound,
  `P(FPR > 7.5%) = P(Binomial(k, 0.075) ≤ j − 1) ≤ 10%`. Computed exactly this session
  (`scipy.stats.binom`):
  - **k = 50:** j = 1, **the highest of the 50 reals** (P = 0.0203). The 2nd-highest
    gives P = 0.1025, just over the bound.
  - **k = 100:** j = 4 (P = 0.0526).
  - **k = 10 and k = 25:** no j meets the bound; even j = 1 gives P = 0.4586 and
    0.1424. j = 1 is used and reported as not meeting it.

  (Q) controls FPR by construction, so **its finding is the TPR it achieves.**
- **(S) Logit shift.** Scores are shifted by median(k reals) − median(validation reals),
  then the k = 0 threshold is applied.

**Reported per view, method and k:**
- mean achieved FPR on the view's **test** reals;
- the share of draws with FPR ≤ 7.5%;
- TPR on test fakes;
- ECE of the shifted probabilities (15 bins).

**Success, per method:** at k = 50, achieved test FPR ≤ 7.5% in ≥ 80% of draws, **judged
on E4 and E5**. E3 and E7 are descriptive.

**FPR macros count each real pool once** (Change 2). The **decision rules use E4 and E5**;
E3 is descriptive under the source-statement principle, and its two generator slices
share one real pool. AUC and AP macros stay over the four unseen slices.

This is pure analysis of Panel B scores: **CPU, no GPU**.

---

## 5. Robustness curves on Panel B

Every Panel B image, real and fake alike, is perturbed and re-scored:

| op | levels |
|---|---|
| JPEG | quality 95, 75, 50, 30 |
| WebP | quality 90, 70, 50 |
| resize-and-restore | 0.75, 0.5 |
| Gaussian blur | σ 0.5, 1, 2 |
| clean | — |

That is 12 conditions plus clean.

- **Models:** the deployable S2, the P2 full_bank S2 (the baseline) and the P2 P-A S2.
- **Cost (derived):** about 9,750 images × 12 × 0.028 s (measured S2 scoring,
  `reports/kaggle_20261005/p2_s2_scores_*.json`) ≈ 0.91 GPU-h per model, so **2.7 GPU-h**
  for three models.
- Perturbations need pixels, so this runs on Kaggle.

**Note, 2026-10-22 (user decisions; before any Panel B score exists):**
- **Models:** `S2-D` **seed 0**, P2 `full_bank_seed0` and P2 `p_a_sd1_seed0`.
- **Rows:** Panel B's **8,655 test rows**. The 750 calibration reals are never test data
  (pre-registration §2); the luma companion is not perturbed.
- **Definitions** (`src/dbp/eval/perturb.py`). Each condition is applied to the whole
  image as distributed, then the deployed protocol's centre crop, identically for reals
  and fakes:
  - **JPEG:** PIL `quality=q`, chroma 4:2:0 (`subsampling=2`);
  - **WebP:** PIL lossy `quality=q`;
  - **resize-and-restore:** bicubic down to `round(side × f)`, then bicubic back to the
    original size;
  - **blur:** PIL `GaussianBlur(radius=σ)`.
- **Clean** is the clean-stage score of the same rows; it is not scored a second time.
- **Sessions:** the final scoring runs in two sessions of one notebook
  (`env/kaggle_run_p3_final.py`):
  1. **clean** — all 24 frozen models on Panel B + luma + the 891 threshold reals;
  2. **robust** — H5.

---

## 6. Optional: content-aware gate S2 + S3 (only if budget allows)

X2b found that the winner depends on E3's content type: S2 on texture, S3-LR/AEROBLADE on
structure.

- **Gate:** a pre-registered gate chooses or weights S2 vs S3 by a content score.
- **Content score:** a texture/structure measure **trained or fixed on the development
  panel only**. E3's content-type labels exist only there and in Panel B.
- **Test:** once, on Panel B.
- **Cost (derived):** S3 on Panel B is about 9,750 × 0.717 GPU-h / 1,000 ≈ **7.0 GPU-h**.
  0.717 is 13.35 GPU-h / 18,614 rows, measured in Session C (`IMPLEMENTATION_LOG.md`
  P2-D step 2).

---

## 7. Budget and schedule

### 7.1 GPU-hours (device-hours, derived)

Per S2 run:
- **Training:** `N_max` = 6,000 steps × 0.86–1.07 s/step = 1.43–1.78 h. The step times
  are measured: `measured_step_seconds` in `reports/kaggle_20261005/p2_s2_*.json`.
- **Held-out evaluation:** every 500 steps (12 evaluations × about 2,500 images ×
  0.028 s ≈ 0.23 h).
- **Scoring:** development and Panel B (about 22,500 images × 0.028 s ≈ 0.18 h).
- **Total:** **plan at 2.2 GPU-h per run**.

| block | runs / items | GPU-h | can run on CPU? |
|---|---|---|---|
| Panel B draw and freeze | — | 0 | **yes** (Kaggle CPU session) |
| 4a new reals through the bank | 4,000 Open Images reals | 4.7 | no |
| 2 LOLO | 8 × 2.2 | 17.6 | no |
| `N*` from the LOLO curves | — | 0 | **yes** (local) |
| 3 bank-size curve | 7 × 2.2 | 15.4 | no |
| 3 `BANK-1L` (descriptive) | 1 × 2.2 | 2.2 | no |
| 4a/4b deployable model and ablations | baseline × 2 seeds, +degradation, +new reals, +both × 2 seeds = 6 runs | 13.2 + 1.7 (25% on the 3 degradation runs) = 14.9 | no |
| 4c few-shot calibration | — | 0 | **yes** (local) |
| 5 robustness | 3 models | 2.7 | no |
| all bootstrap analyses, tables, figures | — | 0 | **yes** (local) |
| **core total** | | **57.5** | |
| 6 optional gate (S3 on Panel B) | | 7.0 | no |
| P2 leftover: X1 format control | | about 0.76 wall | no (or CPU, slow) |

### 7.2 Quota

- **Billing is still unknown (I-042).** Session D charged 12 h for a 3.50 h session, with
  editor GPU time possibly mixed in.
- **Two bounds:**
  - **Per GPU-hour:** quota ≈ device-hours, **about 58 h core**.
  - **Per session-hour:** both T4s run in parallel, so quota ≈ device-hours / 2, about
    29 h.
- **The first P3 GPU session closes I-042.** Read the meter with the editor on CPU
  before, commit only, and read it again after.

### 7.3 Proposed order (≤ 30 GPU-h per week, `muhammadadeelshah` only)

| week | CPU (no quota) | GPU sessions (both T4s via the queue) | device-h |
|---|---|---|---|
| 0 | approve the diff, then **freeze the pre-registration**; approve the image hosts; Phase B code and dry-runs; **Panel B draw and freeze** (Kaggle CPU) | X1 format control (0.76 h), meter reading, closes I-042 | ≈ 1 |
| 1 | — | 4a reconstruction (4.7); LOLO folds 1–6 (13.2) | ≈ 18 |
| 2 | `N*` from the LOLO curves | LOLO folds 7–8 (4.4); bank-size curve (15.4) + `BANK-1L` (2.2); deployable baseline × 2 (4.4) | ≈ 26 |
| 3 | 4c few-shot; LOLO and bank-size analysis on dev | ablations (10.5); robustness (2.7); optional gate (7.0) | ≈ 20 |
| 4 | **Panel B single-shot analysis**; P3 report | — | 0 |

**Why this order:**
- Panel B is frozen first, so nothing trained can be tuned on it.
- LOLO precedes everything that needs `N*`.
- The new reals are reconstructed before the runs that use them.
- Every session is resumable (the P2 queue), so a short week moves work to the next one.

**Cut ladder if quota is short** (in order):
1. Drop the optional gate (−7.0).
2. Run one seed for the ablations instead of two (−4.4).
3. Run the bank-size curve's far-first order at k = 2 and 6 only (−4.4).

LOLO and Panel B are never cut.

---

## 8. Runbook compliance (CLAUDE.md §12)

Every GPU stage gets:
- a percent-format runbook under `env/`, made of thin cells calling committed scripts;
- `env/kaggle_bootstrap.sh` as the only setup;
- a pre-flight with paths, hashes, imports and a 2-image or 2-step smoke per model and
  GPU;
- the fail-fast queue (`p2_run_queue.py`) for parallel runs;
- a report cell that cannot fail;
- `runbook_dryrun.py --ref HEAD` before hand-over;
- the code zip built from git's bytes;
- a stated wall time, quota and outputs.

Every Kaggle failure caused by our code gets a regression test.

## 9. Additions that need your approval (§9b)

- **New dataset and host (§4a):** Open Images V7 only — its image host, data card,
  citation and attribution file.
- **New code:**
  - a Panel B draw and freeze script;
  - P3 run configs (LOLO folds, bank-size points) in a new `configs/p3_runs.yaml`;
  - the synchronised degradation module;
  - fixed-step training with held-out-lineage evaluation in `p2_train_s2.py`;
  - the nested `N*` selector;
  - the few-shot calibration analysis;
  - a robustness perturbation scorer;
  - (optional) the content gate.
- **New runbooks:** Panel B (CPU), reconstruction of new reals, the LOLO / bank-size /
  final queue, robustness.

## 10. Decisions on the P3-A review (2026-10-08)

| question | decision |
|---|---|
| E8 in Panel B | **Yes**, as a secondary fakes-only view: TPR at the k = 0 threshold, descriptive, not in any macro |
| real sources | **Open Images V7 only, 4,000** (3,600 train + 400 held out), CC BY 2.0 per image, attribution file. **Unsplash Lite dropped** (it is E5's real source); **RAISE: no** |
| seeds | two seeds for **S2-B and S2-D** only; LOLO and bank-size single seed |
| thresholds | set by the user; written into `docs/P3_PREREGISTRATION.md` |
| MNW | Panel B frozen **without** MNW. MNW becomes **Panel C**, once-only, under the same frozen analysis |
| added | H1-c; the union k = 0 threshold (491 COCO + 400 Open Images); the k-shot methods with genuine draws (E5 pool 500); FPR macros over real pools; the E3 small-sample note |

**Budget:** **57.5 device-h core**. That is 55.3, plus `BANK-1L` (2.2), under the
user's 60 limit. Open Images 4,000 is the same reconstruction count as before. The 400
extra E5 calibration reals add about 400 × 0.028 s per scored model, which is
negligible.

**Resolved 2026-10-08:**
- **E3:** option (b), the source-statement principle.
- **H2:** also reported without E3.
- **Dedup:** against the full E3/E4/E5 real sets.
- **BANK-k:** SD1 complete from k = 2, with `BANK-1L` added.

**After FROZEN:** approval of the Open Images image host, once it has been read from the
metadata.
