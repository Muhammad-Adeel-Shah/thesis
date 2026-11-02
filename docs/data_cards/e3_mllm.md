# Data card — E3, MLLM-Generated Image Detection Dataset

| field | value |
|---|---|
| set ids | `e3_raw`, `e3_raw_jpeg_matched` (**primary**), `e3_preprocessing` |
| role | evaluation only — never trained on |
| source | `zr-zhang/MLLM-Generated-Image-Detection-Dataset` (Hugging Face **dataset** repo) |
| pinned revision | `1498eead24292a9a2d134476b2c559193b68b9de` |
| licence | `apache-2.0` (Hub tag) |
| gated | no |
| built by | `scripts/p1_build_eval.py`, per `env/kaggle_run_p1_eval.md` |

Every value below was produced by a command in the P1-B3/B7 session (2026-09-20):
`HfApi.dataset_info(...)` plus a full file listing. Nothing is quoted from the prompt.

---

## 1. Correction to the P1 prompt

The prompt described E3 as **"2,904 imgs, 320×320 BMP, already resized"**. That is wrong
in every particular, and the measured facts are used instead:

| prompt said | measured |
|---|---|
| 2,904 images | **4,356 files** = **2,178 unique images × 2 views** |
| BMP | **no BMP at all** — `.png`, `.jpg`, `.jpeg` |
| 320×320 | native and variable in `Raw`; variable and non-square in `Preprocessing` |
| "already resized" | true of `Preprocessing` only; `Raw` is native resolution |

## 2. Composition

Perfectly balanced, three labels × three perceptual regimes, identical in both views:

| regime | real | GPT-Image2-fake | Nano-Banana2-fake | total |
|---|---|---|---|---|
| Texture Images | 216 | 216 | 216 | 648 |
| Structure Images | 313 | 313 | 313 | 939 |
| Hybrid Images | 197 | 197 | 197 | 591 |
| **total** | **726** | **726** | **726** | **2,178** |

Two generators, both closed MLLM image models, so E3 is a genuine cross-generator test
against families absent from the decoder bank.

## 3. Compression history (CLAUDE.md §6b)

### `Raw` — as distributed

| label | n | JPEG | never compressed | formats | quality min/median/max | chroma subsampling |
|---|---|---|---|---|---|---|
| real | 726 | 726 | **0** | `.jpg` 414, `.jpeg` 312 | PENDING-KAGGLE | PENDING-KAGGLE |
| fake | 1,452 | 91 | **1,361** | `.png` 1,361, `.jpeg` 84, `.jpg` 7 | PENDING-KAGGLE | PENDING-KAGGLE |

Per generator: GPT-Image2-fake is 726 PNG and nothing else; Nano-Banana2-fake is 635 PNG
+ 84 `.jpeg` + 7 `.jpg`.

The quality and subsampling cells are `PENDING-KAGGLE` because filling them requires
reading the quantisation tables of 2,178 files, and the files are downloaded on Kaggle
only (CLAUDE.md §6a condition (a)). `p1_build_eval.py` writes them into
`reports/p1_eval_e3_raw_summary.json`; copy them here after that run.

**Verdict: CONFOUNDED.** Never-compressed share: fake 93.7 %, real 0.0 %. Every real is a
JPEG; 93.7 % of fakes have never been compressed. This is the "Fake or JPEG?" confound
[R7] in its strongest form — a classifier reading only the compression history would
score far above chance without looking at a single generator artefact.

Loading pixels rather than files removes the *format* leak. It does **not** remove the
*compression history* leak: a never-compressed PNG has no 8×8 DCT quantisation lattice,
and that absence is visible in the pixels.

### `Raw-JPEG-matched` — the primary view

Built by `p1_build_eval.py --jpeg-match`:

- the **1,361 never-compressed fakes** are re-encoded to JPEG;
- quality and chroma subsampling are drawn **per image** from E3's own 726 real JPEGs,
  repeats included, so the real distribution's shape is reproduced rather than smoothed;
- the seed is `blake2b("e3_raw:" + image_id, 8 bytes)`, so the view is reproducible from
  the manifest alone and does not depend on directory order;
- **native resolution, no resize** (D-001);
- **reals are never re-encoded**, and neither are the 91 fakes that were already JPEG.
  Replacing a genuine compression history with a synthetic one is the same error in the
  other direction;
- originals are untouched — the view is new files plus a manifest.

Per-image quality, subsampling and the seed rule are recorded in
`reports/p1_eval_e3_raw_jpeg_matched_manifest.csv`.

**All claims are made on this view.** The other two are reported for comparison.

### `Preprocessing` — secondary

All 2,178 files are `.jpg`, so compression is *not* confounded with the label here. But
the images are downscaled as well as recompressed, which violates D-001's reasoning:
resampling destroys the high-frequency generator traces the method reads. Usable only as
a confound-controlled cross-check, and always labelled as resized.

## 4. Residual confounds, after matching

Honest limits of the matched view:

- **Re-encoding is not the same as never having been compressed differently.** A matched
  fake has been through exactly one JPEG cycle; a real may have been through several,
  with resizing in between. The matched view equalises the *first-order* cue, not the
  full history.
- **Resolution.** `Raw` sizes are native and differ between generators. Reported per
  label in the manifest; P2 must check whether resolution alone separates the classes.
- **Which libjpeg built the view.** If B2 (I-010) returns `PIXELS-DIFFER`, the matched
  view is a property of the machine that built it. It is built on Kaggle only, and the
  build environment is recorded in `reports/p1_jpeg_equivalence_kaggle.json`.

## 5. Leakage

Checked by perceptual hash against `p1_reals_manifest_t_real.csv` and
`p1_reals_manifest_t_ctrl.csv` at a 6-bit Hamming threshold. Result in
`reports/p1_leakage.json` under `e3_raw`. **PENDING-KAGGLE.** Any `LEAKAGE` entry must be
resolved before a number is reported on this set.

## 6. How to report it

- Primary: **`e3_raw_jpeg_matched`**, split by source format (`png` vs `jpeg`), so a
  reader can see the matching did not merely hide the effect.
- Also report `e3_raw`, labelled "compression-confounded, upper bound".
- Also report `e3_preprocessing`, labelled "resized".
- Break every number down by regime (Texture / Structure / Hybrid) and by generator
  (GPT-Image2 / Nano-Banana2); the balanced design makes that free.

## 7. Licence

`apache-2.0`, one of only two comparatively permissive sources in the P1 corpus (the
other is COCO). No non-commercial restriction. See CLAUDE.md §6c for the sets that do
constrain deployment claims.

## Measured on ingestion, 2026-09-28

`reports/kaggle_20260928/p1_eval_e3_raw_summary.json` and `..._e3_preprocessing_...`.
**2,178 rows per view**: 726 real, 726 GPT-Image2, 726 Nano-Banana2. **Leakage CLEAN**
against both training manifests, in both views.

### Raw: confounded, exactly as expected

`CONFOUNDED; never-compressed share: fake 93.7% vs real 0.4%`. The parameter check
(D-042) shows how badly:

| scope | parameter | lift |
|---|---|---|
| pooled | subsampling | **+28.97** |
| pooled | jpeg_quality | **+29.75** |
| `source_label=GPT-Image2-fake` | subsampling | **+49.79** |
| `source_label=GPT-Image2-fake` | jpeg_quality | **+49.79** |
| `source_label=Nano-Banana2-fake` | subsampling | +43.46 |
| `source_label=Nano-Banana2-fake` | jpeg_quality | +44.63 |
| `subset=Structure Images` | both | +45.78 |
| `subset=Texture Images` | both | +36.96 |
| `subset=Hybrid Images` | both | +27.14 / +28.21 |

On the GPT-Image2 subset every fake is PNG and every real is JPEG, so `not-jpeg → fake`
is a perfect classifier. **No number from the raw view means anything on its own.**

### Raw-JPEG-matched: the view works, with one residual

1,360 never-compressed fakes re-encoded from the 723 real JPEGs of this same set.
Pooled lift collapses to **+0.14** (subsampling) and **+0.23** (quality) — the §6b rule
doing its job.

**One residual survives and must be reported:** `source_label=Nano-Banana2-fake`,
jpeg_quality, **+6.47**. The cause is structural, not a bug: 92 of Nano-Banana2's fakes
were *already* JPEG, so the view leaves them untouched and they keep their own quality
distribution — 139 fakes at q100 against 75 reals. Any Nano-Banana2 number carries this.

### Preprocessing: clean on parameters, compromised by the resize

Lift **+0.00** on both parameters — the whole view went through one pipeline. It remains
**secondary**, because every image was downscaled and no re-encoding undoes that.

## P2 evaluation panel (D-048)

**1,500 rows from the matched view**, allocated proportionally across
generator x content type with a floor of one per non-empty stratum, so the
per-content-type report D-044 requires is possible. Never drawn from the raw view.
The Nano-Banana2 residual of +6.47 travels with the panel and is printed in
`metrics.csv`'s `trivial_baseline` column.
