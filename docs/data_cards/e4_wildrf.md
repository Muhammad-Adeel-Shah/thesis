# Data card — E4, WildRF (test split only)

| field | value |
|---|---|
| set id | `e4_wildrf_test` |
| role | evaluation only — **never** trained on, not even its own `train/` split |
| source repo | `barcavia/RealTime-DeepfakeDetection-in-the-RealWorld` (GitHub) |
| paper | *Real-Time Deepfake Detection in the Real-World* |
| project page | `https://vision.huji.ac.il/ladeda/` |
| distribution | Google Drive file `1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn`, linked from the repo README |
| licence | **`NOASSERTION`** (GitHub API; "Other") → **research use only** |
| built by | `scripts/p1_build_eval.py`, per `env/kaggle_run_p1_eval.md` |

Repository metadata, licence and the Drive file id were read in the P1 session on
2026-09-20 via `api.github.com` and `raw.githubusercontent.com`, both allow-listed by
CLAUDE.md §6a. The repository's `pushed_at` was `2024-11-30T11:14:14Z`.

---

## 1. Why this set

WildRF is images **scraped from social platforms** — Reddit, Twitter, Facebook — rather
than generated under laboratory conditions. Everything in it has been through a
platform's own re-encoding pipeline, so it is the closest thing in the P1 corpus to
deployment conditions, and it is where a detector that has learned a laboratory artefact
will fail.

## 2. Structure, as published

```
WildRF
├── train/{0_real,1_fake}
├── val/{0_real,1_fake}
└── test/
    ├── reddit/{0_real,1_fake}
    ├── twitter/{0_real,1_fake}
    └── facebook/{0_real,1_fake}
```

**Only `test/` is indexed.** Two reasons, both binding:

1. The user's instruction for P1 was "E4 (TEST split only; never train on WildRF)".
2. WildRF's own protocol is to train on one platform and evaluate on unseen ones. Using
   any part of it for training would break that protocol and make a cross-generator
   claim circular.

`--field platform=1` records `reddit` / `twitter` / `facebook` per image, so every
number can be broken down the way the benchmark is defined.

Counts per platform and label: **PENDING-KAGGLE** — the archive is downloaded on Kaggle
only (CLAUDE.md §6a condition (a)). Fill from `reports/p1_eval_e4_wildrf_test_summary.json`
after the run.

## 3. Compression history (CLAUDE.md §6b)

| label | n | JPEG | never compressed | formats | quality min/median/max | chroma subsampling |
|---|---|---|---|---|---|---|
| real | PENDING-KAGGLE | | | | | |
| fake | PENDING-KAGGLE | | | | | |

**Expected verdict: not confounded.** Everything here has been through a social
platform's re-encoder, so both labels should be overwhelmingly JPEG. That expectation is
*not* a substitute for the measurement: `p1_build_eval.py` is run with `--jpeg-match`
anyway, and it will report "not confounded" and build nothing if that holds. The table
is required for every set, including the ones that turn out clean — a clean verdict is
evidence, and only the measurement can produce it.

If it does turn out confounded, the matched view is built by the same rule as E3 and
becomes the primary view for this set.

## 4. Known confounds

- **Platform is a confound, by design.** Each platform has its own re-encoding pipeline,
  and the real/fake mix differs per platform. Always report per platform; a pooled
  number can be moved by the platform mix alone.
- **Provenance is manual.** The authors collected images by keyword and hashtag, so the
  labels are curatorial rather than derived from generation logs. Some "fake" images may
  be heavily edited photographs rather than fully generated, and vice versa.
- **No generator labels.** Unlike E3, the generating model is unknown, so per-generator
  breakdowns are impossible.

## 5. Leakage

Checked by perceptual hash against `p1_reals_manifest_t_real.csv` and
`p1_reals_manifest_t_ctrl.csv` at a 6-bit threshold. Result in `reports/p1_leakage.json`
under `e4_wildrf_test`. **PENDING-KAGGLE.**

Low prior risk — social-media scrapes and COCO/DIV2K are different provenances — but
COCO images do circulate on social platforms, so the check is run rather than assumed.

## 6. Licence and what may be claimed

`NOASSERTION` on GitHub means the authors shipped a licence file the API could not
identify, so **no permissive grant can be assumed**. Treated as **research use only**:

- results on E4 may be reported in the thesis;
- E4 must not appear in any commercial or deployment claim;
- the images are not redistributed. Only manifests, hashes and summaries are committed;
  the image data stays on Kaggle (CLAUDE.md §2).

Recorded in CLAUDE.md §6c alongside the other corpus constraints.

## 7. Retrieval

`gdown==5.2.0`, installed **inside the Kaggle session only**, pinned and logged, and
deliberately not added to `requirements/` (CLAUDE.md §6a condition (c)). The download
happens on Kaggle, never to the local machine.

## Measured on ingestion, 2026-09-28

`reports/kaggle_20260928/p1_eval_e4_wildrf_test_summary.json`. **2,503 rows**, test split
only: 1,251 real / 1,252 fake, across reddit 1,500, twitter 683, facebook 320.
**Leakage CLEAN** against both training manifests.

### The most confounded set in the suite

`CONFOUNDED; never-compressed share: fake 39.5% vs real 2.2%` — 495 of the fakes are PNG
against 28 of the reals. Parameter separability (D-042):

| scope | parameter | lift |
|---|---|---|
| pooled | subsampling | **+19.18** |
| pooled | jpeg_quality | **+25.41** |
| `platform=reddit` | subsampling | **+22.59** |
| `platform=reddit` | jpeg_quality | **+22.79** |

reddit carries it: 480 of its 750 fakes are PNG against 28 of 1,213 reals.

### The matched view helps, and does not finish the job

495 fakes re-encoded from the 1,221 usable real JPEGs. Subsampling drops to **+1.24**,
which is clean. **Quality remains separable at +11.39**, and that residual is structural,
from two sources the §6b rule cannot touch:

1. **28 reals are PNG and are never re-encoded** — reals keep their genuine history by
   design — so `quality = None` occurs only in the real class.
2. **757 fakes were already JPEG**, so they keep their own settings; q85, q92, q95 and
   q99 all skew fake.

**Every E4 number carries this caveat**, and the per-platform breakdown must state that
reddit is the affected platform. A detector can score on E4 without reading a generative
artefact, and the matched view narrows that but does not close it.

### Two greyscale reals (I-039)

2 of the 1,251 reals are greyscale JPEGs, whose chroma subsampling reads as `-1`. They
are excluded from the matched view's *reference* — recorded as
`reference_excluded_unmappable_subsampling: 2` — and stay in the set. Drawing one of them
aborted an earlier build outright.

## 🔴 Red flag on the matched view (D-044)

**E4's matched view is still separable on quality: +11.39.** That is well above the
5-point threshold, and it means a detector can reach roughly 61% on this set — against a
50% baseline — by reading the JPEG quality alone.

**Every E4 table carries this flag**, including the matched-view tables. The wording to
use, or something equivalent in substance:

> *E4 (matched): quality separability +11.39 — up to ~61% of this set is reachable
> without a generative artefact. Numbers are upper bounds.*

The red flag is not removable by better matching: the residual comes from the 28 reals
that are PNG and are never re-encoded, and from the 757 fakes that were already JPEG and
keep their own settings. E4 remains useful as an in-the-wild breadth test; it cannot
carry a precise claim.
## P2 evaluation panel (D-048)

**1,500 rows from the matched view**, stratified by platform x label. The +11.39 quality red flag travels with the panel: `p2_build_panel.py` writes it into the panel summary and `p2_analyze.py` prints it in `metrics.csv`'s `trivial_baseline` column.
