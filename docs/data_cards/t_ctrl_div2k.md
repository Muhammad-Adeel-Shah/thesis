# Data card — T-ctrl, DIV2K train HR

| field | value |
|---|---|
| set id | `t_ctrl` |
| role | **training control, diagnostic-only** — high-quality reals that have never been JPEG-compressed |
| source | `https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip` |
| archive size | **3,530,603,713 bytes** (HTTP `Content-Length`, checked 2026-09-20) |
| archive last modified | `Tue, 14 Feb 2017 14:27:11 GMT` |
| licence | **academic research only** (DIV2K / NTIRE terms) |
| target size | 1,600 rows = **800 source images x 2 non-overlapping crops** |
| working crop | 384 × 384, centre-cropped, **never resized** |
| built by | `scripts/p1_build_reals.py --set t_ctrl`, per `env/kaggle_run_p1_build.md` |

The `Content-Length` and `Last-Modified` above came from a HEAD request in the P1
session. `data.vision.ee.ethz.ch` is allow-listed by CLAUDE.md §6a, and the download
happens **inside the Kaggle session only** — condition (a). It goes to `/tmp`, not to
`/kaggle/working`, because 3.5 GB of extracted PNGs are not worth persisting.

---

## 1. Why a second real corpus

T-real (COCO) is entirely JPEG. If it were the only real corpus, every reconstruction
pair in the project would be built from an image that had already been through a JPEG
quantisation lattice, and there would be no way to tell a decoder's fingerprint apart
from an interaction between the decoder and that lattice.

DIV2K is the control: **2K-resolution PNG, never JPEG-compressed, professionally
captured**. It answers a question T-real cannot:

> Does a decoder's residual signature survive when the input carries no prior
> compression at all?

It also stresses the crop policy from the other end — DIV2K images are around 2040 px on
the short side, so every one of them clears the 384 px threshold, where COCO loses a
substantial fraction.

## 2. Selection

Same rule as T-real: all `.png` files under `DIV2K_train_HR/`, sorted by POSIX relative
path, taken in order until the limit is reached, cropped to 384 × 384 and written as PNG.
Every crop of a source is kept or none is, so a source never contributes a partial group.

DIV2K train HR contains **800 images** at 2K resolution. That was verified without
downloading the archive: a range request for the last 64 KiB returned the zip's
end-of-central-directory record, which reports **801 entries** — 800 PNGs plus the
`DIV2K_train_HR/` directory entry.

**1,600 rows therefore come from 800 photographs**, via `--crops-per-image 2` (decision
on I-021). The two crops are **non-overlapping**: they are laid out along the image's
longer axis and the pair is centred, so both come from the middle of the frame rather
than its edges. At 2K resolution a 384 px crop uses under a fifth of the long side, so
there is ample room between them.

### The condition that makes this safe

Two crops of one photograph are **not** two independent samples. They share lighting,
sensor, scene statistics and post-processing. So every row carries **`source_image_id`**,
and:

> **Any split — train/val/test, leave-one-lineage-out, cross-validation — must group by
> `source_image_id`.** Two crops of one photograph on opposite sides of a boundary is
> leakage, however disjoint the pixels are.

The build logs this as a warning on every run where `crops_per_image > 1`, and records
`split_grouping_column: source_image_id` plus `distinct_source_images` in the summary
JSON, so the constraint travels with the data rather than living only here.

Deduplication drops **whole source groups**, never individual crops, so a
`source_image_id` in the manifest always identifies a complete set of crops.

**T-ctrl remains diagnostic-only.** It is the control that shows whether a model leans on
JPEG structure — lossless reals and their reconstructions, with no JPEG anywhere in the
path — and it is reported alongside T-real rather than pooled with it. Its effective
sample size is 800 photographs, not 1,600 images, and any confidence interval should say
so. If a genuinely larger control set is needed, DIV2K's validation split or Flickr2K are
the honest options — a P2 decision.

Actual kept count: **PENDING-KAGGLE**, from `reports/p1_reals_manifest_t_ctrl.json`.

## 3. Compression history (CLAUDE.md §6b)

| label | n | JPEG | never compressed | formats | quality min/median/max | chroma subsampling |
|---|---|---|---|---|---|---|
| real | PENDING-KAGGLE | 0 (expected) | all (expected) | PNG | n/a | not-jpeg |

Fill from the summary JSON after the build. The expectation is that this set is 100 %
PNG with no JPEG history at all — that is the entire point of it — but the table is
measured rather than assumed, because "all PNG" is a claim like any other.

**T-ctrl and T-real must never be pooled into one training set without saying so.** They
differ systematically in compression history, and a model trained on the union would
have a free cue that has nothing to do with decoders. They are two sets, reported
separately, and the difference between them is a result rather than a nuisance.

## 4. Deduplication

Perceptual hash at a 6-bit threshold, near-duplicates dropped and listed. DIV2K is
curated and near-duplicates are unlikely; the check is run anyway, and a non-empty result
would be worth investigating rather than silently discarding.

## 5. Provenance recorded per image

Same schema as T-real, plus the grouping columns: stable `image_id`
(`<source_image_id>_c<k>` when there is more than one crop), **`source_image_id`**,
`crop_index`, `crop_top`, `crop_left`, the source relative path and dimensions, source
SHA-256, crop SHA-256, source compression metadata and the crop's perceptual hash.

`crop_top` / `crop_left` make the crop reproducible from the original file alone.

## 6. Leakage

These perceptual hashes are the second training side of every evaluation set's leakage
check. DIV2K is a standard super-resolution benchmark and its images do appear in other
datasets, so the check is not a formality.

## 7. Licence and what may be claimed

DIV2K is released for **academic research purposes only**. Consequences:

- results on T-ctrl may be reported in the thesis;
- nothing trained on T-ctrl may support a commercial or deployment claim;
- the images are not redistributed — only manifests and summaries are committed, and the
  crops stay in the Kaggle output dataset (CLAUDE.md §2).

Recorded in CLAUDE.md §6c with the other corpus constraints. Since T-ctrl is *training*
data rather than an evaluation set, this is the constraint that bites hardest: any model
trained on it inherits the restriction.

## 8. Greyscale exclusion (D-039)

Measured 2026-09-26 (`reports/p1_greyscale_rescan_t_ctrl.json`): **6 of 1,600 rows are
greyscale — 3 source photographs (`0230`, `0769`, `0800`), both crops of each.** T-ctrl
is now **1,594 usable rows from 797 source images**.

All six are `rgb_equal` with a channel spread of exactly 0: greyscale pictures stored as
three identical channels in a lossless PNG. No format check could find them, and DIV2K
being PNG means there is no subsampling label to hint at it either — the pixels are the
only evidence.

Both crops of each source were excluded together, so no source contributes a partial
group and the `source_image_id` grouping rule is unaffected.

`unmappable_but_colour: 0`, as it must be: a PNG has no chroma subsampling to be
unreadable, and reading its `not-jpeg` label as "unreadable" is exactly the defect
recorded as I-037.

**Downstream:** 6 × 12 distinct members = **72 of the 49,200 pair rows** are filtered by
the join on `image_id`. With T-real's 83, the corpus is **4,011 usable reals** and
**48,132 usable pair rows**. Nothing was deleted or rebuilt.

