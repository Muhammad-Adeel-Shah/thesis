# Data card — T-real, COCO 2017

| field | value |
|---|---|
| set id | `t_real` |
| role | **training** — the real images every reconstruction pair is built from |
| source | Kaggle dataset `awsaf49/coco-2017-dataset`, **`train2017` split only** |
| licence | CC BY-SA 4.0 (COCO annotations); images are Flickr-sourced under their own terms |
| target size | 2,500 images (1 crop per source image) |
| working crop | 384 × 384, centre-cropped, **never resized** |
| built by | `scripts/p1_build_reals.py --set t_real`, per `env/kaggle_run_p1_build.md` |

The dataset page returned HTTP 200 when checked on 2026-09-20. The mirror is read
**in place** from `/kaggle/input`; it is never copied, because it is tens of gigabytes
and the `/kaggle/working` quota is about 20 GB.

---

## 1. Why COCO

Three reasons, in order of weight:

1. **Baseline parity.** DDA and SEF, the methods this work is measured against, build
   their training data from COCO. Using anything else would make the comparison a
   comparison of corpora.
2. **Subject diversity.** Everyday scenes and objects, which is what the deployment
   target looks like.
3. **It is the prompt source for B6.** The SD1.5 generation set is prompted with COCO
   captions, so the real and fake training images describe the same distribution of
   scenes. A detector trained against fakes of a *different* subject distribution learns
   the subject, not the generator.

## 2. `train2017` only — and why that is not a detail (I-024)

**E7 DDA-COCO's real images are the MSCOCO *validation* set.** If T-real were allowed to
select from `val2017`, E7's evaluation reals would end up inside our training corpus, and
the primary head-to-head number — the one carrying the central claim against SEF — would
be measured on images the model had already seen.

Nothing about that failure is loud. No script errors, no assertion trips; the AUC simply
comes out too high. So it is defended twice:

1. `p1_build_reals.py --root` is pointed at the `train2017` images directory, not at the
   dataset root. The build runbook locates it explicitly and asserts no `val2017` path is
   reachable from it.
2. The perceptual-hash leakage check runs E7 against this manifest anyway, at a 6-bit
   threshold, into `reports/p1_leakage.json`.

If the check ever reports `LEAKAGE`, the fix is to correct the *training* selection and
rebuild — never to drop the offending evaluation images, which would silently redefine
the benchmark.

B6's prompts already come from `captions_train2017.json`, so the generation set is
consistent with this restriction.

## 3. Selection

Deterministic and reproducible:

- every file under the root with an image suffix, **sorted by POSIX relative path**, so
  the `--limit` selects the same 2,500 images on any machine;
- an image smaller than 384 px in either dimension is **skipped**, not upscaled (D-001).
  COCO has many such images; the skip count is recorded in
  `reports/p1_reals_manifest_t_real.json`;
- the kept images are centre-cropped to 384 × 384 and written as PNG. PNG because the
  crop is the input to every decoder and must not carry a second compression generation.

Actual kept count: **PENDING-KAGGLE**. If it falls short of 2,500 the corpus really is
smaller and every later count says so — the size filter is not loosened to make the
number come out.

## 4. Compression history (CLAUDE.md §6b)

COCO is **entirely JPEG**, and that matters twice over.

Measured on Kaggle 2026-09-26 from `reports/p1_reals_manifest_t_real.json`:

| label | n | JPEG | never compressed | formats | quality min/median/max | chroma subsampling |
|---|---|---|---|---|---|---|
| real, as built | 2,500 | 2,500 | 0 | JPEG 2,500 | **80 / 96 / 96** | 4:4:4 × 2,489, `unknown(-1)` × 11 |
| real, **usable** (after §9) | **2,417** | 2,417 | 0 | JPEG 2,417 | **90 / 96 / 96** | **4:4:4 × 2,417** |

The manifest records the **source** file's format, estimated quality and subsampling per
image, before cropping. Three things to read out of those rows:

- The quality distribution is narrow and high, and after the greyscale exclusion it is
  narrower still: **no usable image is below q90**. This is the distribution
  `gen_matched` draws from, so the matched generations are q90–96. It is not a general
  "web JPEG" distribution and no claim should treat it as one.
- The shift from 80 to 90 is not a systematic effect. **Exactly one image in the whole
  corpus was q80, and it happened to be greyscale.** The excluded 83 are otherwise
  ordinary: 67 at q96, 15 at q90, 1 at q80.
- Every usable image is **4:4:4**, so the matched view varies only in quality. The 11
  `unknown(-1)` rows were greyscale JPEGs (§9).

Two consequences:

- **Against T-ctrl (DIV2K, all PNG).** The two corpora differ in compression history, so
  they are kept as separate sets and are never pooled without saying so. Pooling them
  silently would build the confound into the training data itself.
- **Against the B6 generations — now fixed, not merely flagged (I-023).** The SD1.5
  outputs are written as PNG and have never been JPEG-compressed, while these reals all
  have. `scripts/p1_match_generations.py` therefore writes a second variant of every
  generation, encoded with a `(quality, chroma subsampling)` pair drawn **jointly** from
  one named real in *this* manifest, seeded by the generation's id. P2 trains on that
  matched variant by default and reports raw-vs-matched as an ablation. The reals are
  never re-encoded, and the PNG variant is kept.
  Per-image parameters: `reports/p1_gen_matched_manifest.csv`.

## 5. Deduplication

Near-duplicates are found by perceptual hash at a 6-bit Hamming threshold, **dropped**,
and listed in `near_duplicate_groups` in the summary JSON. The first id in each group is
kept.

Duplicate reals would give one picture several votes in training and would leak across
any split. Dropping them is safe here in a way it would not be for an evaluation set:
this is training data we construct, not a benchmark we must use as distributed.

## 6. Provenance recorded per image

`reports/p1_reals_manifest_t_real.csv` carries, per row: `image_id` (blake2b of the
source relative path, so it is stable across machines), the source relative path, source
width and height, source SHA-256, the written crop's SHA-256, the source's compression
metadata, and the crop's perceptual hash.

The SHA-256 pair means a later run can prove both that the source has not changed and
that the crop is the one the numbers were computed on.

## 7. Leakage

The perceptual hashes in this manifest are the training side of every evaluation set's
leakage check (`reports/p1_leakage.json`). Two of those checks are load-bearing rather
than precautionary:

- **E7 DDA-COCO** — same source corpus, different split. See §2.
- **E4 WildRF** — COCO images do circulate on social platforms.

Every row also carries `source_image_id`. T-real takes one crop per source, so it equals
`image_id` here, but any split code must group by it regardless: the same column carries
real weight in T-ctrl, and a split that groups by it works correctly for both.

## 8. Licence and what may be claimed

COCO is CC BY-SA 4.0 for the annotations; the images are Flickr-sourced under their own
terms. It is one of only two comparatively permissive sources in the P1 corpus (the other
is E3, apache-2.0).

**No image data is committed.** Only the manifest and the summary JSON are tracked; the
crops live in the Kaggle output dataset (CLAUDE.md §2).

## 9. Greyscale exclusion (D-039)

**11 of the 2,500 images are greyscale**, found because their chroma subsampling reads
as `unknown(-1)`: Pillow reports `-1` when there are no chroma channels to describe.
They are excluded from use, for two independent reasons.

**They cannot define a chroma reference.** The matched view (CLAUDE.md §6b) copies a
`(quality, chroma subsampling)` pair from one named real onto a generation. A greyscale
real has no such pair. Thirteen of 2,488 generations failed on exactly this, and the
matcher was right to refuse rather than substitute a plausible `4:2:0`.

**They are a label shortcut, which is the more serious reason.** "Has no colour" is
trivially learnable and says nothing about which decoder produced an image. A corpus
whose reals include black-and-white photographs and whose fakes do not hands a detector
a feature that will not generalise anywhere.

### How the exclusion is applied

- **At source.** `p1_build_reals.py` now rejects greyscale sources before writing
  anything: single-channel images, and three-channel images whose largest per-pixel
  channel spread is ≤ 2 (`GREYSCALE_TOLERANCE`). The tolerance is not zero because a
  greyscale picture that has been through JPEG comes back with channels a unit or two
  apart. The verdict is taken on the **whole source**, never on one crop: a colour
  photograph may contain a colourless region.
- **In this already-built corpus.** `p1_rescan_greyscale.py` sets
  `excluded_reason=greyscale` on the affected rows. **Nothing is deleted and nothing is
  rebuilt** — not the PNGs, not the 49,200 B5 pair rows, not the 2,488 generations.
  Consumers filter on that column by joining on `image_id`.
- **No backfill.** The corpus is 2,500 minus the excluded count. It is not topped up to
  a round number, and every later count says so.
- **Generations keep their matched variant.** One whose exact partner is excluded draws
  a colour partner by seed instead, recorded as `pairing=sampled_after_exclusion`.
  Dropping it would shrink the training set for a reason unrelated to the generator.
- **Robustness to black-and-white input** is handled in P2 by *synchronized* greyscale
  augmentation — both members of a pair converted together, p ≈ 0.05–0.1 (D-041) — which
  teaches invariance without recreating the shortcut.

### Measured, 2026-09-26 (`reports/p1_greyscale_rescan_t_real.json`)

**83 of 2,500 images are greyscale — 3.3% of the corpus. T-real is now 2,417 rows.**

| kind | n | how it is found |
|---|---|---|
| `single_channel_source` | **11** | the source JPEG had one channel; visible in the manifest as `subsampling = unknown(-1)` |
| `rgb_equal` | **72** | three channels carrying the same values; **only the pixels reveal this** |

**The second kind is 87% of the problem, and no format inspection would have found it.**
Filtering on `unknown(-1)` alone — which is what the 13 matcher failures pointed at —
would have caught 11 and left 72 greyscale photographs in a corpus of colour ones. That
is the case for testing the pixels rather than the header.

The tolerance is barely load-bearing: **82 of the 83 have a channel spread of exactly
0**, and one has a spread of 2. Only that single image depends on where the threshold
sits.

### Effect on stratification

Per COCO supercategory (an image with several annotated supercategories counts in each,
so the columns overlap):

| supercategory | before | after | excluded |
|---|---|---|---|
| accessory | 375 | 354 | 5.60% |
| vehicle | 616 | 583 | 5.36% |
| person | 1,361 | 1,301 | 4.41% |
| indoor | 351 | 337 | 3.99% |
| outdoor | 246 | 237 | 3.66% |
| electronic | 252 | 243 | 3.57% |
| sports | 513 | 495 | 3.51% |
| animal | 510 | 496 | 2.75% |
| furniture | 598 | 589 | 1.50% |
| kitchen | 438 | 433 | 1.14% |
| appliance | 148 | 147 | 0.68% |
| food | 328 | 327 | 0.30% |

The loss is uneven but small: 0.30% to 5.60%, against a corpus average of 3.3%. Nothing
is decimated, and no supercategory falls below 147 rows. The pattern is plausible —
black-and-white photographs in COCO lean towards street scenes and vintage subjects, so
`accessory`, `vehicle` and `person` lose the most, while `food` and `appliance` barely
move. Worth stating wherever per-category results are reported, but it does not
invalidate the stratification.

### Downstream effect

- **B5 pairs:** 83 excluded reals × 12 distinct members = **996 of the 49,200 pair rows**
  are filtered out by the join on `image_id`. Nothing was deleted or rebuilt.
- **Generations:** all 2,488 keep a matched variant. 68 of them had an excluded exact
  partner and drew a seeded colour partner instead, recorded as
  `pairing=sampled_after_exclusion` (exact 2,002, sampled 418, sampled-after-exclusion
  68). **0 failures**, against 13 before the fix.
- **T-ctrl has not been scanned.** DIV2K is 2K photography and is expected to be entirely
  colour, but "expected" is not "measured"; the same scan should be run on it.
