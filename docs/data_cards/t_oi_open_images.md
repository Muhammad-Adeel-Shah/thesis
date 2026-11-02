# Data card — T-OI, Open Images V7 (validation split, CVDF copies)

| field | value |
|---|---|
| set id | `t_oi` |
| role | **training** reals for `S2-D` and `S2-B+reals` (3,600), and **held-out** reals (400) for the k = 0 union threshold (pre-registration §4). Never test data |
| source | Open Images V7 **validation** split, CC BY 2.0 rows only; image files from the CVDF S3 bucket `open-images-dataset` (`validation/<ImageID>.jpg`) |
| licence | images: CC BY 2.0 per image (per-image attribution in `docs/p3_open_images_attribution.csv`); annotations (not used): CC BY 4.0 |
| size | **4,000** crops: 3,600 `train`, 400 `held_out` |
| working crop | 384 × 384, centre crop of the native image, **never resized**, PNG |
| built by | `scripts/p3_fetch_open_images.py --source s3`, runbook `env/kaggle_run_p3_open_images.py`, code `dbp-code-69cc811`, one Kaggle CPU session |
| stored | Kaggle private dataset `dbp-p3-open-images` (account `muhammadadeelshah`); never committed |

Every value below comes from a committed file, named beside it, or from a page fetched
on 2026-10-17 (sha256 given). Nothing is from memory (CLAUDE.md §9a).

---

## 1. Source pages (fetched 2026-10-17)

| page | sha256 | used for |
|---|---|---|
| `https://storage.googleapis.com/openimages/web/factsfigures_v7.html` | `03f731edf0a6524ae0403253fb3d805b51a42bbaf45a436bfc4491a650fa4fb0` (identical to the P3-A fetch) | licence, citation |
| `https://storage.googleapis.com/openimages/web/download_v7.html` | `6d71544e85e5205878573bf489736927e9cc2c552e56878b5c8ff3a24803954a` (identical to the P3-A fetch) | metadata columns, `Rotation` |
| `cvdfoundation/open-images-dataset` `README.md` at commit `16c6bae89b22dddc7194618bbb0e24852a911d94` | `6ed423bff11ed77e142ff173309a35fe5c4601b576dd7dc616a65ae3dd480f48` | the S3 copies |

**Licence**, verbatim (factsfigures_v7): "The annotations are licensed by Google LLC
under CC BY 4.0 license. The images are listed as having a CC BY 2.0 license. Note: while
we tried to identify images that are licensed under a Creative Commons Attribution
license, we make no representations or warranties regarding the license status of each
image and you should verify the license for each image yourself."

So each row's own `License` field is checked: only rows listing
`https://creativecommons.org/licenses/by/2.0/` are kept (41,620 of the validation split,
`reports/p3_open_images/oi_validation_cc_by_2.csv`, sha256 `5eabde3b…5c2b`).

**The files**, verbatim (CVDF README): "The images are rescaled to have at most 1024
pixels on their longest side, while preserving their original aspect-ratio."
Measured exception: 5 of the 4,000 accepted sources are larger (up to 8,100 × 3,888).

**Rotation**, verbatim (download_v7): "Rotation is the number of degrees that the image
should be rotated counterclockwise to match the Flickr user intended orientation ( 0 ,
90 , 180 , 270 ). nan means that this information is not available." The pixels are used
**as stored** and the value is recorded: 39 accepted rows have 90/180/270 and 493 have
none.

## 2. Why Open Images, and why these copies

- **Platform rule (pre-registration §2):** no shared platform with any Panel B view's
  reals. Unsplash (E5's source) is excluded; Open Images is Flickr. Deduplication covers
  COCO, which is also Flickr.
- **Copies, not originals** (user decision 2026-10-15, log P3-B step 3): Flickr refused
  bulk downloads of the originals from Kaggle and from a home IP. The CVDF copies are the
  same photographs, rescaled to ≤ 1024 px; `OriginalMD5` therefore does not apply and
  each file's sha256 is recorded instead. The test reals are web-processed too, and S2-D's
  degradation chains add compression variety.

## 3. Selection (all re-derived locally, log P3-B step 3, 2026-10-17)

- The CC BY 2.0 rows are walked in a seed-2026 order (`dbp.data.open_images.order_key`);
  the first 4,000 that pass every rule are kept. 4,096 rows were walked.
- **Rejected (87):** greyscale mode L/1 51; greyscale `rgb_equal` (D-039, judged on the
  whole source) 8; shorter side < 320 8; 320–383 6; CMYK 3; pHash duplicates (≤ 6 bits)
  of COCO train2017 6 + 2 at crop level, of E5 reals 2 (crop level), of Panel B 1.
  9 rows were fetched but not needed.
- **Deduplicated against** (sha256 and pHash ≤ 6): Panel B (9,405), every P2 panel view,
  T-real, T-ctrl, the SD1.5 generations and `gen_matched`, the 49,200 bank
  reconstructions, all E3 (726), E4 (1,251) and E5 (14,863) reals, COCO train2017 and
  val2017 (123,287). Source: `reports/kaggle_20261017/p3_open_images/p3_open_images_summary.json`.
- **Split:** seeded (`assign_split`, seed 2026): 400 `held_out`, 3,600 `train`.
- **Within the set:** 0 crop pairs at pHash ≤ 6 bits; 4,000 distinct ids and sha256.

## 4. Compression history (CLAUDE.md §6b)

All crops are PNG; the table describes the **source** JPEG each crop was cut from
(`reports/kaggle_20261017/p3_open_images/p3_open_images_manifest.csv`, sha256
`0623a6b0…5961`).

| split | n | source JPEG | quality min / median / max | chroma subsampling |
|---|---|---|---|---|
| train | 3,600 | 3,600 | 26 / 94 / 100 | 4:2:0 1,620 · 4:4:4 995 · 4:2:2 910 · `unknown(-1)` 75 |
| held_out | 400 | 400 | 50 / 93 / 100 | 4:2:0 158 · 4:4:4 134 · 4:2:2 103 · `unknown(-1)` 5 |

`unknown(-1)` rows are colour images (mode RGB) whose sampling layout Pillow does not
name. Every real here is JPEG-derived; its reconstructions (step 4a) are PNG renderings
of the same crop, so within a pair the compression history is shared, not a label cue.

## 5. Known properties (recorded, no rule broken)

1. 4 crops have channel spread ≤ 2 (a crop of a colour source can be locally colourless;
   D-039 judges the source, as for T-real).
2. 5 sources exceed 1,024 px despite the CVDF note.
3. Source quality: 1 below 50, 40 below 70.
4. Rotation used as stored (above).
5. 68 of 4,000 author names contain HTML entities (e.g. `O&#x27;Brien`), as distributed
   (657 of the 41,620 metadata rows). Unchanged in the attribution file; unescape only
   for display.
6. EXIF camera make/model on 4 of 4,000.

## 6. Use restrictions and citation

- CC BY 2.0 requires attribution: `docs/p3_open_images_attribution.csv` (image id,
  author, original URL, licence; sha256 `d010781f…1093`) travels with any redistribution.
  The images themselves are not redistributed; the Kaggle dataset is private.
- Citation: `docs/CITATIONS.md`, "Open Images V7".
