# Data card — E5, Chameleon (AIDE, ICLR 2025)

| field | value |
|---|---|
| set id | `e5_chameleon` |
| role | evaluation only — the hardest "in the wild" set in the corpus |
| official source | URL list, obtained by emailing the first author (`tattoo.ysl@gmail.com`), per `github.com/shilinyan99/AIDE` |
| images obtained from | **third-party mirror** — community-shared link posted in AIDE issue #1 (uploader and date: **PENDING**, see §7) |
| archive | `Chameleon.zip`, **2,683,359,341 bytes** |
| archive SHA-256 | `a7cdadfe35287dd3e81b5c35e7cce5437e68ba3b4cdbb833ac0686b990c2fba6` |
| download date | 2026-09-21 (by the user, locally) |
| terms | **"Chameleon is only used for academic research. Commercial use in any form is prohibited."** (verbatim, AIDE README) |
| code licence | the AIDE *repository* is MIT; that covers the code, **not** the dataset |
| citation | [`docs/CITATIONS.md`](../CITATIONS.md) |

Everything below the terms line was measured in this session by reading the archive's
central directory and each member's JPEG header. **Nothing was extracted and nothing was
written.** Repository metadata and the licence text were fetched directly from
`api.github.com` and `raw.githubusercontent.com`, not via search.

---

## 1. Counts — they match the paper exactly

| | archive | paper Table 1 | agree |
|---|---|---|---|
| real | **14,863** | 14,863 | ✅ |
| AI-generated | **11,170** | 11,170 | ✅ |
| total | 26,033 | 26,033 | ✅ |

Layout is flat: `Chameleon/test/0_real/` and `Chameleon/test/1_fake/`, all `.jpg`
extensions, filenames are UUIDs. No images were added or removed to make anything match.

## 2. Per-category reporting is **blocked** — there are no category labels

The paper's Table 1 breaks the set down as scene 3,574/2,976, object 3,578/2,016, animal
3,998/313, human 3,713/5,865. **The archive carries none of that.** It is two flat
directories of UUID-named files; there is no manifest, no CSV, no per-category
subdirectory, and the AIDE README describes no mapping.

So the instruction to report per category cannot be satisfied from this archive alone.
It needs one of:

- the **official URL list** from the first author, if it is grouped or labelled by
  category (likely — that is how the paper's table was produced); or
- a separate category-mapping file from the authors.

Until one of those arrives, Chameleon is reported **in aggregate only**, and that
limitation is stated wherever the number appears. The animal split — 3,998 real against
313 fake — is the one that most needs the breakdown, because a pooled metric hides it
completely.

## 3. The resolution confound, which is severe

Measured over all 26,033 images:

| | real (14,863) | fake (11,170) |
|---|---|---|
| width | **exactly 500 for every single image** | 327 – 5,376, median 764 |
| height | 92 – 1,130, median 359 | 128 – 6,720, median 768 |
| megapixels (median) | 0.179 | 0.584 |
| distinct widths | **1** | 158 |

> **A one-line rule — `width == 500 → real` — scores 100.00% accuracy on this set.**
> All 14,863 reals are 500 px wide; exactly **1** of 11,170 fakes is.

The reals have clearly been through a normalising pipeline (fixed 500 px width, and see
§4); the fakes are native resolution. This is not a subtle bias, and no detector result
on raw Chameleon images means anything unless the pipeline is provably blind to image
dimensions.

**Our defence is the fixed square crop**: every image enters the model as a `crop × crop`
array, so width is not observable in the pixels. That removes the cue itself. It does
**not** remove the selection effect in §5.

The premise behind "evaluate at native resolution, images go up to 4K" does not hold
here: only the *fakes* reach 4K. The reals are 0.18 MP thumbnails.

## 4. Compression history (CLAUDE.md §6b)

| label | n | JPEG | never compressed | formats | quality min/p25/median/p75/max | chroma subsampling | progressive |
|---|---|---|---|---|---|---|---|
| real | 14,863 | 14,863 | **0** | JPEG 14,863 | **75 / 75 / 75 / 75 / 75** | 4:2:0 ×14,863 | 0 |
| fake | 11,170 | 10,413 | **757** | JPEG 10,413, PNG 642, WEBP 115 | 64 / 75 / 75 / 75 / 100 | 4:2:0 ×10,401, 4:4:4 ×11, unknown ×1 | 11 |

**Verdict: CONFOUNDED.** Never-compressed share: fake 6.8%, real 0.0% — spread 6.8%,
above the 5% threshold, so the matched view is built.

Note how uniform the reals are: **every one of the 14,863 is exactly quality 75, 4:2:0,
non-progressive**. The reference distribution for the matched view is therefore a single
point, and all 757 never-compressed fakes are encoded at exactly q75 / 4:2:0. The joint
sampler handles this correctly — it draws one observed pair, and here every observed pair
is the same one.

The fakes' own JPEGs are *also* mostly q75/4:2:0, which suggests the mirror re-encoded
much of the set. That is a property of this mirror, not necessarily of the official
release (§7).

## 5. The crop threshold is a class-dependent filter — 320, not 384

Because the reals are 500 px wide and short, a minimum-size filter removes them far
faster than it removes fakes. Measured retention:

| crop | real kept | fake kept | fake share of what survives |
|---|---|---|---|
| 256 | 14,767 (99.4%) | 11,109 (99.5%) | 42.9% |
| 288 | 14,091 (94.8%) | 10,842 (97.1%) | 43.5% |
| **320** | **13,835 (93.1%)** | **10,716 (95.9%)** | **43.6%** |
| **384** (project default) | **6,541 (44.0%)** | **10,553 (94.5%)** | **61.7%** |
| 448 | 6,342 (42.7%) | 10,421 (93.3%) | 62.2% |
| 512 | **0 (0.0%)** | 6,900 (61.8%) | 100.0% |

At the project's standard 384 crop, **56% of the reals are discarded and only 5.5% of the
fakes** — the class balance flips from 57/43 to 38/62, and the surviving reals are the
unusually tall ones. That is a new confound introduced by our own pipeline, on top of the
one already in the data.

**Decision (D-037): Chameleon is evaluated at a 320 px crop, single crop.** Run configs
`full_bank_chameleon` and `4ch_only_chameleon` in `configs/p2_runs.yaml` carry it.
Multi-crop is rejected here for the same reason: two non-overlapping 320 crops would
retain 5,211 reals against 9,200 fakes, reintroducing the bias the single crop avoids.

Even at 320, 6.9% of reals and 4.1% of fakes are dropped, so the reported *n* must be the
kept count, not 26,033.

## 6. Metrics

- Class balance at 320 is 13,835 / 10,716, so **report balanced accuracy and AUC**, and
  give **bootstrap confidence intervals** — the review's instruction, and it applies to
  the aggregate as much as to the (currently unavailable) per-category split.
- The animal category's 3,998 : 313 imbalance is exactly why per-category numbers matter;
  flagged in §2 as blocked.

## 7. Provenance, stated honestly

- **Official route:** the AIDE README says to email the first author for the dataset, and
  the release is a **URL list**, not an image archive.
- **What we have:** an image archive obtained from a **community mirror** linked in AIDE
  issue #1, as directed by the first author by email. It is a third-party artefact: the
  uploader, the date and the issue link are **PENDING** — the review said the link would
  be supplied and it has not been pasted yet. The SHA-256 above pins exactly what was
  scanned, whatever its provenance.
- **Consequence:** the mirror may differ from the official release — the uniform q75 in
  §4 is consistent with re-encoding somewhere in the chain. The counts match the paper
  exactly, which is reassuring but is not proof the pixels are the authors'. Any
  Chameleon number is reported as *"from the community mirror, SHA-256 `a7cdadfe…`"*
  until it can be checked against the official URL list.

## 8. Handling rules

- The archive lives **outside `REPO_ROOT`** (it is at `F:\thesis-2\Chameleon.zip`).
  It briefly sat inside and made the integrity check CRITICAL; see I-013 / D-017.
- No image data is committed. Manifests, hashes and summaries only.
- Ingestion and the matched view run **on Kaggle** (CLAUDE.md §6a condition (a),
  and I-010 for the JPEG encoding). The scan behind this card read headers only.
- Leakage against `p1_reals_manifest_t_real.csv` and `p1_reals_manifest_t_ctrl.csv`:
  **PENDING-KAGGLE**.

## 9. Ingestion command

```
python scripts/p1_build_eval.py --set-id e5_chameleon --root <extracted>/Chameleon \
    --real-glob "test/0_real/*" --fake-glob "test/1_fake/*" \
    --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

No `--field` yet: there is no category column to lift (§2). Add one if the official URL
list supplies categories.

## 10. Checklist

- [x] Counts verified against the paper: 14,863 / 11,170, exact.
- [x] Archive SHA-256 recorded.
- [x] Terms recorded verbatim; added to `CLAUDE.md` §6c.
- [x] Compression table measured; confound confirmed.
- [x] Resolution statistics measured; crop decision taken and wired into `p2_runs.yaml`.
- [ ] Mirror link, uploader and date recorded (**PENDING** — §7).
- [ ] Per-category labels obtained (**BLOCKED** — §2).
- [ ] AIDE citation confirmed against the ICLR proceedings (`CITATIONS.md` currently has
      the repository's arXiv-preprint form).
- [ ] Ingested on Kaggle; leakage `CLEAN`.

## Measured on ingestion, 2026-09-28

`reports/kaggle_20260928/p1_eval_e5_chameleon_summary.json`. **26,033 rows: 14,863 real /
11,170 fake — matching the paper's Table 1 exactly.** **Leakage CLEAN** against both
training manifests.

| check | result |
|---|---|
| compression | `CONFOUNDED; never-compressed share: fake 6.8% vs real 0.0%` — the 642 PNG + 115 WEBP fakes the audit predicted, 757 in total |
| matched view | 757 re-encoded; confounded **False** afterwards |
| separability, raw | subsampling **+2.95**, quality **+3.13** — below the 5-point threshold, so not flagged, but not nothing |
| separability, matched | **+0.05 / +0.22** — the view removes what little there was |
| near-duplicate groups | 282, reported not removed |

The matched view is therefore the view to report from, even though the raw set fell under
the flagging threshold.

**None of this touches the set's real problem**, which is documented above: every one of
the 14,863 reals is exactly 500 px wide, so `width == 500 → real` scores 100.00% on raw
images. The fixed square crop is what defends against that, and it is why Chameleon runs
at 320 px (D-037). Compression is the smaller of the two confounds here, and it is the
one that can be fixed.

## The PRIMARY view is the pipeline-matched one (D-043)

The compression-only matched view fixes the compression confound and **leaves the real
one untouched**: every real is exactly 500 px wide because Chameleon's reals were
thumbnailed, so `width == 500 → real` still scores **100.00%** after it.

So the fakes go through the reals' whole pipeline, not just its JPEG step:

1. downscale to **width 500**, aspect ratio preserved, **LANCZOS**;
2. encode at **q75, 4:2:0** — the reals' own uniform setting;
3. **reals untouched**, originals untouched; this is a derived view.

`--pipeline-match-resample bicubic` produces the sensitivity check, so no claim rests on
one filter's ringing signature.

This deliberately resizes images. CLAUDE.md §7's "never resize" governs the
**reconstruction protocol**, where a resize would fabricate a forensic trace. Here the
resize *is* the trace being matched, and it is applied to the one class that lacks it.

| view | status | why |
|---|---|---|
| **pipeline-matched** | **PRIMARY** | the only view in which width is worth nothing |
| compression-matched | secondary | fixes compression, leaves the width confound |
| raw | reported to show the confound | `width == 500 → real` scores 100.00% |

Verify in the summary that `width_rule_after.accuracy == width_rule_after.baseline`.
Anything else means the view did not do its job.

**What this does not fix:** downscaling a 4K fake to 500 px destroys exactly the
high-frequency detail a decoder fingerprint lives in. The pipeline-matched view is the
*fair* comparison, not the *favourable* one, and a weak score here may reflect the
resize rather than the method. Report it alongside the 320-px crop policy (D-037), which
the user confirmed.

## Provenance of the mirror — resolved 2026-09-28 (I-031)

Read from the GitHub API (`api.github.com`, metadata only; the archive itself was never
downloaded to the local machine, per CLAUDE.md §6a(a)):

| field | value |
|---|---|
| issue | [`shilinyan99/AIDE#1`](https://github.com/shilinyan99/AIDE/issues/1), *"Request for Chameleon dataset"*, opened by `Yanmou-Hui` 2024-07-12 |
| mirror posted by | **`qwqhjcxczx`**, **2025-01-02T03:45:28Z** |
| link | `https://drive.google.com/file/d/1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH/view` |
| archive in hand | `Chameleon.zip`, 2,683,359,341 bytes, SHA-256 `a7cdadfe35287dd3e81b5c35e7cce5437e68ba3b4cdbb833ac0686b990c2fba6` |
| counts | 14,863 real / 11,170 fake — matches the paper's Table 1 exactly |

The first author confirmed by email that the official release is **a URL list only**, and
pointed to this issue.

### What the uploader actually did, and why it changes the reading

In their own words: *"I have downloaded the Chameleon dataset using the image URLs
provided by the author."* So the mirror is **a re-download from the original hosts**, not
a redistribution of the authors' own files.

That is the likely explanation for the confound this card documents above: **every real
is exactly 500 px wide and uniformly JPEG q75 4:2:0.** Image hosts serve resized,
re-encoded derivatives, and a bulk download at a fixed width would produce precisely this
signature. The reals are sourced from Unsplash, which serves width-parameterised JPEGs.

**So the width confound may belong to the mirror rather than to Chameleon.** It cannot be
settled from here — the authors' own copy is not public — and the distinction does not
change what has to be done, since this is the copy being evaluated. But it changes what
may be *said*: the confound is a property of **this distribution of Chameleon**, and a
paper reporting on the authors' copy is not necessarily reporting on the same pixels.
Every Chameleon result states the mirror, its SHA-256 and this caveat.

### Verifying the chain (optional, Kaggle only)

Downloading that Drive id on Kaggle with `gdown` (§6a(c)) and hashing it would confirm
the archive in hand is the one from the issue. Matching `a7cdadfe…` closes the chain;
anything else means the file was re-uploaded or changed and the provenance stops at
"received from the user".

### Measured, 2026-09-28 (`reports/kaggle_20260928/p1_eval_e5_chameleon_summary.json`)

The view does what it was built to do:

| | before | after |
|---|---|---|
| `width == 500 -> real` | **99.996%** (14,863 of 14,863 reals; 1 of 11,170 fakes) | **57.093%** |
| lift over the majority baseline | **+42.90** | **0.00** |
| parameter separability | — | **not separable**, no offenders |
| compression confounded | true, as distributed | **false** |

11,169 fakes were downscaled; 14,863 reals untouched. After the view, knowing an image's
width is worth *exactly* the majority-class baseline — which is the strongest form the
claim can take: not "reduced", but worth nothing.

**The BICUBIC sensitivity check agrees to the digit** (`e5_chameleon_bicubic`): same
before, same after, same verdict. So no Chameleon result rests on one resampling filter's
ringing signature.

One fake was already 500 px wide, which is why "before" is 99.996% rather than 100.000%.
## P2 evaluation panel (D-048)

**2,000 per label from the pipeline-matched view**, 4,000 rows. The compression-only matched view and the raw view are not drawn from; the pipeline-matched one is PRIMARY (D-043), and it is the only view in which `width == 500 -> real` is worth nothing.
