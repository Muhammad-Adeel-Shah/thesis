# Data card — E8, EvalGEN

| field | value |
|---|---|
| set id | `e8_evalgen` |
| role | evaluation only — breadth against 2024–2025 generators and heterogeneous architectures |
| source | `Junwei-Xi/EvalGEN` (Hugging Face **dataset** repo) |
| pinned revision | `86c719a9bfc72b7e5d91bfbb04432774feb4e3b7` |
| licence | `apache-2.0` (Hub tag) |
| gated | no |
| archive | `EvalGEN.zip`, repo total **14,538,202,808 bytes** (13.54 GiB) |
| last modified | 2025-12-15 |
| paper | NeurIPS 2025, `arxiv:2505.14359` — citation in [`docs/CITATIONS.md`](../CITATIONS.md) |
| built by | `scripts/p1_build_eval.py`, per `env/kaggle_run_p1_eval.md` |

Metadata, licence, revision, size and README read by command on 2026-09-20.

---

## 1. Why this set

E3 covers two closed MLLM generators; E7 covers VAE reconstructions of COCO. Neither
covers **auto-regressive** image generation, which is where the architectures are moving.
From the README:

- **553 distinct aligned prompts**, from the GenEval benchmark.
- **~55,300 images, 11,060 per generator** — so 20 images per prompt per generator.
- **Five generators:** Flux (diffusion), GoT (LLM reasoning + diffusion), Infinity
  (bitwise auto-regressive), OmniGen (unified multimodal), NOVA (non-quantised
  auto-regressive).

Three of those five are not diffusion models at all. A decoder-bank probe is built on
the premise that a generator's decoder leaves a fingerprint; an auto-regressive
generator that never runs a VAE decoder is the sharpest available test of where that
premise stops holding. **That is the point of including E8, and a poor score here is a
result, not a failure to hide.**

## 2. It has no real class

The README describes 55,300 images across five generators and names no real source.
Consequently:

- **AUC, average precision and EER are undefined on E8 alone.** A ranking metric needs
  both classes. `dbp.eval.metrics` raises rather than returning 0.5, so this cannot
  be reported by accident.
- E8 must be **paired with a real set** to produce any threshold-free number. The
  defensible pairing is E7's COCO-validation reals, since both come from the same
  authors and the same paper.
- What E8 *can* report on its own is the **detection rate at a threshold calibrated
  elsewhere** — which is exactly what `Calibration` was built for (D-026): fit on a
  held-out real/fake split, apply unchanged, report TPR per generator.

Confirm the absence of a real class when the zip is extracted; if reals are present
after all, this section is wrong and the manifest will say so.

### The metric policy (D-035)

**Primary — fake detection rate at a threshold fitted on our own validation reals**, at
FPR 1% and FPR 5%. The threshold is a `Calibration` fitted on a held-out split of our
reals and applied unchanged (D-026); `evaluate_with_calibration` refuses to apply it to
the set it was fitted on, so the operating point cannot drift onto E8. Reported **per
generator**.

This is the honest primary because it needs no reals inside E8 at all: it asks "of these
known fakes, how many do we catch, at a false-positive budget we fixed elsewhere".

**Secondary — AUC against COCO `val2017` reals re-encoded to JPEG q96**, matched to E8's
uniform q96 by the §6b rule, with this caveat stated every single time it appears:

> GenEval prompts describe compositional scenes ("two apples and a blue bench") that
> differ in content from COCO photographs. An AUC computed across that boundary can be
> inflated by **content** rather than by generative artefacts, and cannot be cleanly
> separated from the detector's actual ability.

**The secondary is never the headline.** It appears below the primary, in the same table,
carrying the caveat. A reader who reads only the first row must not come away with the
inflated number.

Note also that the reals for the secondary come from COCO `val2017` — deliberately the
*validation* split, because T-real trains on `train2017` only (I-024). Using train images
as evaluation reals would leak.

### Budget control (D-036)

E8 is 13.54 GiB and ~55,300 images. The subset rule, wired into `p1_build_eval.py`:

```
--subsample 1000 --stratify-by generator
```

- **1,000 images per generator, 5,000 total.**
- Seeded: a permutation of each stratum's sorted `image_id`s, salted with the set id, so
  the subset does not depend on directory order and is identical on any machine.
- **Stratified by generator**, so per-generator comparisons stay balanced. "The first N
  files" is explicitly rejected: in a set laid out by generator it would return one
  generator.
- The sampler, the seed rule, the per-stratum counts and **every kept `image_id`** are
  recorded in `reports/p1_eval_e8_evalgen_summary.json`, so the subset is a citable
  object rather than whatever that run happened to pick.
- Downloaded to `/tmp` (not persisted); only the subset is kept.

The same flags are available for E7 if the preflight throughput demands it — E7 is
reported per source VAE, so it would be stratified by that field instead.

## 3. Compression history (CLAUDE.md §6b)

The README states the format explicitly:

> **Format:** Unified as JPEG (Quality Factor 96) to simulate real-world web
> environments.

So every image in E8 is expected to be **JPEG at quality 96**, with no never-compressed
images at all.

| label | n | JPEG | never compressed | formats | quality min/median/max | chroma subsampling |
|---|---|---|---|---|---|---|
| fake | PENDING-KAGGLE | expect all | expect 0 | expect JPEG | expect 96/96/96 | PENDING-KAGGLE |
| real | none in this set (see §2) | | | | | |

Two consequences, both worth stating before the measurement:

- **Within E8 there is no compression confound**, because there is only one label. No
  matched view is needed and `p1_build_eval.py` will report "fewer than two labels".
- **Across the pairing there may well be one.** If E8's q96 fakes are paired with E7's
  reals, and those reals have a different quality distribution, the pair is confounded
  even though neither set is confounded alone. The measured quality distribution of both
  goes in this card, and if they differ materially the matched-view rule applies to the
  *pairing*, using E7's reals as the reference.

That is a genuinely new case the §6b rule has to be read carefully to cover, and it is
flagged here rather than discovered later.

## 4. Per-generator reporting

The five generators are the whole reason for the set, so every number is reported per
generator, never pooled. `--field` is used to lift the generator name out of the
directory structure once the zip's layout is known; the runbook prints the tree first.

Expect a wide spread: Flux has a VAE decoder the bank can probe, while Infinity and NOVA
do not. A pooled number would average those two regimes into something uninterpretable.

## 5. Leakage

Checked by perceptual hash against both training manifests, 6-bit threshold. Result in
`reports/p1_leakage.json` under `e8_evalgen`. **PENDING-KAGGLE.**

Prior risk is low — these are generated images from prompts, not photographs — but the
GenEval prompts describe common scenes and some generators reproduce training imagery
closely, so the check is run.

## 6. Size, and a budget note

13.54 GiB compressed. Downloaded to `/tmp` on Kaggle, not to `/kaggle/working`, and only
manifests and summaries come back. The subset rule is now fixed rather than
conditional — see "Budget control" in §2: 1,000 per generator, seeded and stratified,
with every kept id recorded.

## 7. Licence

`apache-2.0`, no non-commercial restriction. Citation recorded verbatim in
`docs/CITATIONS.md`. Its prompts come from GenEval, which should be cited as well; that
reference is marked NOT-VERIFIED until a command reads it.

## Measured on ingestion, 2026-09-28

`reports/kaggle_20260928/p1_eval_e8_evalgen_summary.json`. **5,000 rows, all fake** —
**1,000 from each of Flux, GoT, Infinity, NOVA and OmniGen**, drawn from 55,298 by the
seeded stratified sampler (D-036). Every kept `image_id` is recorded in the summary, so
the subset is a citable object. **Leakage CLEAN** against both training manifests.

**The absence of a real class is confirmed**, which §2 predicted: `compression_confounded`
reports `fewer than two labels with images; nothing to compare`, and the D-042 check
reports `fewer than two labels; separability is undefined` rather than a meaningless
perfect score. No matched view is built, correctly — there is no real distribution in
this set to match against.

E8 is therefore reported exactly as D-035 specifies: **fake detection rate at thresholds
fitted on our own validation reals**, per generator, with the COCO-paired AUC as a
caveated secondary.
## P2 evaluation panel (D-048)

**250 per generator**, 1,250 rows, no reals. AUC on this set alone is undefined (D-035); it contributes TPR at validation-fitted thresholds and nothing else.
