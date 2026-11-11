# Data card — E7, DDA-COCO

| field | value |
|---|---|
| set id | `e7_dda_coco` |
| role | evaluation only — **PRIMARY head-to-head set**, carries the main claim |
| source | `Junwei-Xi/DDA-COCO` (Hugging Face **dataset** repo) |
| pinned revision | `8c9330a3b374bcac46a8045a0e3c09ebcf7868fb` |
| licence | `apache-2.0` (Hub tag) |
| gated | no |
| archive | `DDA-COCO.zip`, repo total **4,301,456,312 bytes** (4.01 GiB) |
| last modified | 2026-02-01 |
| paper | NeurIPS 2025, `arxiv:2505.14359` — citation in [`docs/CITATIONS.md`](../CITATIONS.md) |
| built by | `scripts/p1_build_eval.py`, per `env/kaggle_run_p1_eval.md` |

Metadata, licence, revision, size and README were read by command on 2026-09-20 via
`HfApi.dataset_info` and `hf_hub_download`.

---

## 1. Why this is the primary set

SEF reports **FLUX.1 82.8** and **SD3.5 88.2** here. Those are the numbers this thesis
is measured against, so DDA-COCO is where the central claim is made or lost. Everything
else in the evaluation suite is breadth.

It is also the *hardest* set for the right reason. Quoting its README:

> DDA-COCO is a benchmark specifically designed to evaluate whether AIGI detectors rely
> on **"non-causal features"** (such as compression artifacts or content semantics).

The real and synthetic images are aligned in semantics, size and frequency distribution,
so a detector cannot win by reading dataset bias. That is exactly the discipline this
project imposes on itself elsewhere (D-001, §6b), met here as an external test.

## 2. Composition

From the README, read 2026-09-20:

- **Source (real):** MSCOCO **validation** set.
- **Synthetic:** five subsets, each a different VAE reconstruction (README names SD1.5,
  SD2.1, SDXL among them), with frequency alignment applied.

Per-subset and per-label counts: **PENDING-KAGGLE**. The repository ships a single zip,
so the internal structure is confirmed on download; the runbook prints the tree before
the globs are written.

## 3. A construction overlap worth stating plainly

DDA-COCO's "fake" images are **VAE reconstructions of COCO images**. So are the training
pairs this project builds in B5.

That is not a flaw in either, but it must be said out loud, because it changes what a
good score means:

- a detector trained on DBP pairs and evaluated on DDA-COCO is being tested on the
  *same operation* it was trained on, over a partly overlapping set of VAEs;
- the genuinely novel part of the claim is therefore the **cross-lineage** result — how
  the detector does on DDA-COCO subsets whose VAE is **not** in the bank — not the
  aggregate number;
- the aggregate must be reported broken down per subset, so a reader can see which
  subsets are in-lineage and which are not.

Report it that way, or the headline number flatters itself.

## 4. Leakage — the one that actually bites (I-024)

**DDA-COCO's reals are MSCOCO validation. T-real is built from COCO.** If T-real were
allowed to select from `val2017`, the evaluation reals would be in the training set.

Two defences, both in place:

1. **T-real is restricted to `train2017` at the source**, by pointing
   `p1_build_reals.py --root` at the train images directory rather than the dataset
   root. The build runbook says so and says why. B6's prompts already come from
   `captions_train2017.json`.
2. **The perceptual-hash leakage check is run anyway**, against both
   `p1_reals_manifest_t_real.csv` and `p1_reals_manifest_t_ctrl.csv`, at a 6-bit
   threshold. Result in `reports/p1_leakage.json` under `e7_dda_coco`.
   **PENDING-KAGGLE.**

If the check reports `LEAKAGE`, the correct response is to fix the training selection
and rebuild — **not** to drop the offending evaluation images, which would silently
change the benchmark.

## 5. Compression history (CLAUDE.md §6b)

| label | n | JPEG | never compressed | formats | quality min/median/max | chroma subsampling |
|---|---|---|---|---|---|---|
| real | PENDING-KAGGLE | | | | | |
| fake | PENDING-KAGGLE | | | | | |

Fill from `reports/p1_eval_e7_dda_coco_summary.json` after the run.

**Expectation, stated in advance so the measurement can contradict it:** DDA-COCO was
built to be aligned, and its README names compression artefacts as one of the non-causal
features it controls for, so this table is expected to come back **not confounded**. It
is measured regardless. If it does come back confounded, the JPEG-matched view is built
by the same rule as every other set and becomes the primary view for E7 — but that would
also be a finding about the benchmark worth reporting.

## 6. How to report it

- Per subset (per source VAE), always. The aggregate alone hides the in-lineage /
  cross-lineage distinction that §3 describes.
- Head-to-head against SEF's published FLUX.1 82.8 and SD3.5 88.2, on the matching
  subsets.
- With the leakage verdict stated in the same table, not in a footnote.

## 7. Licence

`apache-2.0`, no non-commercial restriction. Citation required as a matter of academic
practice rather than licence; recorded verbatim in `docs/CITATIONS.md`.

## Measured on ingestion, 2026-09-27

Built by `p1_build_eval.py` on Kaggle from the pinned revision. **34,969 rows**:
5,000 real and 29,969 fake across six source VAEs.

| source_vae | n | subsampling |
|---|---|---|
| FLUX.1 | 4,971 | **4:2:0** |
| sd-vae-ft-ema | 5,000 | 4:4:4 |
| sd-vae-ft-mse | 5,000 | 4:4:4 |
| sdxl-vae | 5,000 | 4:4:4 |
| stable-diffusion-2-1 | 5,000 | 4:4:4 |
| stable-diffusion-3.5-large | 4,998 | 4:4:4 |
| **real** (COCO val2017) | 5,000 | 4:4:4 × 4,990, `unknown(-1)` × 10 |

The reals are **not shipped with the set**: DDA-COCO contains reconstructions only, and
the originals are MSCOCO `val2017`, mounted from the COCO dataset. That is precisely why
T-real is restricted to `train2017` (I-024), and the check confirms it held.

### Leakage: CLEAN

`no leakage against p1_reals_manifest_t_real.csv (2500 training images)` and
`p1_reals_manifest_t_ctrl.csv (1600)`. E7 can carry the head-to-head claim without that
caveat.

The 5,000 near-duplicate groups reported *within* the set are its structure — each
val2017 photo plus its six reconstructions — not contamination. Perceptual hashing
therefore says nothing useful about duplication inside E7; only the cross-set check does.

### ⚠️ The FLUX.1 subset is separable by chroma subsampling alone (D-042)

Compression history is **not** confounded with the label — both classes are entirely
JPEG at q80/96/96 — but the *parameters* are, on one subset:

> **`4:2:0 → fake` scores 100.00% on the FLUX.1 subset**: 4,971 of 4,971 fakes are
> 4:2:0, and 0 of 5,000 reals are. Measured lift over the majority-class baseline:
> **+49.85 points**. Pooled across all six subsets the lift is **+0.03**, which is why
> this has to be measured per subset.

This is how DDA-COCO was published — the FLUX.1 reconstructions were evidently written
through a different encoder path — and it is a property of the distribution, not of
FLUX's VAE.

**It cannot be undone.** Chroma subsampling is destructive, so re-encoding at 4:4:4
upsamples rather than restores; the chroma blur and 2×2 block structure remain in the
pixels. Equalising the headers would conceal the shortcut while leaving the signal, so
no matched view is built for it.

**Therefore:**

- Any FLUX.1 number from E7 is reported as an **upper bound**, with this caveat attached,
  and is never the headline on its own.
- The other five subsets are clean — all 4:4:4, matching the reals — and carry the
  comparison.
- SEF's published FLUX.1 figure was measured on this same distribution. Nothing is
  claimed here about what their detector used; the point is that the subset limits both
  sides, which is an argument for reporting the caveat rather than a bare number.
- An optional bounding ablation — FLUX.1 against reals re-encoded to 4:2:0 — would add a
  second JPEG generation to the reals and so bounds the effect rather than removing it.
  Not run; recorded as an option.

## Reporting rule for FLUX.1 (D-044, mandatory)

**Every E7 FLUX.1 number is printed next to its trivial-rule baseline.** Not in a
footnote, not in the caption — in the table, in the adjacent column:

| subset | our score | trivial-rule baseline | in-bank? |
|---|---|---|---|
| FLUX.1 | *x* | **+49.85 lift: `4:2:0 → fake` scores 100.00%** | per run config |

A reader must be able to see, without leaving the row, that a detector scoring highly on
this subset may be reading chroma subsampling. The quality shortcut (+14.35) is stated in
the same cell.

**P2 must run a sensitivity check on this subset** before any FLUX.1 number is claimed:
score the subset with the chroma channels neutralised — luma-only input, or reals and
fakes brought to a common chroma treatment — and report both numbers. If the score
survives, it is evidence about the decoder; if it collapses, it was chroma. Either result
is publishable; only the unchecked number is not.
## P2 evaluation panel (D-048)

**500 per subset**, stratified by `source_vae` -- the six source VAEs plus the shared real class, 3,500 rows.

Under the primary protocol `p_a_sd1` the unseen subsets are `sdxl-vae`, `stable-diffusion-3.5-large` and `FLUX.1`; `sd-vae-ft-ema`, `sd-vae-ft-mse` and `stable-diffusion-2-1` are in-bank and are reported but excluded from M1 and M2. The subset-to-member mapping is asserted in `p2_score.py`, not inferred from the name: a subset that failed to map would be marked unseen by default, which would inflate every generalisation claim.

FLUX.1 enters the gate in its **luma-only** variant only (pre-registration §8).
