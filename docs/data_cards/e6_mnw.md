# Data card — E6, MNW

**Status: repository resolved; blocked on the download route.**

| field | value |
|---|---|
| set id | `e6_mnw` |
| full name | **Microsoft-Northwestern-WITNESS** benchmark |
| role | evaluation only — **no training, ever** |
| repository | `github.com/microsoft/MNW` (mirror: `nsail-lab/MNW`) |
| default branch / pushed | `main`, `2026-09-19T20:24:20Z` |
| repository licence file | **none** (`license: null` from the GitHub API) — terms are in the README |
| navigator | `https://microsoft.github.io/MNW/` |
| terms | **evaluation only, no training, no commercial use, citation required** |
| citation | ✅ recorded verbatim in [`docs/CITATIONS.md`](../CITATIONS.md) |
| download URL | **none in the README** — access route pending (§5) |
| built by | `scripts/p1_build_eval.py`, once the images are in hand |

Repository metadata and README fetched 2026-09-21 by direct URL (`api.github.com` and
`raw.githubusercontent.com`), not via search — which is why the earlier search-based
attempt failed: MNW is a GitHub repository, and I had been searching Hugging Face.

---

## 1. What MNW is

From the README and the summary paper (IEEE Intelligent Systems 41(02):15-23, 2026):

- **More than 50,000 artifacts** — images, videos **and audio**. Only the image portion
  is in P1 scope; the video portion becomes relevant to the S4 temporal stream later.
- Built for **breadth over depth**: small samples from as many generators as possible,
  rather than millions of samples from a few. That is the opposite of GenImage's shape
  and makes it a good complement to E8.
- Contains **real-world cases encountered by journalists and human rights defenders**,
  annotated by experts — the "in the wild" half that a lab benchmark cannot supply.
- Contains **adversarial examples created with state-of-the-art attacks**. A poor score
  on that portion is an adversarial-robustness result, not a generalisation result, and
  must be reported as such rather than folded into an aggregate.
- Periodically updated, so **the revision must be pinned** and the pin recorded here.

The README also asks that entities purchasing detection solutions avoid using it to
evaluate commercial tools — consistent with the no-commercial-use term.

## 2. Terms, which are already binding

Recorded from the P1 review instruction and added to CLAUDE.md §6c:

- **Evaluation only.** MNW may never appear in a training set, a validation set used for
  model selection, or a threshold-calibration split. `Calibration` refuses to apply a
  threshold to the set it was fitted on (D-026), which makes the calibration half of
  that mechanical.
- **No commercial use.** Any deployment claim must exclude MNW, alongside Synthbuster
  (NC), WildRF and Chameleon (research-only) and DIV2K (academic-only).
- **Citation required.** The set may not be reported on until its entry in
  `docs/CITATIONS.md` is filled in verbatim from its own README.

Verified this session, verbatim from the README:

> **This dataset is intended for evaluation purposes and cannot be used for training or
> for commercial purposes.**

The repository ships no licence file, so this README notice *is* the term. Recorded in
CLAUDE.md §6c.

## 3. Compression history (CLAUDE.md §6b)

| label | n | JPEG | never compressed | formats | quality min/median/max | chroma subsampling |
|---|---|---|---|---|---|---|
| real | PENDING | | | | | |
| fake | PENDING | | | | | |

Same treatment as every other set: `p1_build_eval.py` builds the table, decides whether
compression is confounded with the label, and builds the JPEG-matched view if it is.
Nothing about MNW exempts it from that.

## 4. Leakage

Perceptual hash against `p1_reals_manifest_t_real.csv` and
`p1_reals_manifest_t_ctrl.csv`, 6-bit threshold, into `reports/p1_leakage.json` under
`e6_mnw`. **PENDING.**

## 5. The download route — the remaining blocker

**The README contains no download URL.** The navigator at
`https://microsoft.github.io/MNW/` browses the dataset but is not an archive link. The
access route is to be supplied separately.

When it arrives:

- **If it carries a token or signed URL, that value goes in Kaggle Secrets only.** Never
  in the repository, never in a log line, never in a report — the same rule as `HF_TOKEN`
  (CLAUDE.md §4). The runbook reads it with `UserSecretsClient().get_secret(...)` and
  reports only `present: true/false`.
- The download happens **on Kaggle only** (§6a condition (a)).
- Pin whatever revision or release the route identifies, and record it in the table above
  — MNW is periodically updated, so an unpinned ingestion would not be reproducible.

```
python scripts/p1_build_eval.py --set-id e6_mnw --root <extracted root> \
    --real-glob "<pattern>" --fake-glob "<pattern>" \
    --field modality=<depth> --field adversarial=<depth> \
    --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

The globs are written after the runbook prints the extracted tree, exactly as for E3 and
E4. Restrict them to the **image** portion. Lift a field for the adversarial subset if
the layout exposes one, so it can be reported separately (§1).

## 6. Checklist before E6 appears in any table

- [x] Repository identifier supplied and resolved by command (`microsoft/MNW`).
- [x] Licence/terms read from the repository and recorded verbatim.
- [x] Citation pasted verbatim into `docs/CITATIONS.md`.
- [ ] **Download route supplied** (§5); any token into Kaggle Secrets only.
- [ ] Revision or release pinned in the runbook — MNW is periodically updated.
- [ ] Image portion isolated from the video and audio artifacts.
- [ ] Adversarial subset identified and reported separately.
- [ ] Compression table filled in.
- [ ] Leakage check `CLEAN`.
- [ ] MNW excluded from every commercial or deployment claim.
