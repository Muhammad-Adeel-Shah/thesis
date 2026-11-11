# Kaggle run — P2-B6/B7/B8: calibrate, score once, decide

**CPU only.** Everything expensive is already cached; this session reads features and
writes numbers. That matters, because it means a mistake here costs minutes rather than
GPU-hours — and because there is one step in it that must not be repeated.

---

## Step 0 — the pre-registration must already be committed

```python
prereg = DBP / "docs" / "P2_PREREGISTRATION.md"
assert prereg.is_file(), "commit the pre-registration before scoring the panel"
!cd {DBP} && git log -1 --format="%H %ci" -- docs/P2_PREREGISTRATION.md
```

**Record that hash in the report.** It must predate the scores. `p2_analyze.py` refuses
to run without the file, but only the commit date proves it was fixed *before* the
numbers existed, which is the whole point of pre-registering.

---

## Step 1 — fit the heads, on train and validation only

```python
for protocol in ("p_a_sd1", "4ch_only", "full_bank"):
    !cd {DBP} && python scripts/p2_fit_heads.py --protocol {protocol} \
        --s1-cache outputs/p2/s1 --probe-dir outputs/p2/probe \
        --s2-scores reports/p2_s2_scores_{protocol}_seed0.csv
```

This fits S1's probe (C chosen on validation), S3-LR, S3-Set, every temperature, every
threshold at 1% and 5% FPR, and both fusions. AEROBLADE is training-free; only its
threshold is fitted.

Two lines to check per protocol:

- `S3-Set: val AUC ... (permutation delta N.NNe-NN)` — the invariance is re-asserted
  **after** training, not only at initialisation. The P3 plug-in experiment depends on
  it, and an optimiser can break a property that held at init.
- `f_stack fitted on val_<protocol> over [...]` — the list of streams that entered the
  fusion. A stream missing here is a stream whose validation rows did not align, and
  the log names it.

Everything written in this step carries `fitted_on`. `FittedFusion.score` raises if it
is ever applied to that split.

---

## Step 2 — score the panel. Once.

```python
for protocol, seed in [("p_a_sd1", 0), ("p_a_sd1", 1), ("4ch_only", 0), ("full_bank", 0)]:
    !cd {DBP} && python scripts/p2_score.py --protocol {protocol} --seed {seed} \
        --s2-scores reports/p2_s2_scores_{protocol}_seed{seed}.csv
```

**Single shot.** There is no flag in `p2_score.py` that changes a model, because the
moment a panel score can be re-run with a different setting after it has been seen, the
panel stops being a test set. A re-run is permitted for a crash and nothing else, and
must be recorded in the log with the reason.

Check `coverage` in the summary: a stream with far fewer finite scores than the panel
size has a cache that does not cover the panel, and its slice numbers would be computed
on a different set of images from everyone else's — which is exactly what the shared
panel exists to prevent.

`in_bank` is filled from the run config, per row. An E7 subset name that does not map to
a bank member raises rather than defaulting to "unseen", because defaulting would
inflate every generalisation claim in the thesis.

---

## Step 3 — the pre-registered analysis and the verdict

```python
!cd {DBP} && python scripts/p2_analyze.py \
    --scores reports/p2/scores_p_a_sd1_0.csv --heads reports/p2_heads_p_a_sd1.json
```

Outputs `metrics.csv`, `complementarity.json`, `go_no_go.json`, `headline.json`.

The script implements `docs/P2_PREREGISTRATION.md` and has no flag that changes the
criterion, the primary metric, or how the best single stream is chosen — that is read
out of the validation artefacts, where it was fixed before any panel score existed.

```
VERDICT: GO | NO-GO
  M1: f_stack X.XXXX vs <stream> Y.YYYY -> +Z.ZZ points
  M2: f_stack X.XXXX vs <stream> Y.YYYY -> +Z.ZZ points
```

**Report whichever verdict comes out, unchanged.** A NO-GO is a finding about the
streams and sends the thesis to a deployment-oriented study of the best single stream
plus calibration. It is not a failure and it is not re-run until it turns over.

Then the secondary protocols, which are reported but do not gate:

```python
for protocol in ("4ch_only", "full_bank"):
    !cd {DBP} && python scripts/p2_analyze.py \
        --scores reports/p2/scores_{protocol}_0.csv --heads reports/p2_heads_{protocol}.json
```

---

## Step 4 — the reporting rules (D-044) are not optional

Before any table leaves this session, confirm `metrics.csv` carries its
`trivial_baseline` column populated for:

| slice | what must appear beside every number |
|---|---|
| E7 / FLUX.1 | `4:2:0 → fake` scores 100.00% (lift +49.85); quality +14.35 |
| E4 matched | quality separability +11.39 — ~61% reachable without a generative artefact |
| E3 / Nano-Banana2 | residual +6.47 after matching |

And confirm FLUX.1 is reported as **two numbers, standard / luma-only**, with the
luma-only one being what entered M1 and M2. An unchecked FLUX.1 number is not claimable.

---

## Step 5 — persist

```python
import csv, json, pathlib, shutil
R = DBP / "reports"

for protocol, seed in [("p_a_sd1", 0), ("p_a_sd1", 1), ("4ch_only", 0), ("full_bank", 0)]:
    f = R / "p2" / f"scores_{protocol}_{seed}.csv"
    n = sum(1 for _ in csv.DictReader(f.open(encoding="utf-8"))) if f.is_file() else 0
    print(f"  scores {protocol}/{seed}: {n} rows{'' if n == 12150 else '   <== expect 12150'}")

for name in ("metrics.csv", "complementarity.json", "go_no_go.json", "headline.json"):
    f = R / "p2" / name
    print(f"  {name:24s} {'OK' if f.is_file() else 'MISSING'}")
    assert f.is_file(), f"{name} missing -- the analysis did not complete"

g = json.loads((R / "p2" / "go_no_go.json").read_text(encoding="utf-8"))
print(f"\nVERDICT: {g['verdict']}  (passes on: {g['passes_on'] or 'neither'})")
print(f"  best single stream: {g['best_single_stream']} -- {g['best_single_chosen_on']}")

flagged = [r["slice"] for r in csv.DictReader((R / "p2" / "metrics.csv").open(encoding="utf-8"))
           if r["trivial_baseline"]]
print(f"  slices carrying a trivial-rule baseline: {sorted(set(flagged))}")
assert flagged, "D-044: no slice carries its baseline -- the tables would be unreadable"

shutil.make_archive("/kaggle/working/p2_results", "zip", str(R))
!du -sh /kaggle/working
```

**Save Version with output saving on**, wait for it to finish, and download
`p2_results.zip` to `F:\thesis-2\kaggle_raw\` — never into `REPO_ROOT` (I-032).

---

## Pre-registration §8 — the FLUX.1 pair, and what enters the gate

`p2_score.py` scores the luma companion as its own view, `e7_dda_coco_luma`, joining
each row's metadata from its parent E7 row so the subset stays `FLUX.1`. It needs both
S1 caches, `panel` and `panel_luma`; if the second is missing it warns and the luma rows
carry no S1 score.

`p2_analyze.py` then **substitutes** the slices: `e7_dda_coco/FLUX.1` is reported in
`metrics.csv` but does not enter M1 or M2, and `e7_dda_coco_luma/FLUX.1` takes its place.
The log says so explicitly:

```
§8 substitution: e7_dda_coco/FLUX.1 -> e7_dda_coco_luma/FLUX.1 (in M1 and M2)
reported but outside the gate (§8): ['e7_dda_coco/FLUX.1']
```

If the luma companion was never scored, the script **refuses to produce a verdict**
rather than averaging over one slice fewer — §8 says an unchecked FLUX.1 number is not
claimable, and a gate quietly computed without it would be exactly that.

## Exact duplicates (the E4 caveat)

Pass the frozen panel list so duplicates are measured rather than quoted:

```python
!cd {DBP} && python scripts/p2_analyze.py \
    --scores reports/p2/scores_p_a_sd1_0.csv \
    --heads reports/p2_heads_p_a_sd1.json \
    --panel-hashes reports/p2_panel_images_frozen.csv
```

Two outputs:

* `metrics.csv` — the primary numbers, **as distributed**, each affected slice carrying
  a `caveat` column naming the number of byte-identical groups and rows;
* `metrics_deduplicated.csv` — **analysis only**, each duplicate group counted once.

E4 is the slice this exists for: 25 byte-identical groups over 50 of its 1,500 panel
rows, already recorded by P1 as part of its 77 near-duplicate groups. An evaluation set
is used as distributed, so the primary number keeps them and the sensitivity row says
what they were worth.

---

## Checklist

- [ ] Pre-registration commit hash recorded, and it **predates** the scores.
- [ ] Heads fitted for three protocols; S3-Set invariance re-asserted after training.
- [ ] Panel scored **once** per configuration; coverage checked per stream.
- [ ] `go_no_go.json` written; verdict reported unchanged.
- [ ] `metrics.csv` carries the trivial-rule baselines; FLUX.1 reported as a pair.
- [ ] Results zip downloaded outside `REPO_ROOT`.
- [ ] `§8 substitution` line present in the analyze log; the gate's unseen slices list `e7_dda_coco_luma/FLUX.1` and **not** `e7_dda_coco/FLUX.1`.
- [ ] `metrics_deduplicated.csv` written, and every E4 row in `metrics.csv` carries its duplicate caveat.
