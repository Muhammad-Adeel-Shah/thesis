# Kaggle run — P2-B0/B1: complete the inputs, draw the split and the panel

**Purpose.** Prove the attached corpus is the P1 corpus, rebuild the truncated reals
deterministically, then draw the split and the evaluation panel. **CPU only** — nothing
here needs a GPU, so it does not spend the accelerator quota.

**Why a rebuild rather than an upload.** The Kaggle dataset copy of the reals held
**1,314** of 4,100 PNGs. The complete set is `p1_reals.zip` (1.02 GB) locally, but
uploading it over a slow link is an hour of wall clock for something the build script
reproduces from COCO, which is attached, and DIV2K, which downloads in the session. The
rebuild is
verified by **SHA-256 against the committed manifests**, so "deterministic" is a
checked claim rather than an assumption.

**Inputs to attach**

| input | where from |
|---|---|
| code dataset | this repo at the commit `Step 0` prints |
| `notebook3a585d800a` output (version 353334139) | *Add Input → Your Work* — the 49,200 pairs and the generations |
| the fix-notebook output | the current `gen_matched` (2,488) |
| `awsaf49/coco-2017-dataset` | `train2017` + `annotations/instances_train2017.json` |
| **DIV2K_train_HR** | **not attached — downloaded inside the session** (Step 1a). 800 photographs, `0001.png`-`0800.png` |

---

## Step 0 — stage and verify the code

Use the staging cell from `kaggle_run_p1_build.md`, then:

```python
print("commit :", (DBP / "COMMIT.txt").read_text().strip())
for name in ("p2_verify_inputs.py", "p2_build_splits.py", "p2_build_panel.py"):
    print(f"{name:24s}", (DBP / "scripts" / name).is_file())
assert not str(DBP).startswith("/kaggle/input"), "run from the staged copy, not the mount"
```

All three must print `True`. If any is `False`, the attached code dataset predates P2.

---

## Step 0b — stage the uploaded manifests **without** clobbering the tracked ones

The code dataset already carries the three manifests that matter:
`p1_reals_manifest_t_real.csv` (83 rows flagged), `p1_reals_manifest_t_ctrl.csv` (6) and
`p1_gen_matched_manifest.csv` (2,488 rows). Those are the post-D-039 copies, and their
SHA-256s match what the rescan and the matcher recorded.

Older copies of the same file names are floating around in the Kaggle inputs -- any
`p1_reports` folder from a build-notebook output holds the **pre-D-039** versions, with
0 rows flagged and 2,475 `gen_matched` rows. A `for src in rglob("p1_*manifest*.csv"):
copy(src)` loop will happily overwrite the good ones with those, and nothing downstream
will say so until the gate fails three cells later. So the copy is explicitly guarded:

```python
import csv, pathlib, shutil

REPORTS = DBP / "reports"
TRACKED = {                                  # shipped in the code dataset; never replace
    "p1_reals_manifest_t_real.csv",
    "p1_reals_manifest_t_ctrl.csv",
    "p1_gen_matched_manifest.csv",
}

staged, protected = 0, []
for src in sorted(pathlib.Path("/kaggle/input").rglob("p1_*manifest*.csv")):
    if src.name in TRACKED:
        protected.append(str(src))
        continue
    shutil.copy2(src, REPORTS / src.name)
    staged += 1

print(f"staged {staged} manifest(s); refused to overwrite {len(protected)} tracked one(s)")
for path in protected:
    print("   skipped:", path)
```

Then prove the tracked three are the right ones **before** anything reads them:

```python
def flagged(name):
    rows = list(csv.DictReader((REPORTS / name).open(encoding="utf-8")))
    return len(rows), sum(1 for r in rows if r.get("excluded_reason"))

for name, rows_want, flag_want in (
    ("p1_reals_manifest_t_real.csv", 2500, 83),
    ("p1_reals_manifest_t_ctrl.csv", 1600, 6),
):
    rows, flags = flagged(name)
    print(f"{name:32s} {rows:5d} rows, {flags:3d} flagged (expect {rows_want}, {flag_want})")
    assert (rows, flags) == (rows_want, flag_want), \
        f"{name} is the pre-D-039 copy -- restore it from the code dataset"

gm = sum(1 for _ in csv.DictReader(
    (REPORTS / "p1_gen_matched_manifest.csv").open(encoding="utf-8")))
print(f"{'p1_gen_matched_manifest.csv':32s} {gm:5d} rows (expect 2488)")
assert gm == 2488, "this is the stale 2,475-row manifest"
```

If an assertion fires, re-extract the code dataset over `reports/` -- the staging cell in
Step 0 does exactly that -- and run this cell again.

---

## Step 1a — DIV2K (T-ctrl's source)

DIV2K is **not** a Kaggle dataset here; P1 fetched it into the session and so does P2.
Allow-listed host `data.vision.ee.ethz.ch` (§6a), download on Kaggle only, never to the
local machine (condition (a)). The set is *academic research only* (§6c,
`docs/data_cards/t_ctrl_div2k.md`).

It is **DIV2K_train_HR**, 800 photographs named `0001.png`-`0800.png` — not `valid_HR`,
which holds 100. T-ctrl's 1,600 rows are two non-overlapping crops of each of the 800,
and `reports/kaggle_20260926/p1_reals_manifest_t_ctrl.json` records the source root P1
used as `/tmp/div2k/DIV2K_train_HR`.

```python
import pathlib, subprocess
div2k = pathlib.Path("/tmp/div2k")
div2k.mkdir(parents=True, exist_ok=True)
url = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
archive = div2k / "DIV2K_train_HR.zip"
if not archive.exists():
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(archive), url], check=True)
print("bytes:", archive.stat().st_size, "(expected 3530603713)")
subprocess.run(["unzip", "-q", "-o", str(archive), "-d", str(div2k)], check=True)
DIV2K = div2k / "DIV2K_train_HR"
print(sum(1 for _ in DIV2K.glob("*.png")), "PNGs (expect 800)")
```

`/tmp`, not `/kaggle/working`: ~3.5 GB that is not worth persisting.

---

## Step 1b — rebuild the reals, deterministically

The flags are `--set`, `--root` and `--limit`, and `--limit` counts **rows, not source
images**: t_ctrl's 1,600 rows come from 800 photographs at two crops each. There is no
`--seed` flag — the script calls `seed_everything(2026, deterministic=True)` itself and
walks the source root in sorted order, which is what makes the rebuild reproducible.

```python
!cd {DBP} && python scripts/p1_build_reals.py --set t_real \
    --root {COCO}/train2017 --limit 2500 --crop 384
!cd {DBP} && python scripts/p1_build_reals.py --set t_ctrl \
    --root {DIV2K} --limit 1600 --crop 384 --crops-per-image 2
```

Then re-apply the greyscale flags (D-039) — the build writes a fresh manifest without
them. `--instances` is what produces the per-supercategory effect table:

```python
!cd {DBP} && python scripts/p1_rescan_greyscale.py --set-id t_real \
    --instances {COCO}/annotations/instances_train2017.json
!cd {DBP} && python scripts/p1_rescan_greyscale.py --set-id t_ctrl
```

Expect **83** flagged in t_real and **6** in t_ctrl. A zero means the rescan did not
run, and 89 greyscale reals would go straight back into training.

---

## Step 2 — verify, and refuse to continue if anything differs

```python
!cd {DBP} && python scripts/p2_verify_inputs.py --root {P1} --manifests reports \
    --out reports/p2_input_verification.json
```

This is the gate for the whole phase. It checks counts, per-member pair counts, the
exclusion flags, **every PNG's SHA-256** against `sha256` in the reals manifests and
`jpg_sha256` in `p1_gen_matched_manifest.csv`, and the twelve pair manifests against
`reports/p1_pairs_manifest_digest.json`.

Expected: `VERDICT: PASS`.

| assertion | value |
|---|---|
| reals | 4,100 (2,500 + 1,600) |
| pairs | **49,200**, every member at 4,100 |
| generations | 2,488 |
| `gen_matched` | **2,488** — a count of **2,475 is the stale set and a hard fail** |
| flags | 83 (t_real) + 6 (t_ctrl) |

`--hash-limit N` exists for a smoke run. If you use it, **say so in the report**: it is
not a full verification, and the script records the limit in its JSON so the claim
cannot be overstated later.

Any mismatch stops the run. Do not continue past a `FAIL`.

---

## Step 3 — draw the split (once, ever)

```python
!cd {DBP} && python scripts/p2_build_splits.py \
    --instances {COCO}/annotations/instances_train2017.json \
    --pairs-manifest reports/p1_pairs_manifest.csv
```

80/20, seed 2026, grouped by `source_image_id`, stratified by COCO supercategory
(D-046). Watch for three lines:

- `strata: {...}` — the supercategory partition, including `none` for the DIV2K groups;
- `no content crosses the split: N distinct content ids` — the assertion the module
  exists for. It covers reals, pairs and generations **together**, because a real in
  train and its own reconstruction in validation is exactly what a per-set check misses;
- `groups: N train / M val`.

Generations follow the **caption's** COCO image, never `source_real_id`. Those are
different pictures, and using the latter would put a generation in validation while the
photograph its content came from sits in training.

`reports/p2_splits.csv` and `p2_splits_summary.json` are **committed**. Every threshold,
temperature and stacker coefficient in P2 is fitted on the validation side of this file,
so re-drawing it later silently invalidates all of them.

---

## Step 4 — Chameleon's pipeline-matched view

The derived JPEGs are not in any archive; rebuild them as in `kaggle_run_p1_close.md`
Step 4 (both filters). This view is **PRIMARY** for every Chameleon claim (D-043).

---

## Step 5 — draw the panel

```python
for set_id, manifest in [
    ("e7_dda_coco",            "p1_eval_e7_dda_coco_manifest.csv"),
    ("e8_evalgen",             "p1_eval_e8_evalgen_manifest.csv"),
    ("e3_matched",             "p1_eval_e3_raw_jpeg_matched_manifest.csv"),
    ("e4_wildrf_matched",      "p1_eval_e4_wildrf_test_jpeg_matched_manifest.csv"),
    ("e5_chameleon_pipeline",  "p1_eval_e5_chameleon_pipeline_matched_manifest.csv"),
]:
    !cd {DBP} && python scripts/p2_build_panel.py --set-id {set_id} --manifest reports/{manifest}
```

Expected panel: **12,150 images**, the size the GPU budget was computed against.

| view | panel |
|---|---|
| E7 | 7 × 500 = 3,500 |
| E8 | 5 × 250 = 1,250 |
| E3 matched | 1,500 |
| E4 matched | 1,500 |
| E5 pipeline-matched | 2 × 2,000 = 4,000 |
| T-ctrl diagnostic | 2 × 200 = 400 |

Note the E3 and E4 manifests are the **matched** views, never the raw ones, and E5 is
the **pipeline-matched** one, not the compression-only match.

---

## Step 5b — the image lists

These three files commit the whole GPU budget, so they are built by a tested script
rather than by a notebook cell.

```python
!cd {DBP} && python scripts/p2_build_image_lists.py \
    --root /kaggle/working/p1 \
    --pairs-manifest reports/p1_pairs_manifest.csv
```

| list | expect | what it costs |
|---|---|---|
| `p2_s1_images.csv` | ~55,800 | one DINOv3 forward each -- cheap, so it is exhaustive |
| `p2_panel_images.csv` | **12,150** | the shared panel (D-048) |
| `p2_probe_images.csv` | ~**18,150** | **twelve reconstructions + twelve LPIPS each** |

Every path is resolved and verified to exist **here**, relative to `--root`, which is why
a missing file costs this CPU cell rather than surfacing nine GPU-hours into Session B.
`p1_gen_matched_manifest.csv`'s absolute paths are converted to relative on the way in.

The script also writes `p2_panel_t_ctrl_diagnostic.csv`. That panel has no evaluation
manifest to draw from -- it is 200 lossless DIV2K reals and 200 reconstructions **of
those same reals**, selected on the recorded source rather than on an id prefix -- so it
is drawn here, through the same seeded sampler and the same `PANEL_CAPS` entry as every
other panel. That is what keeps the panel total at 12,150.

**If `p2_probe_images.csv` comes out materially above 18,150, stop and report it**: the
9.63 GPU-h S3 estimate was computed against that number. The levers are
`--train-reals`, `--train-fakes-per-protocol` and `--val-cap`, and lowering them is a
budget decision, not a default.

Read back from the log:

- `corpus rows resolved: N` -- then `0` unresolved, or the run exits 1 and names them;
- one line per protocol, `p_a_sd1` / `4ch_only` / `full_bank`, with how many members it
  was allowed and how many fakes it drew;
- `N subset row(s) are already in the panel and are not listed twice` -- T-ctrl rows are
  both panel rows and training reals, and a duplicate would be reconstructed twice and
  weighted twice in every metric.

---

## Step 6 — persist

**Count before you zip.** An archive of a half-finished session looks exactly like an
archive of a finished one, and the difference only shows up three sessions later.

```python
import csv, json, pathlib, collections, shutil
R = DBP / "reports"

expect = {
    "p2_splits.csv": 0,                       # printed, not asserted: see below
    "p2_panel_e7_dda_coco.csv": 3500,
    "p2_panel_e8_evalgen.csv": 1250,
    "p2_panel_e3_matched.csv": 1500,
    "p2_panel_e4_wildrf_matched.csv": 1500,
    "p2_panel_e5_chameleon_pipeline.csv": 4000,
    "p2_panel_t_ctrl_diagnostic.csv": 400,
}
panel_total = 0
for name, n in expect.items():
    path = R / name
    rows = sum(1 for _ in csv.DictReader(path.open(encoding="utf-8"))) if path.is_file() else 0
    ok = "" if (n == 0 or rows == n) else f"   <== expected {n}"
    print(f"{name:40s} {rows:6d}{ok}")
    if name.startswith("p2_panel_"):
        panel_total += rows

print(f"\npanel total: {panel_total} (expect 12150)")
assert panel_total == 12150, "the panel is incomplete -- do not zip a partial session"

v = json.loads((R / "p2_input_verification.json").read_text(encoding="utf-8"))
print("input verification:", v["verdict"], "| hash limit:", v["hash_limit"])
assert v["verdict"] == "PASS", v["failures"]

sp = json.loads((R / "p2_splits_summary.json").read_text(encoding="utf-8"))
print("split:", sp["split"]["totals"], "| crossing:", sp["leakage"]["crossing"])
assert sp["leakage"]["crossing"] == [], "content crosses the split"

shutil.make_archive("/kaggle/working/p2_inputs_reports", "zip", str(R))
!du -sh /kaggle/working
```

**Save Version → Quick Save with output saving on.** Wait for the version to finish
before closing the tab; a draft session that ends early keeps none of this. Download
`p2_inputs_reports.zip` to `F:\thesis-2\kaggle_raw\` — never into `REPO_ROOT`, which is
a protected-set change (I-032).

---

## Checklist

- [ ] Step 0 printed the expected commit and three `True`s.
- [ ] Tracked manifests intact after staging: **83** / **6** flagged, **2,488** gen_matched rows.
- [ ] DIV2K_train_HR downloaded: **800 PNGs**, archive 3 530 603 713 bytes.
- [ ] Reals rebuilt (`--set`/`--root`/`--limit`, t_ctrl at `--limit 1600`); rescan shows **83** and **6** flagged.
- [ ] `p2_verify_inputs.py` → **PASS**, with the hash limit stated if one was used.
- [ ] `gen_matched` is **2,488**, not 2,475.
- [ ] Split drawn; **no content crosses**; `p2_splits.csv` saved.
- [ ] Chameleon pipeline-matched view rebuilt, both filters.
- [ ] Panel drawn: **12,150** rows across six views.
- [ ] Image lists written; `p2_probe_images.csv` at ~**18,150**, 0 unresolved rows.
- [ ] Output saved and the zip downloaded outside `REPO_ROOT`.
