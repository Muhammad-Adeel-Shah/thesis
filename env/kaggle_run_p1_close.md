# Kaggle run — P1 closing session

**Purpose.** Produce the four things P1 needs before it can be merged: a count of the
pairs actually on disk, a re-measured decoder-bank CSV saved as an artefact, the
Chameleon pipeline-matched view, and a persisted output. One session, **CPU only except
Step 3**, which needs a GPU.

**Inputs.** The code dataset (commit printed by Step 0), the build notebook output
`notebook3a585d800a` (version 353334139) via *Add Input → Your Work*, the fix notebook
output (the current `gen_matched` and the manifests carrying `excluded_reason`), and the
Chameleon dataset.

**Accelerator: GPU T4 ×2** — Step 3 re-measures real weights. Everything else is CPU.

---

## Step 0 — stage and verify

Use the staging cell from `kaggle_run_p1_build.md`. Before anything else:

```python
print("commit    :", (DBP / "COMMIT.txt").read_text().strip())
src = (DBP / "scripts" / "p1_build_eval.py").read_text()
print("pipeline  :", "build_pipeline_matched_view" in src)
print("separable :", "parameter_separability" in src)
assert not str(DBP).startswith("/kaggle/input"), "run from the staged copy, not the mount"
```

Both must print `True`. If either is `False` the attached code dataset is an old version,
and nothing below will do what it says.

Copy the manifests in — **from the fix notebook's output, not the build's**, because only
those carry `excluded_reason`:

```python
import shutil, pathlib
for name in ("p1_reals_manifest_t_real.csv", "p1_reals_manifest_t_ctrl.csv",
             "p1_gen_manifest.csv", "p1_gen_matched_manifest.csv", "p1_throughput.csv"):
    hits = sorted(pathlib.Path("/kaggle/input").rglob(name))
    assert hits, f"{name} not found"
    shutil.copy2(hits[-1], DBP / "reports" / name)
    print("copied", hits[-1])

import csv
flagged = sum(1 for r in csv.DictReader(open(DBP / "reports" / "p1_reals_manifest_t_real.csv",
                                             encoding="utf-8")) if r.get("excluded_reason"))
print("t_real flagged:", flagged, "(expect 83 — if 0, this is the build's stale manifest)")
```

---

## Step 1 — count what is actually on disk

The manifests say 49,200 pair rows. This checks the images exist, which is a different
claim.

```python
import pathlib, collections
P1 = next(p for p in pathlib.Path("/kaggle/input").rglob("outputs/p1") if (p / "pairs").is_dir())
print("root:", P1)

counts = {}
for name, pattern in (("reals", "*.png"), ("pairs", "*.png"),
                      ("generated", "*.png"), ("gen_matched", "*.jpg")):
    d = P1 / name
    counts[name] = sum(1 for _ in d.rglob(pattern)) if d.exists() else 0
print(counts)

per_member = {m.name: sum(1 for _ in m.glob("*.png"))
              for m in sorted((P1 / "pairs").iterdir()) if m.is_dir()}
for key, n in per_member.items():
    print(f"  {key:16s} {n:6d}{'' if n == 4100 else '   <== SHORT'}")

assert counts["reals"] == 4100, counts
assert counts["pairs"] == 49200, f"expected 49,200, found {counts['pairs']}"
assert counts["generated"] == 2488, counts
print("\nusable after the greyscale exclusion: 48,132 of 49,200 (89 reals x 12 members)")
```

If a member is short, its per-member manifest is ahead of the files. Delete **that
member's** manifest and re-run `p1_build_pairs.py --members <key>`; the resume logic
rebuilds only what is missing.

---

## Step 2 — the fp16 audit (read-only)

The review flagged that `realesrgan_x4` measured 59.862/59.554 dB in P0 and 44.94 dB in
P1. Both are real; they are different measurements (I-040). What matters is what the
build actually did:

```python
import csv, collections
for key in ("realesrgan_x4", "movq_k22", "sdxl"):
    hits = sorted(pathlib.Path("/kaggle/input").rglob(f"p1_pairs_manifest_{key}.csv"))
    rows = list(csv.DictReader(hits[-1].open(encoding="utf-8")))
    print(f"{key:16s} rows={len(rows)}  precision={dict(collections.Counter(r['precision'] for r in rows))}")
```

**Expected: `{'fp32': 4100}` for all three.** That is the conservative outcome and needs
no rebuild. If any shows `fp16`, delete that member's manifest and rebuild it in fp32
before Step 3.

---

## Step 3 — re-measure the bank and keep the artefact (GPU)

P1's 13-member gate passed three times, but that session's CSV was never saved, so the
result exists only as console output. This produces the artefact.

```python
!cd {DBP} && python scripts/p0_bank_smoke_test.py --device cuda --crops 512,1024 \
    --members sd15,sd21,sd_ft_ema,sd_ft_mse,sdxl,sdxl_fp16fix,sd35,flux,dcae,realesrgan_x4,flux2_klein,qwen_image,movq_k22 \
    --compare-to reports/kaggle_20260920/p0_bank_smoke.csv
```

Expect `26/26 rows OK`, `regression check PASSED` over 20 pairs, and
`byte-identical decoders: ['sd21', 'sd_ft_mse']` and nothing else. Then keep both files:

```python
import datetime, shutil, pathlib
stamp = datetime.date.today().strftime("%Y%m%d")
dest = DBP / "reports" / f"kaggle_{stamp}"
dest.mkdir(parents=True, exist_ok=True)
for name in ("p0_bank_smoke.csv", "p0_decoder_identity.json", "p1_throughput.csv"):
    src = DBP / "reports" / name
    if src.is_file():
        shutil.copy2(src, dest / name)
        print("kept", name, src.stat().st_size, "bytes")
```

---

## Step 4 — Chameleon's pipeline-matched view (D-043)

The compression-only matched view does **not** remove Chameleon's real confound: every
real is 500 px wide because it was thumbnailed, so `width == 500 → real` still scores
100.00%. This view puts the fakes through the same pipeline.

Extract Chameleon as in `kaggle_run_p1_eval.md` Step 6, then:

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e5_chameleon \
    --root {E5_ROOT} \
    --real-glob "test/0_real/*" --fake-glob "test/1_fake/*" \
    --jpeg-match \
    --pipeline-match-width 500 --pipeline-match-quality 75 \
    --pipeline-match-subsampling 4:2:0 --pipeline-match-resample lanczos \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

Then the sensitivity check, which differs only in the filter:

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e5_chameleon_bicubic \
    --root {E5_ROOT} \
    --real-glob "test/0_real/*" --fake-glob "test/1_fake/*" \
    --pipeline-match-width 500 --pipeline-match-quality 75 \
    --pipeline-match-subsampling 4:2:0 --pipeline-match-resample bicubic \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

Read out of `p1_eval_e5_chameleon_summary.json` → `pipeline_matched_view`:

- `width_rule_before.accuracy` — expect **1.0**, the confound as distributed;
- `width_rule_after.accuracy` — must equal `width_rule_after.baseline`, meaning width is
  worth nothing once both classes share it;
- `parameter_separability.separable` — expect **false**; every image is now q75 4:2:0;
- `resized` ≈ 11,169 and `reals_untouched` 14,863.

**This view is PRIMARY for every Chameleon claim.** The compression-only matched view
becomes secondary, and the raw view is reported only to show the confound. If the two
filters disagree materially, say so — that is what the sensitivity check is for.

This step writes ~11,000 JPEGs and takes roughly 15 minutes per filter.

---

## Step 4b — verify the Chameleon chain (optional, 5 minutes)

The archive's provenance is now documented (I-031): posted by `qwqhjcxczx` in AIDE issue
#1 on 2025-01-02, Drive id `1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH`. This confirms the file in
hand is that file. On Kaggle only, with `gdown` (§6a(c)):

```python
!python -m pip install --quiet "gdown==5.2.0"
import hashlib, pathlib, subprocess
dest = pathlib.Path("/tmp/cham_check"); dest.mkdir(parents=True, exist_ok=True)
archive = dest / "Chameleon.zip"
subprocess.run(["gdown", "--id", "1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH", "-O", str(archive)],
               check=True)
h = hashlib.sha256()
with archive.open("rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""):
        h.update(chunk)
print("bytes :", archive.stat().st_size, "(expected 2683359341)")
print("sha256:", h.hexdigest())
print("chain :", "CONFIRMED" if h.hexdigest() ==
      "a7cdadfe35287dd3e81b5c35e7cce5437e68ba3b4cdbb833ac0686b990c2fba6" else "DIFFERS")
```

`CONFIRMED` closes the chain from the issue to the evaluated pixels. `DIFFERS` means the
file was re-uploaded or altered, and the provenance stops at "received from the user" —
report it either way, and delete `/tmp/cham_check` afterwards.

---

## Step 5 — persist

```python
import shutil
shutil.make_archive("/kaggle/working/p1_close_reports", "zip", str(DBP / "reports"))
!du -sh /kaggle/working
```

Then **Save Version → Quick Save with output saving on**, and download
`p1_close_reports.zip` to `F:\thesis-2\kaggle_raw\` — never into `REPO_ROOT`, which is a
protected-set change (I-032).

---

## Checklist

- [ ] Step 0 printed the expected commit and both `True`s; t_real shows **83** flagged.
- [ ] Pairs on disk: **49,200**, every member at 4,100.
- [ ] `realesrgan_x4`, `movq_k22`, `sdxl` all recorded **fp32**; no rebuild needed.
- [ ] `regression check PASSED`; `p0_bank_smoke.csv` and `p0_decoder_identity.json` saved
      under `reports/kaggle_<date>/`.
- [ ] Chameleon pipeline-matched view built, `width_rule_after` == baseline, both filters.
- [ ] Output saved and the reports zip downloaded outside `REPO_ROOT`.
