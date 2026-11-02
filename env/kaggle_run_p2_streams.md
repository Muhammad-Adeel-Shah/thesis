# Kaggle run — P2-B3/B4/B5: train the three streams

**Accelerator: GPU T4 ×2.** This is where the GPU budget goes. Budget after the raise
to **32 GPU-h** (≈ 16 Kaggle session-hours on T4 ×2):

| item | GPU-h | T4×2 session-h |
|---|---:|---:|
| S1 feature cache (33,909 corpus images, once) | 0.49 | 0.24 |
| S1 scoring: panel + five full views + 1,000 luma | 1.20 | 0.60 |
| **S2 LoRA training** — 4 runs × 2.25 h allowance | 9.00 | 4.50 |
| S2 inference: panel ×4 + full views ×4 + luma ×4 | 4.81 | 2.40 |
| **S3 probe features** (12 members × 19,011, once) | 10.09 | 5.04 |
| subtotal | 25.59 | 12.79 |
| +20% (LPIPS ×12, I/O, restarts) | 5.12 | 2.56 |
| **TOTAL** | **30.70** | **15.35** |

Against the 32 GPU-h cap that leaves **+1.30 GPU-h**. Rates are the ones the committed
table was built from; the counts are measured by `p2_build_image_lists.py` in Step 0b.

Two approved changes moved these numbers, in opposite directions:

* **T-ctrl leaves every training list** (P2 prompt B1: diagnostic-only). 20,722 rows —
  1,594 control reals and their 19,128 reconstructions — are withheld, so the S1 cache
  drops from 54,631 to **33,909**. The 400-row diagnostic panel is unchanged and is now
  fully out-of-sample, which is what makes the JPEG-reliance check mean anything.
* **The luma companion is scored** (pre-registration §8). 1,000 rows join the probe
  list, and because the S3 subset can no longer draw T-ctrl rows it redraws from
  `t_real` and reaches its full validation cap, so the probe list goes 17,941 →
  **19,011**: +861 over the 18,150 the original estimate assumed.

S3 at 10.09 GPU-h is still under the 11 GPU-h stop-and-report line in Step 1. Two numbers in it are derived
rather than measured — the S2 step time and the LPIPS cost — and both are measured
before their full run, below.

**Check the quota meter after the first GPU session and report what it actually
charged.** Kaggle counts session time, so a T4 ×2 session should consume about half the
GPU-hours listed. That is the assumption the raise to 32 rests on; it needs confirming
once, not trusting.

---

**Inputs to attach.** Four, and no evaluation corpus among them — that is what
`kaggle_run_p2_panel.md` bought:

| input | what for |
|---|---|
| the code dataset | the commit Step 0 prints |
| `dbp-p2-panel` (**private**) | the 12,150 frozen panel images + the luma companion |
| `adeelshah29/p1-data` + the pairs notebook output + Session A's `gm` | the corpus root |
| Session B's own earlier output, on a resumed run | the feature caches and checkpoints |

---

## Step 0 — stage, then assert the panel BEFORE any GPU work

Nothing below touches the accelerator until these assertions pass. A panel that is not
the panel invalidates every number in P2, and finding that out after nine GPU-hours is
the expensive way to find it out.

```python
import collections, csv, hashlib, json, pathlib
R = DBP / "reports"
print("commit :", (DBP / "COMMIT.txt").read_text().strip())

splits = R / "p2_splits.csv"
assert splits.is_file(), "run kaggle_run_p2_inputs.md first"
print("split rows:", dict(collections.Counter(
    r["split"] for r in csv.DictReader(splits.open(encoding="utf-8")))))

# --- the panel: count, then digest -----------------------------------------
PANEL = next(p for p in pathlib.Path("/kaggle/input").rglob("e7_dda_coco")
             if p.is_dir()).parent
print("panel root:", PANEL)

rows = list(csv.DictReader((R / "p2_panel_images.csv").open(encoding="utf-8")))
print("panel rows:", len(rows), "(expect 12150)")
assert len(rows) == 12150, "this is not the 12,150-row panel the budget describes"

recorded = json.loads((R / "p2_panel_digest.json").read_text(encoding="utf-8"))
digest = hashlib.sha256(
    "\n".join(sorted(f"{r['image_id']}\t{r['sha256']}" for r in rows)).encode()
).hexdigest()
print("digest listed  :", recorded["panel_digest"])
print("digest computed:", digest)
assert digest == recorded["panel_digest"], "the panel list does not match its digest"

missing = [r["relative_path"] for r in rows if not (PANEL / r["relative_path"]).is_file()]
print("missing files  :", len(missing), missing[:3])
assert not missing, "the panel dataset is incomplete"

# --- the image lists must carry the root column ----------------------------
for name in ("p2_s1_images.csv", "p2_probe_images.csv"):
    listed = list(csv.DictReader((R / name).open(encoding="utf-8")))
    assert "root" in listed[0], f"{name} predates the root column -- re-run Step 4 of the panel runbook"
    print(f"{name:24s} {len(listed):6d} rows, roots "
          f"{dict(collections.Counter(r['root'] for r in listed))}")
```

The digest is recomputed from the list rather than trusted, and every path is checked to
exist. Those two together are what "this is the panel" means.

---

## Step 0b — rebuild the image lists (CPU, before the accelerator matters)

The lists in Session A's output predate two approved changes: T-ctrl was in every
training list, and the luma companion was in none of them. Rebuilding is a CPU step and
must happen before Step 1, because the probe list is what Step 1 calibrates against.

```python
!cd {DBP} && python scripts/p2_build_image_lists.py \
    --root {P1} \
    --pairs-manifest reports/p1_pairs_manifest.csv \
    --panel-list reports/p2_panel_images_frozen.csv \
    --luma-list reports/p2_panel_images_luma.csv
```

`--panel-list` takes the **frozen** list, not `p2_panel_images.csv`: this script rewrites
that filename with a `root` column in place of `sha256`, so the hash-bearing copy is the
input and the resolvable copy is the output.

Expect, exactly:

```
diagnostic-only (t_ctrl): 20722 row(s) withheld from every training list (1594 real, 19128 reconstruction)
luma companion: 1000 row(s) ... added to the probe list
p2_s1_images.csv            33909 rows
p2_panel_images.csv         12150 rows
p2_probe_images.csv         19011 rows
p2_luma_images.csv           1000 rows
```

```python
import csv
R = DBP / "reports"
for name, want in (("p2_s1_images.csv", 33909), ("p2_panel_images.csv", 12150),
                   ("p2_probe_images.csv", 19011), ("p2_luma_images.csv", 1000)):
    n = sum(1 for _ in csv.DictReader((R / name).open(encoding="utf-8")))
    print(f"{name:24s} {n:6d} (expect {want}){'' if n == want else '   <== WRONG'}")
    assert n == want, name

# B1: no training list may contain a T-ctrl row.
ctrl = {r["image_id"] for r in csv.DictReader(
    (R / "p1_reals_manifest_t_ctrl.csv").open(encoding="utf-8"))}
def is_ctrl(i): return i in ctrl or ("__" in i and i.rsplit("__", 1)[0] in ctrl)
s1 = [r["image_id"] for r in csv.DictReader((R / "p2_s1_images.csv").open(encoding="utf-8"))]
print("t_ctrl rows in the S1 list:", sum(1 for i in s1 if is_ctrl(i)), "(expect 0)")
assert not any(is_ctrl(i) for i in s1)
```

If any count differs, stop: the GPU estimate above is computed against these numbers.

---

## Step 1 — S3 calibration: measure LPIPS before paying for it

The budget carries the cost of twelve LPIPS-vgg forwards per image inside its 20%
contingency, because it was never measured in P0 or P1. Measure it:

```python
!cd {DBP} && python scripts/p2_probe_features.py --images reports/p2_probe_images.csv \
    --root {P1} --panel-root {PANEL} --device cuda --members sd15 --calibrate 50
```

Read `projected_full_run_gpu_hours` out of `reports/p2_probe_timing_shard0.json` and
multiply by 12 members. If the projection exceeds **11 GPU-h**, stop and report before
running the full extraction — the ladder's first step is to cut the panel caps, and
that decision is the user's.

---

## Step 2 — S3 probe features, sharded over both GPUs

Two cells, one per GPU. The bank is fixed at test time, so this runs **once** and every
protocol's head reuses it — the single biggest saving in the budget.

```python
# GPU 0
!cd {DBP} && CUDA_VISIBLE_DEVICES=0 python scripts/p2_probe_features.py \
    --images reports/p2_probe_images.csv --root {P1} --panel-root {PANEL} \
    --device cuda --shard 0 --num-shards 2
```

```python
# GPU 1
!cd {DBP} && CUDA_VISIBLE_DEVICES=1 python scripts/p2_probe_features.py \
    --images reports/p2_probe_images.csv --root {P1} --panel-root {PANEL} \
    --device cuda --shard 1 --num-shards 2
```

Each member writes `outputs/p2/probe/p2_probe_<key>.npz` and skips a member already
done, so a session that dies part-way resumes rather than restarting.

Precision follows P1's recorded `fp16_build_allowed`: `movq_k22`, `sdxl` and
`realesrgan_x4` run **fp32**, the other nine fp16. The script logs which. That matters
because a training fake and a test image must be reconstructed under the same numerics.

**These PSNRs will not match the P1 pair manifests.** P1 built at crop 384; S3 probes at
320 (D-045). Different crop, different number — by design, not a regression.

---

## Step 3 — S1 feature cache

```python
!cd {DBP} && python scripts/p2_extract_s1.py --images reports/p2_s1_images.csv \
    --root {P1} --panel-root {PANEL} --device cuda --batch 32 --tag train
!cd {DBP} && python scripts/p2_extract_s1.py --images reports/p2_panel_images.csv \
    --root {P1} --panel-root {PANEL} --device cuda --batch 32 --tag panel
!cd {DBP} && python scripts/p2_extract_s1.py --images reports/p2_luma_images.csv \
    --root {P1} --panel-root {PANEL} --device cuda --batch 32 --tag panel_luma
```

The luma pass needs its **own tag**. `p2_extract_s1.py` names its output
`p2_s1_<backbone>_<tag>.npz` and keys features by image id; the luma rows carry
`<id>__luma` so they cannot collide, and `p2_score.py` reads both caches. Without this
third call the FLUX.1 luma slice has no S1 score and, by pre-registration §8, the
standard FLUX.1 number is not claimable.

`--panel-root` is not optional once a list contains `root=panel` rows: the resolver
raises rather than reading a panel path against the corpus root.

Confirm in the log: **4 register tokens**, **400 patch tokens** at crop 320, feature dim
**1536**. The register count is read from the model config — a hard-coded 0 would
average four register vectors into the patch mean and still produce a plausible-looking
feature.

The backbone is `facebook/dinov3-vitb16-pretrain-lvd1689m` at the P0 pin. It is
**gated**: `HF_TOKEN present: True` must appear, from Kaggle Secrets only.

---

## Step 4 — S2, four runs

Order matters: the primary protocol's two seeds first, so a budget overrun costs a
secondary protocol rather than the gate.

```python
for protocol, seed in [("p_a_sd1", 0), ("p_a_sd1", 1), ("4ch_only", 0), ("full_bank", 0)]:
    !cd {DBP} && python scripts/p2_train_s2.py --protocol {protocol} --seed {seed} \
        --root {P1} --pairs-manifest reports/p1_pairs_manifest.csv \
        --allowance-hours 2.25 --calibrate 20
```

Each run first times **20 real training steps** — forward, backward, optimiser, gradient
checkpointing on — and converts the 2.25 GPU-h allowance into a step ceiling. It logs:

```
measured X.XXXX s/step at batch 16 -> N steps inside 2.25 GPU-h (ceiling 8000)
```

**The approved guard.** If `N < 2000` the script exits with code 3 and writes
`status: STOPPED-UNDERTRAINED`. Do not raise the ceiling or lower the floor to get past
it: an undertrained S2 is not a weak result, it makes the whole stream comparison
invalid. Report the measured step time and stop.

fp16 autocast with a GradScaler, never bf16 — the T4 is sm_75.

---

## Step 5 — S2 inference

Score the panel with each run, and the full views with each run (the budget raise paid
for all four):

```python
for protocol, seed in [("p_a_sd1", 0), ("p_a_sd1", 1), ("4ch_only", 0), ("full_bank", 0)]:
    ...  # p2_train_s2 checkpoints -> score panel + full views -> reports/p2_s2_scores_*.csv
```

---

## Step 6 — persist

**Count before you zip.** This is the session whose artefacts cost GPU-hours; losing
them silently means paying for them twice.

```python
import json, pathlib, shutil
R, OUT = DBP / "reports", DBP / "outputs" / "p2"

members = [
    "sd15", "sd21", "sd_ft_ema", "sdxl", "sdxl_fp16fix", "sd35",
    "flux", "dcae", "realesrgan_x4", "flux2_klein", "qwen_image", "movq_k22",
]
done = []
for key in members:
    npz = OUT / "probe" / f"p2_probe_{key}.npz"
    side = npz.with_suffix(".done.json")
    state = "OK" if (npz.is_file() and side.is_file()) else "MISSING"
    print(f"  probe {key:16s} {state}")
    if state == "OK":
        done.append(key)
assert len(done) == 12, f"only {len(done)}/12 probe caches complete: missing " \
                        f"{sorted(set(members) - set(done))}"

for tag in ("train", "panel"):
    f = OUT / "s1" / f"p2_s1_dinov3-vitb16_{tag}.npz"
    print(f"  s1 {tag:6s} {'OK' if f.is_file() else 'MISSING'}")
    assert f.is_file(), f"S1 {tag} cache missing"

for protocol, seed in [("p_a_sd1", 0), ("p_a_sd1", 1), ("4ch_only", 0), ("full_bank", 0)]:
    f = R / f"p2_s2_{protocol}_seed{seed}.json"
    if not f.is_file():
        print(f"  s2 {protocol}/{seed}: NOT RUN")
        continue
    d = json.loads(f.read_text(encoding="utf-8"))
    print(f"  s2 {protocol}/{seed}: {d['status']} steps={d.get('steps_run')} "
          f"val_auc={d.get('best_val_auc')} gpu_h={d.get('gpu_hours')}")
    assert d["status"] != "STOPPED-UNDERTRAINED", \
        "an undertrained run invalidates the stream comparison -- report, do not zip past it"

shutil.make_archive("/kaggle/working/p2_streams_reports", "zip", str(R))
shutil.make_archive("/kaggle/working/p2_features", "zip", str(OUT))
!du -sh /kaggle/working
```

**Save Version with output saving on**, and wait for it to finish. The feature caches
are the expensive artefact — if they are lost, Steps 2 and 3 are paid for twice.
Download both zips to `F:\thesis-2\kaggle_raw\`, never into `REPO_ROOT` (I-032).

**Then read the quota meter and write down what it actually charged** against the
15.22 session-hours estimated above. That is I-042, and the whole 32 GPU-h budget
rests on it.

---

## Checklist

- [ ] Step 0 passed: panel **12,150** rows, digest **recomputed and matching**, 0 missing files, every list row carrying a `root`.
- [ ] Step 0b: lists rebuilt — **33,909 / 12,150 / 19,011 / 1,000** — and **0** T-ctrl rows in the S1 list.
- [ ] S1 extracted under three tags: `train`, `panel`, `panel_luma`.
- [ ] LPIPS calibration run; projected S3 cost under 11 GPU-h, or reported and stopped.
- [ ] Probe features for **all 12** members present in `outputs/p2/probe/`.
- [ ] S1 log shows 4 register tokens, 400 patch tokens, dim 1536, `HF_TOKEN present: True`.
- [ ] Four S2 runs, each with a **measured** step time and `steps >= 2000`.
- [ ] Every run's `best_val_auc` and `gpu_hours` recorded.
- [ ] **Quota meter checked** and the actual charge reported against the estimate.
- [ ] Both zips downloaded outside `REPO_ROOT`.
