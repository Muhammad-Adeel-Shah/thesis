# Kaggle run — P1 evaluation sets (B7)

**Purpose.** Index the approved evaluation sets, describe their compression history,
build the JPEG-matched views where compression is confounded with the label, and check
for leakage against the training corpora.

**Test-only.** Nothing here trains anything. No evaluation image is ever modified; the
matched views are *new* files alongside untouched originals.

**Session.** GPU is not needed — this is I/O and Pillow. Internet **on**. Run it after
`kaggle_run_p1_build.md`, because the leakage check compares against the training
manifests that run produces.

**Sets in scope.**

| set | source | licence | status |
|---|---|---|---|
| **E7 DDA-COCO** | `Junwei-Xi/DDA-COCO` @ `8c9330a3b374bcac46a8045a0e3c09ebcf7868fb` | apache-2.0 | **PRIMARY head-to-head set** |
| **E8 EvalGEN** | `Junwei-Xi/EvalGEN` @ `86c719a9bfc72b7e5d91bfbb04432774feb4e3b7` | apache-2.0 | breadth: 5 generators, 3 non-diffusion |
| **E3 MLLM** | `zr-zhang/MLLM-Generated-Image-Detection-Dataset` @ `1498eead24292a9a2d134476b2c559193b68b9de` | apache-2.0 | three views, matched view primary |
| **E4 WildRF** | `barcavia/RealTime-DeepfakeDetection-in-the-RealWorld` (Drive) | `NOASSERTION` | **test split only** |
| **E5 Chameleon** | AIDE (ICLR 2025); images from a community mirror, SHA-256 `a7cdadfe…` | academic research only | **320 px crop, not 384** — see Step 6 |

Not run here, and why:

| set | status |
|---|---|
| E6 MNW | repository resolved (`github.com/microsoft/MNW`), terms and citation recorded — blocked on the **download route**, which the README does not contain. Any token goes in Kaggle Secrets only. See `docs/data_cards/e6_mnw.md` |
| E2 Synthbuster | deferred to P2 with its own audit (RAISE RAW/TIFF size, conversion without resizing, the resolution confound against 1024-px fakes, CC-BY-NC-SA terms) |
| GenImage | **optional, and only if the budget allows after the main build.** `github.com/GenImage-Dataset/GenImage`, Google Drive, million-scale. **Ask before downloading** — do not start it on your own initiative |

E1, the private zero-shot set, has been **withdrawn** (P1 review): it is not public and
has no paper, so it cannot support a claim. E7 and E8 replace it.

---

## Step 1 — Cells 1–3 from the preflight

Stage the repo, set the caches, run the bootstrap. `HF_TOKEN` is needed for E3.

Also copy the training manifests in, so the leakage check has something to compare
against. Attach the P1 build output dataset and:

```python
import pathlib, shutil
src = pathlib.Path("/kaggle/input/<your-p1-build-dataset>/reports")
dst = pathlib.Path(DBP) / "reports"
for name in ("p1_reals_manifest_t_real.csv", "p1_reals_manifest_t_ctrl.csv",
             "p1_gen_manifest.csv"):
    if (src / name).exists():
        shutil.copy2(src / name, dst / name)
        print("copied", name)
    else:
        print("MISSING:", name, "-> the leakage check will record NOT-RUN for it")
```

A missing manifest is recorded as `NOT-RUN` with its reason rather than silently
skipped: a leakage check that quietly did nothing is worse than no leakage check.

---

## Step 2 — E7 DDA-COCO (primary)

```python
from huggingface_hub import snapshot_download
E7 = snapshot_download(
    repo_id="Junwei-Xi/DDA-COCO",
    repo_type="dataset",
    revision="8c9330a3b374bcac46a8045a0e3c09ebcf7868fb",
    local_dir="/tmp/e7",
)
print(E7)
```

The repo ships a single `DDA-COCO.zip` (repo total 4,301,456,312 bytes). Extract to
`/tmp` and **print the tree before writing any globs** — the internal layout is not
documented and must not be guessed:

```python
import collections, pathlib, zipfile
z = pathlib.Path("/tmp/e7/DDA-COCO.zip")
with zipfile.ZipFile(z) as f:
    names = f.namelist()
    f.extractall("/tmp/e7x")
print(len(names), "entries")
for depth in (1, 2, 3):
    c = collections.Counter("/".join(n.split("/")[:depth]) for n in names if n.count("/") >= depth - 1)
    print(f"--- depth {depth} ---")
    for k, v in sorted(c.items())[:40]:
        print(f"   {k:70s} {v}")
```

Write the `--real-glob` / `--fake-glob` and a `--field` for the source VAE from what that
prints, then:

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e7_dda_coco --root /tmp/e7x \
    --real-glob "<real pattern>" --fake-glob "<fake pattern>" \
    --field source_vae=<depth> --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

**The leakage check is the point of this set, not a formality (I-024).** DDA-COCO's reals
are MSCOCO **validation**; T-real is built from `train2017` only for exactly this reason.
If `reports/p1_leakage.json` reports `LEAKAGE` under `e7_dda_coco`, **fix the training
selection and rebuild T-real** — do not drop the offending evaluation images, which would
silently change the benchmark.

Also note what E7's fakes are: **VAE reconstructions of COCO**, which is the same
operation B5 performs. Report per source VAE, and separate the subsets whose VAE is in
our bank from the ones that are not — see `docs/data_cards/e7_dda_coco.md` §3.

---

## Step 3 — E8 EvalGEN

```python
from huggingface_hub import snapshot_download
E8 = snapshot_download(
    repo_id="Junwei-Xi/EvalGEN",
    repo_type="dataset",
    revision="86c719a9bfc72b7e5d91bfbb04432774feb4e3b7",
    local_dir="/tmp/e8",
)
print(E8)
```

13.54 GiB compressed. Print the tree the same way before writing globs.

Two things about E8 that change how it is run:

1. **It appears to contain no real images** — its README describes ~55,300 images across
   five generators and names no real source. If that holds, pass `--fake-glob` only. AUC
   and AP are then undefined and `dbp.eval.metrics` will raise rather than return 0.5;
   E8 is reported as a detection rate at a threshold calibrated elsewhere, or paired with
   E7's reals. Confirm from the extracted tree and record the answer in the data card.
2. **Every image is JPEG quality 96**, per its README. So there is no confound *within*
   E8 (one label), but pairing it with reals of a different quality distribution creates
   one *across* the pairing. Record E8's measured quality distribution and compare it to
   whichever reals it is paired with.

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e8_evalgen --root /tmp/e8x \
    --fake-glob "<fake pattern>" --field generator=<depth> \
    --subsample 1000 --stratify-by generator \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

`--subsample 1000 --stratify-by generator` keeps **1,000 per generator, 5,000 total**
(D-036), chosen by a seeded permutation of each generator's sorted image ids rather than
by file order. The sampler, the seed rule, the per-stratum counts and every kept id land
in `reports/p1_eval_e8_evalgen_summary.json`. Delete the extracted tree afterwards; only
the subset is kept.

The same two flags are available for E7 if the preflight throughput demands it —
stratify by the source-VAE field there instead.

`--field generator=<depth>` is required, not optional: the five generators are the whole
reason for the set and every number is reported per generator, never pooled. Three of
them (Infinity, NOVA, and OmniGen's auto-regressive path) never run a VAE decoder, which
is the sharpest available test of where the decoder-bank premise stops holding. A poor
score there is a result to report, not a failure to hide.

---

## Step 4 — E3

Not gated, apache-2.0. Downloaded inside the Kaggle session (CLAUDE.md §6a condition
(a)), to `/tmp`, which is not persisted.

```python
from huggingface_hub import snapshot_download
E3 = snapshot_download(
    repo_id="zr-zhang/MLLM-Generated-Image-Detection-Dataset",
    repo_type="dataset",
    revision="1498eead24292a9a2d134476b2c559193b68b9de",
    local_dir="/tmp/e3",
)
print(E3)
```

The revision is the repository SHA read from the Hub on 2026-09-20. Pinning it means a
later re-upload cannot silently change what "E3" means, exactly as for the model pins.

### What E3 actually is

The P1 prompt described E3 as "2,904 images, 320×320 BMP, already resized". That is
**wrong in every particular**. Measured from the Hub file listing on 2026-09-20:

- **4,356 image files** = **2,178 unique images × 2 views** (`Raw`, `Preprocessing`).
- **No BMP at all.** `Raw` is `.png` / `.jpg` / `.jpeg`; `Preprocessing` is all `.jpg`.
- Perfectly balanced: **726 real / 726 GPT-Image2-fake / 726 Nano-Banana2-fake**,
  across Texture 648 / Structure 939 / Hybrid 591 (197 / 313 / 216 per label per regime).
- `Raw` is native resolution and varies per image. `Preprocessing` is downscaled JPEG —
  variable size, not square.

And the reason the matched view exists, from the same listing:

| `Raw` | PNG | JPEG |
|---|---|---|
| real | **0** | 726 |
| fake | **1361** | 91 |

Every real is a JPEG and 93.7 % of fakes have never been compressed. A detector can
score well there by reading the compression history alone [R7]. Loading pixels rather
than files stops the *format* leaking; it does not stop the *compression history*.

### Three views

```python
E3ARGS = ["--field", "subset=2", "--field", "source_label=3",
          "--leak-check-against",
          "reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv"]
```

```python
# Raw, as distributed — reported, but never the basis of a claim.
!cd {DBP} && python scripts/p1_build_eval.py --set-id e3_raw --root /tmp/e3 \
    --real-glob "images/Raw/*/real/*" --fake-glob "images/Raw/*/*-fake/*" \
    --field subset=2 --field source_label=3 --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

`--jpeg-match` produces **`e3_raw_jpeg_matched`**, the **PRIMARY view for every claim**:
only the 1,361 never-compressed fakes are re-encoded, at native resolution, with quality
and chroma subsampling drawn per image from E3's own 726 real JPEGs, seeded by
`blake2b("e3_raw:" + image_id)`. The reals are untouched.

```python
# Preprocessing — secondary. Resized AND recompressed; always labelled as such.
!cd {DBP} && python scripts/p1_build_eval.py --set-id e3_preprocessing --root /tmp/e3 \
    --real-glob "images/Preprocessing/*/real/*" \
    --fake-glob "images/Preprocessing/*/*-fake/*" \
    --field subset=2 --field source_label=3 \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

No `--jpeg-match` here: `Preprocessing` is already all-JPEG, so the script will report
"not confounded" and build nothing. The problem with this view is the resize, which no
re-encoding can undo — hence secondary.

Confirm in `reports/p1_eval_e3_raw_summary.json`:
- `compression_confounded: true` for the raw view, with the spread near 93.7 %;
- `jpeg_matched_view.compression_confounded: false` for the matched one.

If the second is still `true`, the matched view did not do its job — stop and report it.

---

## Step 5 — E4 WildRF (test split only)

Source: `barcavia/RealTime-DeepfakeDetection-in-the-RealWorld` (GitHub, licence
`NOASSERTION` → **research use only**, recorded in CLAUDE.md §6c). The dataset is a
Google Drive file linked from its README.

`gdown` is installed **inside the session only** and pinned (CLAUDE.md §6a condition
(c)). It is deliberately *not* added to `requirements/`.

```python
!python -m pip install --quiet "gdown==5.2.0"
!python -c "import gdown; print('gdown', gdown.__version__)"
```

```python
import pathlib, subprocess, zipfile
FILE_ID = "1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn"   # from the repo README, read 2026-09-20
dest = pathlib.Path("/tmp/wildrf"); dest.mkdir(parents=True, exist_ok=True)
archive = dest / "wildrf.zip"
if not archive.exists():
    subprocess.run(["gdown", "--id", FILE_ID, "-O", str(archive)], check=True)
print("bytes:", archive.stat().st_size)
with zipfile.ZipFile(archive) as z:
    z.extractall(dest)
print(sorted(p.name for p in dest.iterdir()))
```

The published layout is `train/`, `val/` and `test/{reddit,twitter,facebook}/{0_real,1_fake}`.
**Only `test/` is indexed.** WildRF's own protocol trains on one platform and tests on
unseen ones; using any of it for training would both break that protocol and make the
cross-generator claim circular.

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e4_wildrf_test \
    --root /tmp/wildrf/WildRF \
    --real-glob "test/*/0_real/*" --fake-glob "test/*/1_fake/*" \
    --field platform=1 --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

If the root is not `/tmp/wildrf/WildRF`, find it:

```python
import pathlib
print([str(p) for p in pathlib.Path("/tmp/wildrf").rglob("test") if p.is_dir()])
```

E4 is social-media imagery, so most of it is already JPEG and the script will very
likely report "not confounded" and build no matched view. That verdict still goes in the
data card — CLAUDE.md §6b requires the table for **every** set, including the ones that
turn out to be clean. `--field platform=1` gives the per-platform breakdown the WildRF
protocol is defined over.

---

## Step 6 — E5 Chameleon

The archive is **already downloaded locally** by the user
(`F:\thesis-2\Chameleon.zip`, 2,683,359,341 bytes, SHA-256
`a7cdadfe35287dd3e81b5c35e7cce5437e68ba3b4cdbb833ac0686b990c2fba6`). Upload it as its
own private Kaggle dataset — do **not** put it in the code dataset, and never inside
`REPO_ROOT` (it briefly was, and made the integrity check CRITICAL; D-017).

Verify the hash after upload before anything else:

```python
import hashlib, pathlib
src = next(pathlib.Path("/kaggle/input").rglob("Chameleon.zip"))
h = hashlib.sha256()
with src.open("rb") as f:
    for chunk in iter(lambda: f.read(8 << 20), b""):
        h.update(chunk)
print(src, src.stat().st_size)
print("sha256:", h.hexdigest())
assert h.hexdigest() == "a7cdadfe35287dd3e81b5c35e7cce5437e68ba3b4cdbb833ac0686b990c2fba6"
```

```python
import zipfile
with zipfile.ZipFile(src) as z:
    z.extractall("/tmp/e5")
```

Expected layout, verified from the archive's central directory: `Chameleon/test/0_real/`
(14,863 `.jpg`) and `Chameleon/test/1_fake/` (11,170 `.jpg`). Those counts match the
paper's Table 1 exactly.

```python
!cd {DBP} && python scripts/p1_build_eval.py --set-id e5_chameleon \
    --root /tmp/e5/Chameleon \
    --real-glob "test/0_real/*" --fake-glob "test/1_fake/*" --jpeg-match \
    --leak-check-against reports/p1_reals_manifest_t_real.csv,reports/p1_reals_manifest_t_ctrl.csv
```

### Three things about Chameleon that are not optional

**1. The crop is 320, not 384.** Every one of the 14,863 reals is exactly 500 px wide
and short (median height 359); the fakes are native resolution up to 5376×6720. At a 384
crop only **44.0%** of reals survive against **94.5%** of fakes, which flips the class
balance from 57/43 to 38/62 and keeps only the unusually tall reals. At 320 it is 93.1%
/ 95.9%. Run configs `full_bank_chameleon` and `4ch_only_chameleon` carry `eval_crop:
320, eval_crops_per_image: 1`. Multi-crop is rejected here — two 320 crops would retain
5,211 reals against 9,200 fakes.

**2. Width alone separates the classes perfectly.** A one-line rule `width == 500 → real`
scores **100.00%** on this set (14,863/14,863 reals; 1/11,170 fakes). The fixed square
crop removes that cue from the pixels, which is the only reason any Chameleon number is
meaningful. Say so wherever the number appears.

**3. No category labels exist in the archive.** The paper reports scene / object /
animal / human, but the archive is two flat directories of UUID filenames. Per-category
reporting is **blocked** until the official URL list or a mapping file arrives. Report in
aggregate and state the limitation — the animal split (3,998 real : 313 fake) is exactly
what a pooled number hides.

Expect `compression_confounded: true` (fakes 6.8% never-compressed, reals 0.0%) and the
matched view to re-encode 757 fakes at exactly q75 / 4:2:0 — every one of the 14,863
reals is q75 / 4:2:0, so the reference distribution is a single point.

Report balanced accuracy and AUC with bootstrap confidence intervals.

---

## Step 7 — collect

```python
import pathlib, shutil
out = pathlib.Path("/kaggle/working/p1_eval_reports")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
for f in sorted(pathlib.Path(f"{DBP}/reports").glob("p1_eval_*")):
    shutil.copy2(f, out / f.name)
leak = pathlib.Path(f"{DBP}/reports/p1_leakage.json")
if leak.exists():
    shutil.copy2(leak, out / leak.name)
for f in sorted(out.iterdir()):
    print(f.name, f.stat().st_size)
shutil.make_archive("/kaggle/working/p1_eval_results", "zip", out)
```

Publish the matched-view images as a dataset through **Save Version → output → dataset**.
Download only the reports zip to `F:\thesis-2\kaggle_raw\`.

---

## Checklist

- [ ] E7 indexed per source VAE; **leakage against T-real `CLEAN`** (I-024). If not,
      rebuild T-real from `train2017` — never drop evaluation images.
- [ ] E8 indexed per generator; whether it contains reals recorded in its data card;
      measured quality distribution recorded (README says JPEG q96).
- [ ] E3 pinned to its dataset revision, all three views indexed.
- [ ] `e3_raw` confounded; `e3_raw_jpeg_matched` **not** confounded.
- [ ] Matched view: 1,361 re-encoded, 0 reals re-encoded, no image resized.
- [ ] E4 indexed from `test/` only, with the per-platform breakdown.
- [ ] `p1_leakage.json` has an entry for every set, and every one says `CLEAN`.
      Any `LEAKAGE` entry must be resolved before P2 reports a number on that set.
- [ ] Compression tables copied into the data cards; per-set counts reported.
- [ ] E8 subsampled 1,000 per generator, seeded and stratified; kept ids recorded.
- [ ] E5 Chameleon: archive SHA-256 verified after upload; counts 14,863 / 11,170;
      crop 320 single-crop; per-category limitation stated.
- [ ] E6 MNW still blocked on its download route.
