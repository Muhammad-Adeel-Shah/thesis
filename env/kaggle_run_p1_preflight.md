# Kaggle run — P1 preflight (B1 + B2)

**Purpose.** Replace the guesses the P1 budget rests on with measurements, and prove the
P1 code changes did not move a single number the P0 run produced. Nothing is built here;
if something is wrong, this is the cheap place to find out.

**Session.** Accelerator **GPU T4 x2**, internet **on**. Expect roughly 45–90 minutes,
most of it the ~5 GB bank download and the SD1.5 checkpoint.

**Secrets.** Add-ons → Secrets → `HF_TOKEN`. It is read through `kaggle_secrets` or the
environment and is never printed, logged or written to disk.

---

## Step 1 — build the upload zip (locally, before the session)

Exactly as in `kaggle_run_p0.md`: tracked files only, with the commit embedded so every
number can be tied to the code that produced it.

```powershell
cd F:\thesis-2\Implementation
$sha = git rev-parse HEAD
git archive --format=zip --prefix=thesis_dbp/ `
    --add-virtual-file="thesis_dbp/COMMIT.txt:$sha" `
    -o F:\thesis-2\thesis_dbp_p1.zip HEAD:thesis_dbp
python -c "import zipfile;z=zipfile.ZipFile(r'F:\thesis-2\thesis_dbp_p1.zip');print(len(z.namelist()),'files')"
```

Upload as a **private** Kaggle Dataset (suggested slug `dbp-p1`). Remember that Kaggle
extracts the zip when it creates the dataset, so Cell 1 below handles both shapes.

---

## Step 2 — notebook cells

### Cell 1 — stage into a writable location

```python
import os, pathlib, shutil, zipfile

SRC = pathlib.Path("/kaggle/input/dbp-p1")
WORK = pathlib.Path("/kaggle/working/Implementation")
WORK.mkdir(parents=True, exist_ok=True)

zips = sorted(SRC.rglob("*.zip"))
trees = sorted(p for p in SRC.rglob("thesis_dbp") if p.is_dir())
if zips:
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORK)
elif trees:
    shutil.copytree(trees[0], WORK / "thesis_dbp", dirs_exist_ok=True)
else:
    raise FileNotFoundError(f"no zip or thesis_dbp/ under {SRC}: "
                            f"{sorted(p.name for p in SRC.iterdir())}")

DBP = (WORK / "thesis_dbp").resolve()
assert not str(DBP).startswith("/kaggle/input"), f"read-only mount: {DBP}"
probe = DBP / "outputs" / "logs"
probe.mkdir(parents=True, exist_ok=True)
(probe / ".writable").write_text("ok"); (probe / ".writable").unlink()
print(DBP, "commit:", (DBP / "COMMIT.txt").read_text().strip())
```

### Cell 2 — caches off the /kaggle/working quota

```python
import os
os.environ["HF_HOME"] = "/tmp/hf_home"
os.environ["TORCH_HOME"] = "/tmp/torch_home"
os.environ["PIP_CACHE_DIR"] = "/tmp/pip_cache"
for key in ("HF_HOME", "TORCH_HOME", "PIP_CACHE_DIR"):
    os.makedirs(os.environ[key], exist_ok=True)
    print(key, "=", os.environ[key])
```

### Cell 3 — bootstrap

```python
!bash {DBP}/env/kaggle_bootstrap.sh
```

Read step [7/7]. If it reports that `peft` cannot be imported because of `torchao`
(issue I-011), run the uninstall it prints, **restart the kernel**, and re-run Cells 1–3.

### Cell 4 — token into the process, then pins

The token is put in `os.environ` for the child processes and never echoed.

```python
import os
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    print("Kaggle Secrets unavailable:", type(exc).__name__, exc)
print("HF_TOKEN present:", bool(os.environ.get("HF_TOKEN")))

!cd {DBP} && python scripts/p0_verify_pins.py
```

**Expect 15 pins, 0 failed.** If any line says `REVISION-NOT-FOUND`, stop: the registry
is wrong and nothing measured afterwards would mean anything.

### Cell 5 — tests and lint, on this runtime

```python
!cd {DBP} && python -m pytest -q tests -m "not network"
!cd {DBP} && python -m ruff check .
```

Both must pass. `test_gpu_smoke.py` runs for real here rather than skipping.

### Cell 6 — regression against the P0 T4 run (real weights)

This is the check that the P1 decoder-bank changes left the ten original members
bit-identical. It re-measures them and compares the deterministic columns against the
committed P0 CSV.

```python
!cd {DBP} && python scripts/p0_bank_smoke_test.py --device cuda --crops 512,1024 \
    --members sd15,sd21,sd_ft_ema,sd_ft_mse,sdxl,sdxl_fp16fix,sd35,flux,dcae,realesrgan_x4 \
    --compare-to reports/kaggle_20260920/p0_bank_smoke.csv
```

**Expect `regression check PASSED` over 20 (key, crop) pairs.** A non-zero exit means the
P1 refactor changed a reconstruction: stop and report which member and column moved.

Note the timings in this run will differ from the P0 CSV — three warm-ups are now
discarded (I-009) — and that is exactly why timing columns are excluded from the
comparison.

### Cell 7 — the three new members, for the first time on real weights

```python
!cd {DBP} && python scripts/p0_bank_smoke_test.py --device cuda --crops 512,1024 \
    --members flux2_klein,qwen_image,movq_k22
```

Watch for: all rows `OK`; `deterministic=True`; `max_abs_delta_rerun=0`; and the
`p0_decoder_identity.json` groups — if any new member hashes identically to an existing
one it is an alias, not a lineage, and `decoder_bank.yaml` must say so before B5 runs.

### Cell 7b — what dtype is FLUX.2-klein actually stored in? (I-020)

FLUX.2-dev's `vae/diffusion_pytorch_model.safetensors` is **336,213,556 bytes** and
klein's is **168,120,878 bytes** — almost exactly 2×. That ratio is *consistent with*
fp32 versus half-precision storage of the same architecture, but it is an observation,
not evidence of identity: two different models of the same shape would show the same
ratio. FLUX.2-dev returns 403 for this token, so the claim stays **UNVERIFIED** and no
identity claim is made in either direction.

What can be settled here is one half of it — klein's stored dtype:

```python
from safetensors import safe_open
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "black-forest-labs/FLUX.2-klein-4B",
    filename="vae/diffusion_pytorch_model.safetensors",
    revision="e7b7dc27f91deacad38e78976d1f2b499d76a294",
    token=os.environ.get("HF_TOKEN"),
)
with safe_open(path, framework="pt") as f:
    dtypes = {}
    total = 0
    for key in f.keys():
        t = f.get_slice(key)
        dtypes[t.get_dtype()] = dtypes.get(t.get_dtype(), 0) + 1
        n = 1
        for d in t.get_shape():
            n *= d
        total += n
print("tensor dtypes:", dtypes)
print("parameters   :", total)
print("bytes/param  :", 168120878 / total)
```

Record `dtypes`, the parameter count and bytes-per-parameter in the report. A result of
`{'BF16': n}` or `{'F16': n}` at ~2 bytes/parameter makes the size ratio explicable
without any assumption about dev's weights; `{'F32': n}` at ~4 would mean the ratio has a
different cause entirely, and that is worth knowing.

Either way, **I-020 stays open and the identity stays UNVERIFIED.** This measures klein,
not the pair.

---

### Cell 8 — throughput, fp16 eligibility and the B6 cost (B1)

```python
!cd {DBP} && python scripts/p1_throughput_bench.py --device cuda \
    --crops 384 --batches 1,2,4 --time-generation 10
```

Produces `reports/p1_throughput.csv` and `reports/p1_backbone_throughput.json`.

What to read out of it:
- `fp16_build_allowed` per member — B5 reads this file, so it decides the build.
  `sdxl` and `sdxl_fp16fix` are expected to be `False` under the 55 dB rule.
- `ms_per_img` at crop 384 per member → the real B5 budget, replacing the P0
  extrapolation.
- `generation.median_ms_per_image` → the real B6 budget, which was **UNMEASURED** before
  this run.

### Cell 9 — JPEG encoder equivalence (B2, issue I-010)

```python
!cd {DBP} && python scripts/p1_jpeg_equivalence.py \
    --compare-to reports/p1_jpeg_equivalence.json \
    --out reports/p1_jpeg_equivalence_kaggle.json
```

The `--compare-to` file is the **local** run, committed to the repository
(Pillow 12.3.0 / libjpeg 8.0 / Windows). The verdict is one of:

| verdict | meaning | consequence |
|---|---|---|
| `IDENTICAL` | byte-identical at all 24 cells | matched views could be built anywhere |
| `BYTES-DIFFER-PIXELS-MATCH` | container/entropy coding only | build in one place; a rebuild elsewhere gives the same pixels |
| `PIXELS-DIFFER` | different libjpeg numerics | **matched views must be built on Kaggle only**, and the data card must name the build machine |

A non-zero exit is expected for the middle and last cases; it is a finding, not a failure.

### Cell 10 — collect the reports

Reports only. No images, no weights.

```python
import pathlib, shutil
out = pathlib.Path("/kaggle/working/p1_preflight_reports")
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)
for f in sorted(pathlib.Path(f"{DBP}/reports").glob("*")):
    if f.is_file():
        shutil.copy2(f, out / f.name)
        print(f.name, f.stat().st_size)
shutil.make_archive("/kaggle/working/p1_preflight_results", "zip", out)
print("done")
```

Download `p1_preflight_results.zip` to `F:\thesis-2\kaggle_raw\` (CLAUDE.md §7b — **never**
inside `REPO_ROOT`). The extracted reports are then committed under
`thesis_dbp/reports/kaggle_<date>/`.

---

## Gate before the long build run

Do not start `kaggle_run_p1_build.md` until all of these hold:

- [ ] 15 pins resolve, 0 fatal.
- [ ] pytest and ruff pass on the Kaggle runtime.
- [ ] Cell 6 prints `regression check PASSED`.
- [ ] Cell 7: all three new members `OK` and deterministic.
- [ ] Cell 7b: klein's stored dtype and bytes-per-parameter recorded (I-020 stays open).
- [ ] `p1_throughput.csv` has an `fp16_build_allowed` verdict for every member.
- [ ] `generation.median_ms_per_image` is a real number, and
      `2500 x that / 3600000` hours fits the remaining budget.
- [ ] The JPEG verdict is recorded, whatever it is.
