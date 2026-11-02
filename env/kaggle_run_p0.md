# Running the P0 checks on Kaggle (T4)

The local machine has no NVIDIA GPU, so every fp16, VRAM and T4-timing number is
`PENDING-GPU` until this runs. **All numbers reported in the thesis come from here.**

Session settings: **Accelerator = GPU T4 x2**, **Internet = On**.
Do not pick P100: PyTorch CUDA-13 builds dropped pre-Turing (sm_60) support [R19], and
`kaggle_bootstrap.sh` will abort if it sees one.

---

## Step 1 — Package the repo (run locally)

The default path uploads a zip of the **tracked files only**, built with `git archive`,
so no weights, caches, `.venv` or data can be included by accident. No `git push` is
involved in P0.

```powershell
cd F:\thesis-2\Implementation
$sha = git rev-parse p0-env-setup
git archive --format=zip --prefix=thesis_dbp/ `
    --add-virtual-file=thesis_dbp/COMMIT.txt:$sha `
    -o F:\thesis-2\thesis_dbp_p0.zip p0-env-setup:thesis_dbp
```

`--add-virtual-file` (git ≥ 2.38; verified on 2.54.0.windows.1) embeds the full commit
hash as `COMMIT.txt`. Kaggle has no `.git` directory, so without this the environment
report there would say `git: unavailable` and its numbers could not be tied back to the
code that produced them. `env_report.py` reads `COMMIT.txt` when `.git` is absent and
records it as `git_commit`, with `git_commit_source: "COMMIT.txt"`.

Check what you are about to upload:

```powershell
python -c "import zipfile;z=zipfile.ZipFile(r'F:\thesis-2\thesis_dbp_p0.zip');print(len(z.namelist()),'files',round(sum(i.file_size for i in z.infolist())/1e6,2),'MB')"
```

It should be well under 1 MB. If it is not, something untracked was included — stop and
check `git status` before uploading.

Then: **kaggle.com → Datasets → New Dataset → Upload** `thesis_dbp_p0.zip`,
set visibility to **Private**, and title it `dbp-p0`. It mounts at
`/kaggle/input/dbp-p0/`.

> **Alternative (optional):** clone from GitHub instead. This requires pushing the
> branch first, which P0 deliberately does not do.

---

## Step 2 — Add the HF token as a Kaggle Secret

**Add-ons → Secrets → Add a new secret**, named exactly `HF_TOKEN`, then attach it to
the notebook. The bootstrap reads it via `kaggle_secrets.UserSecretsClient` and never
prints it.

Without it, `sd35` and `flux` are reported `BLOCKED-ACCESS` and the two 16-channel
lineages — the core claim against SEF — are not tested.

---

## Step 3 — Notebook cells (copy-paste in order)

### Cell 1 — stage the repo into a writable location

**Kaggle extracts an uploaded zip when it creates the dataset**, so by the time the
notebook runs there is usually no `.zip` left under `/kaggle/input` -- only the
extracted tree. In P0 the `*.zip` glob below therefore found nothing, the fallback
`copytree` was never reached either, and `DBP` silently pointed at the read-only
`/kaggle/input`. Two scripts then died with
`OSError: [Errno 30] Read-only file system` while creating their log directory.

This version locates `thesis_dbp/` wherever it actually is, copies it into
`/kaggle/working`, and **asserts the result is writable** before going further.

```python
import os, pathlib, shutil, zipfile

SRC = pathlib.Path("/kaggle/input/dbp-p0")
WORK = pathlib.Path("/kaggle/working/Implementation")
WORK.mkdir(parents=True, exist_ok=True)

zips = sorted(SRC.rglob("*.zip"))
trees = sorted(p for p in SRC.rglob("thesis_dbp") if p.is_dir())

if zips:
    print("staging from zip:", zips[0].name)
    with zipfile.ZipFile(zips[0]) as z:
        z.extractall(WORK)
elif trees:
    # Kaggle already extracted the upload; copy the tree instead.
    print("staging from extracted tree:", trees[0])
    shutil.copytree(trees[0], WORK / "thesis_dbp", dirs_exist_ok=True)
else:
    raise FileNotFoundError(
        f"neither a .zip nor a thesis_dbp/ directory found under {SRC}. "
        f"Contents: {sorted(p.name for p in SRC.iterdir())}"
    )

DBP = (WORK / "thesis_dbp").resolve()
assert DBP.is_dir(), f"{DBP} was not created"
assert not str(DBP).startswith("/kaggle/input"), (
    f"DBP is under the read-only input mount: {DBP}"
)

# Prove it is writable NOW, rather than finding out inside setup_logging().
probe = DBP / "outputs" / "logs"
probe.mkdir(parents=True, exist_ok=True)
(probe / ".writable").write_text("ok", encoding="utf-8")
(probe / ".writable").unlink()

print(DBP, "->", len(list(DBP.rglob("*"))), "entries; writable: yes")
print("commit:", (DBP / "COMMIT.txt").read_text().strip() if (DBP / "COMMIT.txt").exists()
      else "COMMIT.txt MISSING")
```

### Cell 2 — caches off the /kaggle/working quota

`/kaggle/working` has a ~20 GB quota and is persisted. The bank weights are ~4.9 GB and
are not worth persisting, so they go to `/tmp`, which is not.

```python
import os
os.environ["HF_HOME"]    = "/tmp/hf_home"
os.environ["TORCH_HOME"] = "/tmp/torch_home"      # 553 MB VGG16 for LPIPS
os.makedirs("/tmp/hf_home", exist_ok=True)
os.makedirs("/tmp/torch_home", exist_ok=True)

from kaggle_secrets import UserSecretsClient
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
print("HF_TOKEN present:", bool(os.environ.get("HF_TOKEN")))   # never print the value
```

### Cell 3 — bootstrap

```python
!bash {DBP}/env/kaggle_bootstrap.sh
```

Expect: T4 sm_75 detected, `OK: torch / torchvision / numpy are untouched`, then
`OK: unchanged.` If the guardrail aborts, **stop and send me the output** — do not
force the install.

### Cell 4 — environment report

```python
!python {DBP}/scripts/p0_env_check.py
```

### Cell 5 — verify every pinned revision still resolves

Run this **before** the 4.9 GB download. A pin that points at nothing must be caught
here, not three hours into a session. During P0 five pinned SHAs were format-valid but
fabricated, so this check is not hypothetical.

It covers all 12 pins: the 9 Hub bank members, `realesrgan_x4` (pinned by URL +
SHA-256 rather than a commit), and both backbones.

`scripts/p0_verify_pins.py` is the single source of truth for this check. From P1
onward every data-building script calls it before writing any data.

```python
!python {DBP}/scripts/p0_verify_pins.py
```

Expected: **12 pins checked, 0 failed, 0 blocked.** Anything else — stop and send me the
output. A `BLOCKED-ACCESS` line means the `HF_TOKEN` secret is missing or a licence was
not accepted; a `REVISION NOT FOUND` line means the registry is wrong and must be fixed
before any data is built.

### Cell 6 — decoder-bank smoke test (the long one, ~20–40 min)

```python
!python {DBP}/scripts/p0_bank_smoke_test.py --device cuda --crops 512,1024
```

This downloads ~4.9 GB on the first run. It measures, per member: PSNR, LPIPS-vgg,
mean|r|, ms/img, peak VRAM, fp16 parity and determinism, and saves an
input/recon/amplified-residual PNG triplet per member.

### Cell 7 — backbone + LoRA dry-run

```python
!python {DBP}/scripts/p0_backbone_smoke_test.py --device cuda
```

Both backbones must succeed: DINOv3 gating was approved on 2026-09-19 and confirmed by
a real gated download, so a `BLOCKED-ACCESS` line here means the Kaggle `HF_TOKEN`
secret is missing or belongs to a different account.

### Cell 8 — tests and lint

```python
!cd {WORK} && python -m pytest -q -rs {DBP}/tests
!cd {DBP} && python -m ruff check .
```

Run `ruff` from **inside** `thesis_dbp`. Invoked from the parent it treats that
directory as a second project root and writes a stray `.ruff_cache/` outside the write
scope.

`-rs` prints the reason for every skip, so a skip is never mistaken for a pass.

### Cell 9 — lock file and collection

```python
import datetime, shutil, pathlib
stamp = datetime.date.today().strftime("%Y%m%d")
lock = pathlib.Path(f"{DBP}/env/lock/lock_kaggle_{stamp}.txt")
lock.parent.mkdir(parents=True, exist_ok=True)
!python -m pip freeze > {lock}

out = pathlib.Path("/kaggle/working/p0_results"); out.mkdir(exist_ok=True)
for f in pathlib.Path(f"{DBP}/reports").glob("*"):
    shutil.copy(f, out / f.name)
shutil.copy(lock, out / lock.name)
shutil.make_archive("/kaggle/working/p0_results", "zip", out)
print(sorted(p.name for p in out.iterdir()))
```

---

## Step 4 — Send these files back to me

Download `/kaggle/working/p0_results.zip` and paste back the contents of:

| file | what it settles |
|---|---|
| `reports/env_report.json` | GPU, driver, torch/CUDA/cuDNN, package versions, Pillow/libjpeg |
| `reports/p0_bank_smoke.csv` | **the main table** — per-member PSNR, LPIPS, ms/img, peak VRAM, fp16_allowed, determinism |
| `reports/p0_decoder_identity.json` | which decoders are byte-identical (is `sd21` the same as `sd_ft_mse`?) |
| `reports/p0_backbone_smoke.json` | DINOv2 tokens, params, LoRA trainable count, DINOv3 access status |
| `env/lock/lock_kaggle_<date>.txt` | the Kaggle-side dependency lock |
| the printed output of **Cell 5** | that all 12 pins still resolve on the Hub |
| the pytest and ruff output from **Cell 8** | pass/skip/fail with reasons |

Also worth a look before sending: the PNG triplets in `outputs/p0/`. The amplified
residual for each member should show structure, not uniform noise — a flat grey
residual usually means the member silently failed rather than reconstructed.

I will then fill the real numbers into `docs/IMPLEMENTATION_LOG.md`, replacing every
`PENDING-GPU` marker, and update the decoder-bank status table.
