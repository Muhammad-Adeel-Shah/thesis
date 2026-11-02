# Local development setup (Windows / CPU)

The local environment is for **development and CPU smoke tests only**. Every number
that appears in the thesis comes from the Kaggle T4 run (see `kaggle_run_p0.md`).

Audited host (P0, 2026-09-19): Windows 11, Python 3.13.1, 17 GB RAM, no NVIDIA GPU,
125 GB free on `F:`, 32 GB free on `C:`.

## 1. Create the virtual environment

Guardrail #4 forbids installing into the system Python. Everything lives in
`thesis_dbp/.venv` (git-ignored).

```powershell
cd F:\thesis-2\Implementation
python -m venv thesis_dbp\.venv
thesis_dbp\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. Redirect the caches to F:

`C:` is 83 % full and the bank weights are ~4.9 GB, so **all** large caches are
pointed at `F:`. Set these per-process only; do not set them machine-wide.

```powershell
$root = "F:\thesis-2\Implementation\thesis_dbp"
$env:HF_HOME        = "$root\.cache\hf"      # ~4.3 GB of VAE + backbone weights
$env:TORCH_HOME     = "$root\.cache\torch"   # 553 MB VGG16 that LPIPS downloads
$env:PIP_CACHE_DIR  = "$root\.cache\pip"
```

## 3. Install torch (CPU build) explicitly, before anything else

Installing torch *first* and *by exact version* means no later resolve can
substitute a different build. Use the official PyTorch selector index for CPU:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

> On a machine **with** an NVIDIA GPU, pick the matching CUDA index from
> <https://pytorch.org/get-started/locally/> instead. Never mix the two.

## 4. Install the project

```powershell
python -m pip install -r thesis_dbp\requirements\dev.txt
python -m pip install -e thesis_dbp
python -m pip check
```

## 5. Provide the HF token

The token is read from `os.environ["HF_TOKEN"]` only. It is never printed, logged
or written to disk. Set it as a **Windows user environment variable** and restart
the terminal, or put it in `REPO_ROOT/.env` (already git-ignored).

Gated repositories requiring an accepted licence:
`stabilityai/stable-diffusion-3.5-large`, `black-forest-labs/FLUX.1-schnell`,
`facebook/dinov3-vitb16-pretrain-lvd1689m`.

## 6. Run the P0 checks

```powershell
python thesis_dbp\scripts\p0_env_check.py
python thesis_dbp\scripts\p0_bank_smoke_test.py --device cpu
python thesis_dbp\scripts\p0_backbone_smoke_test.py --device cpu
python thesis_dbp\scripts\p0_verify_integrity.py --self-test
pytest -q -rs thesis_dbp\tests
cd thesis_dbp; ruff check .; cd ..
```

> Run `ruff` **from inside `thesis_dbp`**. Invoked from `REPO_ROOT` it treats the
> repository root as a second project root and writes `REPO_ROOT\.ruff_cache\`, which is
> outside the P0 write scope and trips the protected-asset integrity check.

CPU runs report timings in the `cpu_ms_per_img` column. The `ms_per_img`,
`peak_gpu_mb`, `fp16_allowed` and `OOM@1024` columns stay `PENDING-GPU`, because
fp16 numerics and VRAM limits cannot be reproduced on CPU.

## 7. Write the lock file

After a green run:

```powershell
python -m pip freeze > thesis_dbp\env\lock\lock_local_20260919.txt
```
