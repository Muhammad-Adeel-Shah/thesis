#!/usr/bin/env bash
# P5 cloud setup (AWS g5.2xlarge / RunPod), run ON the instance from the repo checkout:
#     bash env/cloud_p5_setup.sh /data
# - a venv on the data volume (never the root disk); caches on the data volume;
# - torch / torchvision pinned to the version Kaggle MEASURED in the P5 pilot pre-flight
#   (2026-11-21: torch 2.11.0+cu130, CUDA 13.0, driver 580.178.04), so training (A10G /
#   A5000) and scoring (Kaggle T4) share torch 2.11.0. The CUDA build is TORCH_INDEX
#   (default cu130, as Kaggle); a driver older than R580 cannot run it -- the CUDA check
#   below then fails loudly and `TORCH_INDEX=cu128` installs the same torch on CUDA 12.8.
#   Either way the record states the build used;
# - requirements/base.txt after torch, refused if it changes torch, torchvision or numpy;
# - the environment recorded under reports/p5_cloud/ and env/lock/ (no secret in either).
# - pyarrow, pinned to the version Kaggle preinstalls (the P5 pilot pre-flight measured
#   23.0.1, reports/kaggle_20261121/p5_pilot/preflight.json): approval A3 uses Kaggle's
#   preinstalled pyarrow, so it is not in requirements/base.txt -- but a fresh cloud venv has
#   none, and p5_fetch_reals.py reads the Pexels manifest (parquet) with it. Its absence
#   killed the first AWS session at its first line (2026-11-24);
# - the venv interpreter is P5_VENV_PYTHON (default python3) and must be >= 3.11
#   (diffusers 0.42.0 requires it; the AMI's python3 is 3.10).
# Secrets: HF_TOKEN (and later KAGGLE_USERNAME / KAGGLE_KEY) are exported in the
# interactive shell with `read -rs`; this script only reports whether they are present.
set -euo pipefail
DATA="${1:?usage: bash env/cloud_p5_setup.sh <data volume mount>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
export HF_HOME="$DATA/cache/hf" TORCH_HOME="$DATA/cache/torch" PIP_CACHE_DIR="$DATA/cache/pip"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$REPO/reports/p5_cloud" "$REPO/env/lock"

echo "HF_TOKEN present: $([ -n "${HF_TOKEN:-}" ] && echo true || echo false)"
[ -n "${HF_TOKEN:-}" ] || { echo "FATAL: export HF_TOKEN first (read -rs HF_TOKEN; export HF_TOKEN)"; exit 1; }
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv || {
  echo "FATAL: no NVIDIA GPU visible"; exit 1; }

VENV_PYTHON="${P5_VENV_PYTHON:-python3}"
"$VENV_PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else
    f"FATAL: {sys.executable} is Python {sys.version.split()[0]}; diffusers 0.42.0 needs >= 3.11 "
    "(set P5_VENV_PYTHON)")'
"$VENV_PYTHON" -m venv "$DATA/venv"
# shellcheck disable=SC1091
source "$DATA/venv/bin/activate"
python -m pip install --quiet --upgrade pip
TORCH_INDEX="${TORCH_INDEX:-cu130}"
echo "torch 2.11.0 / torchvision 0.26.0 from the ${TORCH_INDEX} index"
python -m pip install --quiet "torch==2.11.0" "torchvision==0.26.0" \
  --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}"
before="$(python -c 'import torch,torchvision,numpy;print(torch.__version__,torchvision.__version__,numpy.__version__)')"
PYARROW="pyarrow==23.0.1"        # Kaggle's preinstalled version (see the header)
python -m pip install --quiet --dry-run --report "$DATA/pip_dryrun.json" \
  -r "$REPO/requirements/base.txt" "$PYARROW"
python - "$DATA/pip_dryrun.json" <<'PY'
import json, sys
touched = [i["metadata"]["name"] for i in json.load(open(sys.argv[1]))["install"]
           if i["metadata"]["name"].lower() in ("torch", "torchvision", "numpy")]
sys.exit(f"FATAL: requirements would change {touched}") if touched else print("dry-run: torch/torchvision/numpy untouched")
PY
python -m pip install --quiet -r "$REPO/requirements/base.txt" "$PYARROW"
python -m pip install --quiet --no-deps -e "$REPO"
after="$(python -c 'import torch,torchvision,numpy;print(torch.__version__,torchvision.__version__,numpy.__version__)')"
[ "$before" = "$after" ] || { echo "FATAL: protected versions changed: $before -> $after"; exit 1; }

python - <<'PY' | tee "$REPO/reports/p5_cloud/env_${STAMP}.txt"
import platform, subprocess, torch, av, diffusers, transformers, peft, pyarrow
import pyarrow.parquet  # noqa: F401  -- p5_fetch_reals.py reads the Pexels manifest with it
print("python", platform.python_version(), "|", platform.platform())
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "CUDA not available"
for i in range(torch.cuda.device_count()):
    print("gpu", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
print("driver", subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip())
print("av", av.__version__, "| diffusers", diffusers.__version__, "| transformers",
      transformers.__version__, "| peft", peft.__version__, "| pyarrow", pyarrow.__version__)
import sys; sys.path.insert(0, "src")
from dbp.data.video import encoder_inventory
print("encoders", encoder_inventory())
PY
python -m pip freeze > "$REPO/env/lock/cloud_p5_${STAMP}.txt"
echo "SETUP OK: venv $DATA/venv; record reports/p5_cloud/env_${STAMP}.txt; lock env/lock/cloud_p5_${STAMP}.txt"
