#!/usr/bin/env bash
# P5 cloud setup (AWS g5.2xlarge / RunPod), run ON the instance from the repo checkout:
#     bash env/cloud_p5_setup.sh /data
# - a venv on the data volume (never the root disk); caches on the data volume;
# - torch / torchvision pinned to the versions Kaggle records (snapshot: 2.10.0+cu128 /
#   0.25.0+cu128), so training (A10G / A5000) and scoring (Kaggle T4) share them;
# - requirements/base.txt after torch, refused if it changes torch, torchvision or numpy;
# - the environment recorded under reports/p5_cloud/ and env/lock/ (no secret in either).
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

python3 -m venv "$DATA/venv"
# shellcheck disable=SC1091
source "$DATA/venv/bin/activate"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet "torch==2.10.0" "torchvision==0.25.0" \
  --index-url https://download.pytorch.org/whl/cu128
before="$(python -c 'import torch,torchvision,numpy;print(torch.__version__,torchvision.__version__,numpy.__version__)')"
python -m pip install --quiet --dry-run --report "$DATA/pip_dryrun.json" -r "$REPO/requirements/base.txt"
python - "$DATA/pip_dryrun.json" <<'PY'
import json, sys
touched = [i["metadata"]["name"] for i in json.load(open(sys.argv[1]))["install"]
           if i["metadata"]["name"].lower() in ("torch", "torchvision", "numpy")]
sys.exit(f"FATAL: requirements would change {touched}") if touched else print("dry-run: torch/torchvision/numpy untouched")
PY
python -m pip install --quiet -r "$REPO/requirements/base.txt"
python -m pip install --quiet --no-deps -e "$REPO"
after="$(python -c 'import torch,torchvision,numpy;print(torch.__version__,torchvision.__version__,numpy.__version__)')"
[ "$before" = "$after" ] || { echo "FATAL: protected versions changed: $before -> $after"; exit 1; }

python - <<'PY' | tee "$REPO/reports/p5_cloud/env_${STAMP}.txt"
import platform, subprocess, torch, av, diffusers, transformers, peft
print("python", platform.python_version(), "|", platform.platform())
print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
assert torch.cuda.is_available(), "CUDA not available"
for i in range(torch.cuda.device_count()):
    print("gpu", i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
print("driver", subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True).stdout.strip())
print("av", av.__version__, "| diffusers", diffusers.__version__, "| transformers",
      transformers.__version__, "| peft", peft.__version__)
import sys; sys.path.insert(0, "src")
from dbp.data.video import encoder_inventory
print("encoders", encoder_inventory())
PY
python -m pip freeze > "$REPO/env/lock/cloud_p5_${STAMP}.txt"
echo "SETUP OK: venv $DATA/venv; record reports/p5_cloud/env_${STAMP}.txt; lock env/lock/cloud_p5_${STAMP}.txt"
