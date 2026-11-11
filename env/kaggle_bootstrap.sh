#!/usr/bin/env bash
# ===========================================================================
# DBP -- Kaggle bootstrap (idempotent; safe to re-run in the same session)
# ===========================================================================
# Enforces guardrail #4: the preinstalled torch / torchvision / numpy are NEVER
# upgraded, downgraded or reinstalled. The script performs a pip resolution
# dry-run first and ABORTS if the resolution would touch any of them.
#
# Usage (from a Kaggle cell):  bash thesis_dbp/env/kaggle_bootstrap.sh
#
# This is the ONLY environment step a runbook may take (CLAUDE.md section 12). A cell
# that pip-installs one package instead is how spandrel went missing and one GPU sat
# idle for 3.5 hours in the first S3 commit.
#
# DBP_DRYRUN=1 (the local runbook dry-run, CPU): the GPU checks are skipped, nothing is
# installed or uninstalled, and the editable install is skipped -- the resolution is
# still computed and the guardrail still enforced, so a requirement that would move
# torch fails here too. The local venv is never modified by a dry-run.
#
# DBP_CPU_SESSION=1 (a Kaggle CPU session, e.g. the P3 Panel B draw): the GPU checks
# are skipped; everything else -- the resolution, the guardrail, the installs -- runs.
#
# DBP_WITH_GDOWN=1 (P3 Panel B only, CLAUDE.md section 6a(c)): installs gdown, pinned to
# the version P1 used, with --no-deps, inside the session only (never in requirements/).
# torch / torchvision / numpy are printed before and after and must be unchanged, and
# `import gdown` must work, or the run stops.
# ===========================================================================
set -euo pipefail

DBP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROTECTED="torch torchvision numpy"
DRYRUN="${DBP_DRYRUN:-0}"
CPU_SESSION="${DBP_CPU_SESSION:-0}"
WITH_GDOWN="${DBP_WITH_GDOWN:-0}"
GDOWN_PIN="gdown==5.2.0"
if [ "${CPU_SESSION}" = "1" ]; then
    echo "CPU SESSION: GPU checks skipped; installs and guardrails run."
fi
if [ "${DRYRUN}" = "1" ]; then
    echo "DRY-RUN MODE: GPU checks skipped; no package will be installed or removed."
fi

echo "=== [1/7] runtime ============================================="
python - <<'PY'
import sys, platform
print("python  ", sys.version.split()[0], platform.platform())
PY

if [ "${DRYRUN}" = "1" ] || [ "${CPU_SESSION}" = "1" ]; then
    echo "(nvidia-smi and CUDA checks skipped: dry-run or CPU session)"
elif ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi not found. Enable a GPU accelerator for this Kaggle session." >&2
    exit 1
else
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

[ "${DRYRUN}" = "1" ] || [ "${CPU_SESSION}" = "1" ] || python - <<'PY'
import sys, torch
if not torch.cuda.is_available():
    sys.exit("FATAL: torch.cuda.is_available() is False. Set Accelerator = GPU T4 x2.")
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
sm = cap[0] * 10 + cap[1]
print(f"gpu      {name}  sm_{sm}  cuda={torch.version.cuda}  cudnn={torch.backends.cudnn.version()}")
if sm < 70:
    sys.exit(f"FATAL: {name} is sm_{sm}. PyTorch CUDA-13 builds dropped pre-Turing "
             f"(sm_70). Switch the accelerator to T4 (sm_75). [R19]")
if sm == 60:
    sys.exit("FATAL: P100 (sm_60) is unsupported. Switch to T4.")
if sm != 75:
    print(f"WARNING: expected T4 (sm_75) but found sm_{sm}. "
          f"fp16 parity thresholds in the P0 report were calibrated on T4.")
PY

echo
echo "=== [2/7] recording protected package versions ================"
BEFORE="$(python - <<'PY'
import importlib.metadata as md
print(" ".join(f"{p}=={md.version(p)}" for p in ("torch", "torchvision", "numpy")))
PY
)"
echo "before: ${BEFORE}"

echo
echo "=== [3/7] pip resolution dry-run (guardrail #4) ==============="
DRY="$(mktemp)"
python -m pip install --dry-run --quiet --report "${DRY}" -r "${DBP_DIR}/requirements/base.txt" >/dev/null
python - "${DRY}" ${PROTECTED} <<'PY'
import json, sys
report, protected = sys.argv[1], set(sys.argv[2:])
plan = json.load(open(report, encoding="utf-8")).get("install", [])
names = {i["metadata"]["name"].lower().replace("_", "-"): i["metadata"]["version"] for i in plan}
print("would install/change:", ", ".join(f"{k}=={v}" for k, v in sorted(names.items())) or "(nothing)")
hits = protected & names.keys()
if hits:
    sys.exit(f"FATAL (guardrail #4): resolution would change {sorted(hits)}. "
             f"Pin an older compatible version of the requesting library instead, or stop and ask.")
print("OK: torch / torchvision / numpy are untouched by this resolution.")
PY
rm -f "${DRY}"

echo
echo "=== [4/7] install ============================================="
# --no-deps is NOT used: the dry-run above already proved the full dependency
# closure is safe, and we want the real transitive deps (e.g. einops).
#
# --quiet is NOT used either. In P0 it swallowed the real reason the editable
# install failed (the repo was mounted read-only under /kaggle/input) and left
# only "error: subprocess-exited-with-error" on screen, which cost a session.
if [ "${DRYRUN}" = "1" ]; then
    echo "(dry-run: install skipped; the resolution above is what Kaggle would install)"
else
python -m pip install -r "${DBP_DIR}/requirements/base.txt"
fi

# The editable install is a convenience, not a requirement: every script inserts
# src/ on sys.path itself. If pip cannot write an .egg-link -- a read-only mount,
# most often -- say so plainly and fall back to PYTHONPATH rather than aborting.
if [ "${DRYRUN}" = "1" ]; then
    echo "(dry-run: editable install skipped -- it would repoint the local venv)"
elif python -m pip install -e "${DBP_DIR}"; then
    echo "OK: editable install succeeded."
else
    echo "WARNING: 'pip install -e' failed (see the error above)."
    echo "  Falling back to PYTHONPATH. The usual cause is a read-only source tree:"
    echo "  copy the repo into /kaggle/working and re-run from there."
    export PYTHONPATH="${DBP_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    echo "  PYTHONPATH = ${PYTHONPATH}"
fi

echo
echo "=== [5/7] verifying protected packages are unchanged =========="
AFTER="$(python - <<'PY'
import importlib.metadata as md
print(" ".join(f"{p}=={md.version(p)}" for p in ("torch", "torchvision", "numpy")))
PY
)"
echo "after : ${AFTER}"
if [ "${BEFORE}" != "${AFTER}" ]; then
    echo "FATAL: protected packages changed during install." >&2
    echo "  before: ${BEFORE}" >&2
    echo "  after : ${AFTER}"  >&2
    exit 1
fi
echo "OK: unchanged."
python -m pip check || echo "WARNING: pip check reported conflicts (see above)."

if [ "${WITH_GDOWN}" = "1" ]; then
    echo
    echo "=== [5b/7] gdown (opt-in; session only; ${GDOWN_PIN}, --no-deps) ==="
    CORE_BEFORE="$(python -c 'import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)')"
    echo "torch torchvision numpy before: ${CORE_BEFORE}"
    if [ "${DRYRUN}" = "1" ]; then
        echo "(dry-run: would run: python -m pip install --no-deps ${GDOWN_PIN})"
    else
        python -m pip install --no-deps --quiet "${GDOWN_PIN}"
    fi
    CORE_AFTER="$(python -c 'import torch, torchvision, numpy; print(torch.__version__, torchvision.__version__, numpy.__version__)')"
    echo "torch torchvision numpy after : ${CORE_AFTER}"
    if [ "${CORE_BEFORE}" != "${CORE_AFTER}" ]; then
        echo "FATAL: installing gdown changed torch/torchvision/numpy." >&2
        exit 1
    fi
    if [ "${DRYRUN}" != "1" ]; then
        python -c "import gdown; print('gdown', gdown.__version__)" || {
            echo "FATAL: import gdown failed after the --no-deps install." >&2; exit 1; }
    fi
fi

echo
echo "=== [6/7] environment ========================================="
# Non-persisted cache: protects the ~20 GB /kaggle/working quota.
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export TORCH_HOME="${TORCH_HOME:-/tmp/torch_home}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"
echo "HF_HOME    = ${HF_HOME}"
echo "TORCH_HOME = ${TORCH_HOME}"

# HF token: Kaggle Secrets first, then the environment. Never printed.
python - <<'PY'
import os
tok = None
try:
    from kaggle_secrets import UserSecretsClient
    tok = UserSecretsClient().get_secret("HF_TOKEN")
except ImportError:
    print("  (kaggle_secrets not importable; falling back to the environment)")
except Exception as exc:
    # Never a silent except (guardrail #8): say which lookup failed and why.
    print(f"  (Kaggle Secrets lookup failed: {type(exc).__name__}: {exc}; "
          f"falling back to the environment)")
tok = tok or os.environ.get("HF_TOKEN")
print(f"HF_TOKEN present: {bool(tok)}")
if not tok:
    print("  -> sd35, flux and dinov3 will be reported BLOCKED-ACCESS.")
    print("  -> Add a Kaggle Secret named HF_TOKEN (Add-ons > Secrets) to enable them.")
PY

echo
echo "=== [7/7] torchao conflict check (issue I-011) ================"
set +e
# torchao 0.10.0 ships with the Kaggle image and breaks the peft import path with
# "only versions above 0.16.0 are supported", which took out both LoRA dry-runs in
# P0. It is not a dependency of this project, so it is detected here and the exact
# uninstall command is printed rather than run: removing a package from someone
# else's image is the operator's decision, not the bootstrap's.
python - <<'PY'
import importlib.metadata as md
import sys

try:
    version = md.version("torchao")
except md.PackageNotFoundError:
    print("torchao not installed: nothing to do.")
else:
    print(f"torchao {version} is installed.")
    # Importing peft is NOT a sufficient probe: on Kaggle 2026-09 the import succeeds
    # and the failure surfaces later, inside get_peft_model -> dispatch_torchao ->
    # is_torchao_available(), which RAISES when torchao < 0.16.0 instead of returning
    # False. That took out both LoRA dry-runs in the P1 preflight after this step had
    # reported "no action needed" (I-033). Probe the function that actually raises.
    blocked = None
    try:
        import peft  # noqa: F401
    except Exception as exc:
        blocked = f"peft cannot be imported: {type(exc).__name__}: {exc}"
    else:
        try:
            from peft.import_utils import is_torchao_available
        except ImportError as exc:
            print(f"  peft has no import_utils.is_torchao_available ({exc}); "
                  "probe skipped, LoRA path unverified.")
        else:
            try:
                is_torchao_available()
            except Exception as exc:
                blocked = f"peft's LoRA dispatcher raises: {type(exc).__name__}: {exc}"

    if blocked:
        print(f"  {blocked}")
        print("  This is issue I-011: torchao will be removed below.")
        sys.exit(3)
    print("  peft imports AND its LoRA dispatcher resolves; no action is needed.")
PY
TORCHAO_STATUS=$?
set -e
if [ "${TORCHAO_STATUS}" = "3" ]; then
    if [ "${DRYRUN}" = "1" ]; then
        echo "  (dry-run: torchao would be uninstalled here)"
    else
        # Removing it here rather than printing the command: a runbook cell that has to
        # read this output and act on it is a step nobody takes in commit mode. torchao
        # is not a DBP dependency and removing it does not touch torch. Every stage
        # runs in a fresh subprocess, so no kernel restart is needed afterwards.
        python -m pip uninstall -y torchao
        python - <<'PY'
from peft.import_utils import is_torchao_available
is_torchao_available()
print("  torchao removed; peft's LoRA dispatcher now resolves.")
PY
    fi
elif [ "${TORCHAO_STATUS}" != "0" ]; then
    echo "FATAL: the torchao probe itself failed (exit ${TORCHAO_STATUS})." >&2
    exit 1
fi

echo
echo "bootstrap complete. Next:"
echo "  P0 re-check : python ${DBP_DIR}/scripts/p0_env_check.py"
echo "                python ${DBP_DIR}/scripts/p0_bank_smoke_test.py"
echo "                python ${DBP_DIR}/scripts/p0_backbone_smoke_test.py"
echo "  P1          : see ${DBP_DIR}/env/kaggle_run_p1_preflight.md"
