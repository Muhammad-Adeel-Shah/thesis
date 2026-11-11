"""P0 environment check.

Writes ``reports/env_report.json`` and prints a human-readable summary. Contains no
secrets: only ``hf_token_present: true|false``.

Usage:
    python scripts/p0_env_check.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
REPO_ROOT = THESIS_DBP.parent
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.utils.env_report import build_report, write_report  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect the DBP environment report.")
    parser.add_argument(
        "--out",
        type=Path,
        default=THESIS_DBP / "reports" / "env_report.json",
        help="Destination JSON path.",
    )
    args = parser.parse_args()

    log = setup_logging("p0_env_check", log_dir=THESIS_DBP / "outputs" / "logs")

    report = build_report(REPO_ROOT, bank_config=THESIS_DBP / "configs" / "decoder_bank.yaml")
    write_report(report, args.out)

    py, torch_info, pkgs = report["python"], report["torch"], report["packages"]
    log.info("OS            : %s", py["platform"])
    log.info("Python        : %s (venv=%s)", py["version"], py["in_venv"])
    log.info(
        "torch         : %s | cuda=%s | cudnn=%s | is_available=%s",
        torch_info.get("torch_version"),
        torch_info.get("torch_cuda_version"),
        torch_info.get("cudnn_version"),
        torch_info.get("cuda_is_available"),
    )
    log.info("torchvision   : %s", torch_info.get("torchvision_version"))

    if report["nvidia_smi"]["available"]:
        for gpu in report["nvidia_smi"]["gpus"]:
            log.info("GPU           : %s", gpu)
        for dev in torch_info.get("devices", []):
            log.info(
                "  device %d    : %s sm_%d %d MB",
                dev["index"],
                dev["name"],
                dev["sm"],
                dev["total_memory_mb"],
            )
            if dev["sm"] == 60:
                log.warning(
                    "P100 (sm_60) detected. PyTorch CUDA-13 builds dropped pre-Turing "
                    "support; switch the Kaggle accelerator to T4 (sm_75). [R19]"
                )
    else:
        log.warning("GPU           : NONE (%s)", report["nvidia_smi"]["reason"])
        log.warning("               fp16 parity, peak VRAM and 1024 crops will be PENDING-GPU.")

    log.info(
        "Pillow        : %s (libjpeg %s)",
        report["pillow"].get("pillow"),
        report["pillow"].get("libjpeg"),
    )
    log.info(
        "key packages  : diffusers=%s transformers=%s peft=%s spandrel=%s lpips=%s",
        pkgs.get("diffusers"),
        pkgs.get("transformers"),
        pkgs.get("peft"),
        pkgs.get("spandrel"),
        pkgs.get("lpips"),
    )
    for label, disk in report["disk"].items():
        if "free_gb" in disk:
            log.info("disk %-9s: %.1f GB free of %.1f GB", label, disk["free_gb"], disk["total_gb"])
    log.info("RAM           : %s GB", report["ram_gb"])
    log.info("network       : %s", report["network"])
    log.info("HF_TOKEN present: %s", report["hf_token_present"])

    git = report["git"]
    if git.get("available"):
        log.info(
            "git           : %s @ %s (dirty=%s)",
            git["branch"],
            git["commit"][:12],
            git["dirty"],
        )

    bank = report.get("bank")
    if bank:
        log.info("bank          : %d members, all revisions pinned", bank["member_count"])

    log.info("report written: %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
