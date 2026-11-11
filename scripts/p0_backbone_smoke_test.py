"""P0 backbone smoke test and LoRA attach dry-run (section 6.5).

Loads the DINO-family backbone, reports its shape and cost, then attaches PEFT LoRA
adapters to the attention projections **that are actually present** and runs a single
forward pass. No optimizer is created and no training step is taken.

Usage:
    python scripts/p0_backbone_smoke_test.py --device cpu
    python scripts/p0_backbone_smoke_test.py --device cuda --backbones dinov2-base,dinov3-vitb16
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.backbones.loader import (  # noqa: E402
    BACKBONES,
    DEFAULT_INPUT_SIZE,
    BackboneLoadError,
    find_attention_projection_names,
    forward_once,
    load_backbone,
    lora_dry_run,
)
from dbp.utils.hf import get_hf_token  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

#: Forward passes discarded before the timed one (issue I-009).
TIMING_WARMUPS = 3


def _is_access_error(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("gatedrepo", "401", "403", "awaiting a review", "access to model")
    )


def run_one(
    name: str, device: torch.device, size: int, log: Any, warmups: int = TIMING_WARMUPS
) -> dict[str, Any]:
    """Load one backbone, measure it and run the LoRA dry-run."""
    entry = BACKBONES[name]
    result: dict[str, Any] = {
        "name": name,
        "repo": entry["repo"],
        "revision": entry["revision"],
        "gated": entry["gated"],
        "license": entry["license"],
        "status": "NOT-RUN",
    }

    try:
        backbone = load_backbone(name, device=device, token=get_hf_token())
    except BackboneLoadError as exc:
        message = str(exc)
        result["status"] = "BLOCKED-ACCESS" if _is_access_error(message) else "ERROR"
        result["detail"] = message[:400]
        log.error("%-14s %s: %s", name, result["status"], message[:200])
        return result

    is_cuda = device.type == "cuda"
    result.update(
        {
            "hidden_size": backbone.hidden_size,
            "patch_size": backbone.patch_size,
            "num_parameters": backbone.num_parameters,
            "config": backbone.config,
        }
    )
    log.info(
        "%-14s hidden=%d patch=%d params=%.1fM",
        name,
        backbone.hidden_size,
        backbone.patch_size,
        backbone.num_parameters / 1e6,
    )

    if size % backbone.patch_size:
        log.warning(
            "%-14s input %d is not divisible by patch %d; the model may pad or crop.",
            name,
            size,
            backbone.patch_size,
        )

    try:
        forward = forward_once(
            backbone.model,
            size=size,
            device=device,
            use_fp16_autocast=is_cuda,
            warmups=warmups,
        )
        result["forward"] = forward
        expected_patches = (size // backbone.patch_size) ** 2
        result["token_breakdown"] = {
            "total": forward["num_tokens"],
            "patch_tokens_expected": expected_patches,
            "cls_plus_registers": forward["num_tokens"] - expected_patches,
            "num_register_tokens_config": backbone.config.get("num_register_tokens", 0),
        }
        log.info(
            "%-14s tokens=%d (%d patch + %d cls/registers) forward=%.1f ms peak=%s MB fp16=%s",
            name,
            forward["num_tokens"],
            expected_patches,
            forward["num_tokens"] - expected_patches,
            forward["forward_ms"],
            forward["peak_gpu_mb"] if is_cuda else "PENDING-GPU",
            forward["fp16_autocast"],
        )
        if not is_cuda:
            result["forward"]["peak_gpu_mb"] = "PENDING-GPU"
            result["forward"]["cpu_forward_ms"] = forward.pop("forward_ms")
            result["forward"]["forward_ms"] = "PENDING-GPU"
    except Exception as exc:
        result["status"] = "ERROR"
        result["detail"] = f"forward failed: {type(exc).__name__}: {exc}"[:400]
        log.error("%-14s forward ERROR %s", name, result["detail"])
        return result

    discovered = find_attention_projection_names(backbone.model)
    result["attention_projection_modules_present"] = discovered
    log.info("%-14s attention projections present: %s", name, discovered)

    try:
        result["lora"] = lora_dry_run(backbone.model, r=8, lora_alpha=8, size=size, device=device)
        log.info(
            "%-14s LoRA r=8 -> %d trainable / %d total (%.4f%%)",
            name,
            result["lora"]["params_trainable"],
            result["lora"]["params_total_after"],
            result["lora"]["trainable_pct"],
        )
        result["status"] = "OK"
    except Exception as exc:
        result["status"] = "ERROR"
        result["detail"] = f"lora dry-run failed: {type(exc).__name__}: {exc}"[:400]
        log.error("%-14s LoRA ERROR %s", name, result["detail"])

    del backbone
    if is_cuda:
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 backbone smoke test.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--size", type=int, default=DEFAULT_INPUT_SIZE)
    parser.add_argument(
        "--warmups",
        type=int,
        default=TIMING_WARMUPS,
        help="Forward passes discarded before the timed one (issue I-009).",
    )
    args = parser.parse_args()

    log = setup_logging("p0_backbone_smoke", log_dir=THESIS_DBP / "outputs" / "logs")
    log.info("seed: %s", seed_everything(2026, deterministic=True))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log.error("--device cuda requested but no CUDA device is available.")
        return 2
    log.info("device: %s | input size: %d", device, args.size)

    names = [n.strip() for n in args.backbones.split(",") if n.strip()]
    unknown = [n for n in names if n not in BACKBONES]
    if unknown:
        log.error("unknown backbone(s): %s; have %s", unknown, sorted(BACKBONES))
        return 2

    results = [run_one(name, device, args.size, log, args.warmups) for name in names]

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": str(device),
        "input_size": args.size,
        "hf_token_present": get_hf_token() is not None,
        "backbones": results,
    }
    dest = THESIS_DBP / "reports" / "p0_backbone_smoke.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", dest)

    for result in results:
        if result["status"] != "OK":
            log.warning("%-14s %s %s", result["name"], result["status"], result.get("detail", ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
