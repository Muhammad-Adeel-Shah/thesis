"""P5 pre-flight (CLAUDE.md section 12.2): refuse before anything long starts.

Checks, all recorded in ``reports/p5_pilot/preflight.json``:
- **encoders** (approval A3): PyAV's FFmpeg has ``libx264`` and ``libx265``, and a
  25-frame clip round-trips through each at the input frame count; the FFmpeg library
  versions are recorded;
- **imports**: ``av``, ``pyarrow`` (Kaggle's preinstalled one), and every diffusers video
  class the bank names, with the diffusers version;
- **the frozen S2-D models**: each LoRA and head sha256 equals the P5 expectation **and**
  the committed P3 record (``env/p3_final_expect.json``) -- two sources that must agree;
- **the threshold file** (``p3_analysis_full.json``): its sha256 equals the expectation;
- **disk**: free space under each given root;
- **pins** (CLAUDE.md 7a): every video-bank commit and VAE file, via
  ``p0_verify_pins.require_video_pins`` -- skipped only with ``--skip-pins`` (dry-run).

Exit 0 = every check passed, 1 = at least one failed.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.data.video import encode_roundtrip, encoder_inventory  # noqa: E402
from dbp.decoders.registry import load_video_bank  # noqa: E402


def check_encoders(problems: list[str]) -> dict[str, Any]:
    inventory = encoder_inventory()
    clip = np.random.default_rng(0).integers(0, 256, (25, 64, 64, 3), dtype=np.uint8)
    for codec in ("libx264", "libx265"):
        if not inventory[codec]:
            problems.append(f"encoder {codec} is not available in PyAV's FFmpeg")
            continue
        try:
            frames, size = encode_roundtrip(clip, codec, 28, 30)
            inventory[f"{codec}_roundtrip"] = {"frames": len(frames), "bytes": size}
        except Exception as exc:
            problems.append(f"{codec} round-trip failed: {type(exc).__name__}: {exc}")
    return inventory


def check_imports(problems: list[str], bank: Any, names: list[str]) -> dict[str, str]:
    """Every module the expectation lists (the Kaggle one includes the preinstalled
    ``pyarrow``), and every diffusers class the bank names."""
    found: dict[str, str] = {}
    for name in dict.fromkeys(["av", "diffusers", *names]):
        try:
            found[name] = getattr(importlib.import_module(name), "__version__", "?")
        except ImportError as exc:
            problems.append(f"import {name}: {exc}")
    if "diffusers" in found:
        import diffusers

        for spec in bank.select(("bank", "reserve")):
            if not hasattr(diffusers, spec.cls):
                problems.append(f"diffusers {found['diffusers']} has no {spec.cls} ({spec.key})")
    return found


def check_models(problems: list[str], expect: dict[str, Any], s2d_root: Path) -> list[dict]:
    record = json.loads((THESIS_DBP / "env" / "p3_final_expect.json").read_text("utf-8"))
    committed = {m["id"]: m["expect"] for m in record["models"]}
    out = []
    for model in expect["models"]:
        row: dict[str, Any] = {"id": model["id"]}
        if not expect.get("dryrun_fixture") and committed.get(model["id"]) != {
                "lora_sha256": model["lora_sha256"], "head_sha256": model["head_sha256"]}:
            problems.append(f"{model['id']}: the P5 expectation differs from the committed "
                            f"P3 record env/p3_final_expect.json")
        lora = s2d_root / model["lora"] / "adapter_model.safetensors"
        head = s2d_root / model["head"]
        for label, path, want in (("lora", lora, model["lora_sha256"]),
                                  ("head", head, model["head_sha256"])):
            if not path.is_file():
                problems.append(f"{model['id']}: no {label} at {path}")
                continue
            got = file_sha256(path)
            row[f"{label}_sha256"] = got
            if got != want:
                problems.append(f"{model['id']}: {label} sha256 {got} != {want}")
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--s2d-root", type=Path, required=True)
    parser.add_argument("--disk", action="append", default=[], metavar="PATH=GB")
    parser.add_argument("--skip-pins", action="store_true", help="dry-run only")
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports" / "p5_pilot")
    args = parser.parse_args()
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    bank = load_video_bank(THESIS_DBP / "configs" / "video_bank.yaml")
    problems: list[str] = []
    import os

    if (expect.get("dryrun_fixture") or args.skip_pins) and os.environ.get("DBP_DRYRUN") != "1":
        problems.append("dryrun_fixture / --skip-pins are dry-run seams; refused outside "
                        "DBP_DRYRUN=1")
    result: dict[str, Any] = {"encoders": check_encoders(problems),
                              "imports": check_imports(problems, bank, expect["imports"]),
                              "models": check_models(problems, expect, args.s2d_root)}
    thresholds = THESIS_DBP / expect["thresholds"]["path"]
    got = file_sha256(thresholds) if thresholds.is_file() else None
    result["thresholds_sha256"] = got
    if got != expect["thresholds"]["sha256"]:
        problems.append(f"{thresholds}: sha256 {got} != {expect['thresholds']['sha256']}")
    result["disk_free_gb"] = {}
    for item in args.disk:
        path, _, need = item.partition("=")
        Path(path).mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path).free / 1e9
        result["disk_free_gb"][path] = round(free, 1)
        if free < float(need):
            problems.append(f"{path}: {free:.1f} GB free < {need} GB needed")
    if args.skip_pins:
        result["pins"] = "SKIPPED (--skip-pins, dry-run)"
    else:
        from p0_verify_pins import PinVerificationError, require_video_pins

        try:
            result["pins"] = [r.status for r in require_video_pins()]
        except PinVerificationError as exc:
            problems.append(str(exc))
    result["problems"] = problems
    result["ok"] = not problems
    args.reports.mkdir(parents=True, exist_ok=True)
    tmp = args.reports / "preflight.json.tmp"
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    tmp.replace(args.reports / "preflight.json")
    print(json.dumps(result, indent=1)[:4000])
    print("PRE-FLIGHT", "PASSED" if not problems else f"FAILED: {problems}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
