"""P3b: record the two frozen ViT-L models' hashes before any Panel B score (local, CPU).

The trainer is byte-identical and S2-D holds no lineage out, so its training report records
no hash of the weights. This script makes the reference the scoring pre-flight verifies
against (user decision 2026-11-12), from the files of the training session's output:

- the final snapshot ``outputs/p3b/s2/<run>/snapshots/step_05000/`` (``lora/`` + ``head.pt``)
  -- the model scored -- and the trainer's own last checkpoint ``<run>/checkpoint.pt``;
- the committed training report (``reports/kaggle_20261111/p3b_train/p2_s2_<run>.json``).

For each run it checks that the report says a finished ViT-L run (status ``OK``, backbone
``dinov3-vitl16`` at its pin, ``steps_run`` = ``stop_at`` = the step), that the checkpoint is
at that step, and that **every LoRA tensor and every head tensor of the snapshot equals the
checkpoint's** (``torch.equal``): the snapshot is the weights the trainer ended with. It
then records the LoRA, head and adapter-config sha256, the snapshot digest
(``snapshot_digest``, as P3's records) and the checkpoint's sha256. Weights are never
committed; only this record is. Exit 0 when every run passes.

Usage:
    python scripts/p3b_model_record.py --run s2d_p35_seed0=<dir> --run s2d_p35_seed1=<dir>
        --reports reports/kaggle_20261111/p3b_train --out reports/p3b_results/p3b_model_record.json

where ``<dir>`` holds ``checkpoint.pt`` and ``snapshots/step_05000/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.backbones.loader import BACKBONES  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.eval.heldout_scores import snapshot_digest  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_s2d_inputs import write_record  # noqa: E402

BACKBONE = "dinov3-vitl16"
STEP = 5000
#: peft's live state-dict keys carry the adapter name; the saved adapter drops it.
ADAPTER = ".default"


def saved_key(live: str) -> str:
    """A live LoRA state-dict key as ``save_pretrained`` writes it."""
    return live.replace(ADAPTER, "")


def compare(checkpoint: dict[str, Any], lora: dict[str, Any], head: dict[str, Any]
            ) -> list[str]:
    """Every tensor of the snapshot equals the checkpoint's; the list of differences."""
    import torch

    problems: list[str] = []
    live = {saved_key(k): v for k, v in checkpoint["model"].items()}
    if set(live) != set(lora):
        problems.append(f"LoRA keys differ: {len(set(live) ^ set(lora))} not in both")
    problems += [f"LoRA {k} differs" for k in sorted(set(live) & set(lora))
                 if not torch.equal(live[k].cpu(), lora[k].cpu())]
    if set(checkpoint["head"]) != set(head):
        problems.append("head keys differ")
    problems += [f"head {k} differs" for k in sorted(set(checkpoint["head"]) & set(head))
                 if not torch.equal(checkpoint["head"][k].cpu(), head[k].cpu())]
    return problems


def check_report(report: dict[str, Any], step: int) -> list[str]:
    """The training report says a finished ViT-L run at ``step``."""
    want = {"status": "OK", "mode": "train", "backbone": BACKBONE,
            "revision": BACKBONES[BACKBONE]["revision"], "steps_run": step, "stop_at": step}
    return [f"report {k} {report.get(k)!r}, expected {v!r}" for k, v in want.items()
            if report.get(k) != v]


def record_run(run: str, directory: Path, report_path: Path, step: int) -> dict[str, Any]:
    """One run's checks and hashes."""
    import torch
    from safetensors.torch import load_file

    snapshot = directory / "snapshots" / f"step_{step:05d}"
    files = {"adapter": snapshot / "lora" / "adapter_model.safetensors",
             "config": snapshot / "lora" / "adapter_config.json",
             "head": snapshot / "head.pt", "checkpoint": directory / "checkpoint.pt"}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing or not report_path.is_file():
        return {"run": run, "ok": False,
                "problems": [f"missing: {missing or [str(report_path)]}"]}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    problems = check_report(report, step)
    checkpoint = torch.load(files["checkpoint"], map_location="cpu", weights_only=True)
    if checkpoint.get("step") != step:
        problems.append(f"checkpoint at step {checkpoint.get('step')}, expected {step}")
    head = torch.load(files["head"], map_location="cpu", weights_only=True)
    problems += compare(checkpoint, load_file(str(files["adapter"])), head)
    return {
        "run": run, "step": step, "ok": not problems, "problems": problems,
        "snapshot": f"outputs/p3b/s2/{run}/snapshots/step_{step:05d}",
        "snapshot_sha256": snapshot_digest(snapshot),
        "lora_sha256": file_sha256(files["adapter"]),
        "adapter_config_sha256": file_sha256(files["config"]),
        "head_sha256": file_sha256(files["head"]),
        "checkpoint_sha256": file_sha256(files["checkpoint"]),
        "bytes": {k: p.stat().st_size for k, p in files.items()},
        "report": report_path.relative_to(THESIS_DBP).as_posix()
        if report_path.is_relative_to(THESIS_DBP) else str(report_path),
        "report_sha256": file_sha256(report_path),
        "tensors_equal_to_checkpoint": not any("differ" in p for p in problems),
    }


def main() -> int:
    """Record every run; 0 when all pass."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="RUN=DIR")
    parser.add_argument("--reports", type=Path, required=True,
                        help="Directory of the committed training reports p2_s2_<run>.json.")
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    log = setup_logging("p3b_model_record")
    runs = {}
    for item in args.run:
        run, sep, path = item.partition("=")
        if not sep:
            log.error("--run expects RUN=DIR, got %r", item)
            return 1
        runs[run] = record_run(run, Path(path), args.reports / f"p2_s2_{run}.json", args.step)
    record = {"what": "P3b frozen ViT-L models: hashes recorded before any Panel B score",
              "backbone": BACKBONE, "revision": BACKBONES[BACKBONE]["revision"],
              "runs": runs, "ok": all(r["ok"] for r in runs.values())}
    write_record(args.out, record)
    for run, entry in runs.items():
        log.info("%s: %s %s", run, "OK" if entry["ok"] else "FAILED",
                 entry.get("snapshot_sha256", "") if entry["ok"] else entry["problems"])
    log.info("MODEL RECORD: %s -> %s", "OK" if record["ok"] else "FAILED", args.out)
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
