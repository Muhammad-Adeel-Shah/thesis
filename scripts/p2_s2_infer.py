"""P2 Session D: score image lists with one trained S2 run.

The worker that trained a run scores with it straight away, so S2 scoring needs no
GPU session of its own. Three lists per run:

* ``panel`` -- the frozen test panel (11,775 rows), scored once;
* ``luma`` -- the FLUX.1 luma-only companion (978 rows, ids ``<id>__luma``), which the
  pre-registration (section 8) requires before the standard FLUX.1 number is claimable;
* ``val`` -- the validation rows of the S1 list (``view == val``). ``p2_fit_heads.py``
  calibrates S2 and fits both fusions on validation scores only; without them S2 would
  drop out of every fusion.

**The model scored with is the run's selected one**: the LoRA adapters and head saved at
the best validation AUC (``outputs/p2/s2/<run id>/lora`` and ``head.pt``), which is
what early stopping selects -- not the last step's training checkpoint. Images are
centre-cropped to the run's ``eval_crop`` (320, D-045), exactly as S1 and S3 see them.

Each list is written to ``outputs/p2/s2/<run id>/scores/<list>.csv`` (``image_id,score``,
the head's logit), and a summary with every file's sha256 to
``reports/p2_s2_scores_<run id>.json``.

Usage:
    python scripts/p2_s2_infer.py --protocol p_a_sd1 --seed 0 --root /kaggle/working/p1 \\
        --panel-root /kaggle/input/.../panel --list panel=reports/p2_panel_images.csv \\
        --list luma=reports/p2_luma_images.csv --list val=reports/p2_s1_images.csv \\
        --only val=val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.backbones.loader import BACKBONES, load_backbone  # noqa: E402
from dbp.data.manifest import file_sha256, write_manifest  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.streams.s1_semantic import (  # noqa: E402
    expected_patch_tokens,
    normalise_batch,
    pool_tokens,
    processor_normalisation,
    register_token_count,
)
from dbp.streams.s2_artifact import build_head  # noqa: E402
from dbp.train.augment import centre_crop  # noqa: E402
from dbp.utils.hf import get_hf_token  # noqa: E402
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import run_id, s2_run_paths  # noqa: E402
from dbp.utils.watchdog import start_watchdog  # noqa: E402

SCORE_COLUMNS = ("image_id", "score")


def parse_named(items: list[str], flag: str) -> dict[str, str]:
    """``NAME=VALUE`` pairs; the value may itself contain ``=`` or ``:``."""
    out: dict[str, str] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.strip() or not value.strip():
            raise ValueError(f"--{flag} expects NAME=VALUE, got {item!r}")
        if name in out:
            raise ValueError(f"--{flag} names {name!r} twice")
        out[name.strip()] = value.strip()
    return out


def select_rows(
    rows: list[dict[str, str]], view: str | None, limit: int | None
) -> list[dict[str, str]]:
    """Rows of one list: only ``view`` when given, then the first ``limit``."""
    if view is not None:
        rows = [r for r in rows if r.get("view") == view]
    return rows[:limit] if limit else rows


def load_selected_model(backbone: Any, run_paths: Any, device: str) -> tuple[Any, Any]:
    """The run's selected LoRA adapters on the backbone, and its head.

    Raises:
        FileNotFoundError: If the run has no selected model.
        RuntimeError: If no LoRA layer was attached -- scoring the bare backbone with
            a trained head would produce plausible numbers that mean nothing.
    """
    import torch
    from peft import PeftModel

    if not (run_paths.lora / "adapter_config.json").is_file() or not run_paths.head.is_file():
        raise FileNotFoundError(f"no selected model under {run_paths.run_dir}")
    model = PeftModel.from_pretrained(backbone.model, str(run_paths.lora), is_trainable=False)
    adapted = sum(1 for name, _ in model.named_modules() if name.endswith("lora_A"))
    if adapted == 0:
        raise RuntimeError(f"{run_paths.lora}: no LoRA layer was attached")
    model.to(device).eval()
    head = build_head(backbone.hidden_size)
    head.load_state_dict(torch.load(run_paths.head, map_location="cpu", weights_only=True))
    head.to(device).eval()
    return model, head


def main() -> int:
    """Score every named list with one run. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", required=True, type=Path, help="Corpus root.")
    parser.add_argument("--panel-root", type=Path, default=None)
    parser.add_argument("--list", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--only", action="append", default=[], metavar="NAME=VIEW",
                        help="Keep only rows of this view in list NAME.")
    parser.add_argument("--limit", type=int, default=0, help="Smoke test: first N rows.")
    parser.add_argument("--backbone", default="dinov3-vitb16", choices=sorted(BACKBONES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "s2")
    parser.add_argument("--reports-dir", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--runs-config", type=Path,
                        default=THESIS_DBP / "configs" / "p2_runs.yaml",
                        help="The run file the protocol is defined in (P3: p3_runs.yaml).")
    parser.add_argument("--deadline-hours", type=float, default=0.0,
                        help="Hard wall-clock limit; past it the process exits with 2.")
    args = parser.parse_args()
    log = setup_logging("p2_s2_infer")
    if args.deadline_hours > 0:
        start_watchdog(args.deadline_hours, log)

    import torch

    import p2_probe_features as probe

    try:
        lists = parse_named(args.list, "list")
        views = parse_named(args.only, "only")
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if not lists or set(views) - set(lists):
        log.error("give at least one --list; --only may only name a listed NAME")
        return 2

    rid = run_id(args.protocol, args.seed)
    paths = s2_run_paths(args.out_dir, args.reports_dir, rid)
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    run = load_run_configs(args.runs_config, bank).by_id(args.protocol)

    token = get_hf_token()
    log.info("HF_TOKEN present: %s", token is not None)
    backbone = load_backbone(args.backbone, args.device, token)
    n_register = register_token_count(backbone.config)
    n_patch = expected_patch_tokens(run.eval_crop, backbone.patch_size)
    normalisation = processor_normalisation(backbone.repo, backbone.revision, token)
    model, head = load_selected_model(backbone, paths, args.device)
    log.info("%s: selected model from %s, crop %d", rid, paths.run_dir, run.eval_crop)

    roots = probe.resolve_roots(args.root, args.panel_root)
    use_amp = args.device.startswith("cuda")
    summary_lists: dict[str, Any] = {}
    paths.scores_dir.mkdir(parents=True, exist_ok=True)
    for name, list_path in lists.items():
        rows = select_rows(probe.read_image_list(Path(list_path)), views.get(name),
                           args.limit or None)
        if not rows:
            log.error("list %s (%s, view %s) selects no rows", name, list_path, views.get(name))
            return 2
        started = time.perf_counter()
        scores: list[float] = []
        for start in range(0, len(rows), args.batch):
            chunk = rows[start : start + args.batch]
            crops = np.stack([centre_crop(load_rgb_uint8(probe.row_path(r, roots)),
                                          run.eval_crop) for r in chunk])
            batch = normalise_batch(
                crops, normalisation["image_mean"], normalisation["image_std"]
            ).to(args.device)
            with torch.no_grad():
                # fp16 autocast, never bf16: the T4 is sm_75 (as in training).
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    hidden = model(pixel_values=batch).last_hidden_state
                logits = head(pool_tokens(hidden.float(), n_register, n_patch))
            scores.extend(float(v) for v in logits.float().cpu().numpy())
            if start and (start // args.batch) % 50 == 0:
                log.info("  %s %d/%d", name, start, len(rows))
        if not np.all(np.isfinite(scores)):
            log.error("list %s: %d non-finite score(s)", name, int((~np.isfinite(scores)).sum()))
            return 1
        out = paths.scores(name)
        write_manifest([{"image_id": r["image_id"], "score": f"{s:.6f}"}
                        for r, s in zip(rows, scores, strict=True)], out, SCORE_COLUMNS)
        seconds = time.perf_counter() - started
        summary_lists[name] = {
            "source": Path(list_path).name, "view": views.get(name), "rows": len(rows),
            "scores": str(out.relative_to(paths.run_dir)), "sha256": file_sha256(out),
            "seconds": round(seconds, 1), "seconds_per_image": round(seconds / len(rows), 5),
        }
        log.info("%s: %d row(s) in %.1f s (%.4f s/image) -> %s", name, len(rows), seconds,
                 seconds / len(rows), out.name)

    trained = (json.loads(paths.report.read_text(encoding="utf-8"))
               if paths.report.is_file() else {})
    summary = {
        "run_id": rid, "protocol": args.protocol, "seed": args.seed,
        "backbone": backbone.name, "revision": backbone.revision,
        "dry_run_random_init": bool(backbone.config.get("dry_run_random_init")),
        "crop": run.eval_crop, "crop_policy": "centre", "score": "head logit",
        "model": "selected at best validation AUC",
        "best_step": trained.get("best_step"), "best_val_auc": trained.get("best_val_auc"),
        "lora_sha256": file_sha256(paths.lora / "adapter_model.safetensors"),
        "head_sha256": file_sha256(paths.head),
        "limit": args.limit or None, "lists": summary_lists,
    }
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.scores_summary_tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    paths.scores_summary_tmp.replace(paths.scores_summary)
    log.info("wrote %s", paths.scores_summary.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
