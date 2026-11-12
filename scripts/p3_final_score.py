"""P3 final scoring, step 2: score frozen models on Panel B, once (one worker per GPU).

Run by ``p2_run_shards.py`` as ``p3_final_score.py --members A,B --device cuda <args>``,
after ``p3_final_preflight.py`` has written the plan. For each model in ``--members``:

* the weights are re-hashed and must equal the plan's (nothing changed since the
  pre-flight), then loaded the way every S2 scorer loads them (``load_selected_model``);
* **clean stage:** Panel B (9,405), its luma companion (700) and the 891 threshold reals
  are scored with the deployed protocol -- the run's centre crop, no augmentation, the
  head logit -- from the local copy the pre-flight verified;
* **robust stage (H5):** Panel B's test rows under each of the 12 conditions of
  ``dbp.eval.perturb``, applied to the whole image before the centre crop;
* one CSV per list (``image_id, score``) under
  ``reports/p3_final_scores/<model>/<stage>/``, and
  the summary ``reports/p3_final/<model>_<stage>.json`` last, so a summary means done.

**Once only:** a model whose stage summary exists is refused (exit 1) -- the pre-flight
carries complete stages forward and never plans them again. ``--limit N`` (the smoke)
scores N rows per list into ``--out-root`` and is never a result.

**Per-list resume** (``--resume-lists``, opt-in; P3b, user decision 2026-11-12): every
finished list is recorded the moment it ends (``<list>.done.json`` beside its CSV: rows,
the CSV's sha256, the model's hashes, the list file's sha256, the backbone). A later run
**keeps** every list whose record matches the model, the list and the backbone, and whose
CSV has the record's hash -- it is never scored again -- and scores only the lists with no
saved scores. A record that does not match is refused, never overwritten. A CSV with no
record (a crash between the two writes) holds scores nobody recorded; it is scored again
and the summary names it. Without the flag nothing changes.

**Timing** for the fail-fast projection: each list times its first batch apart
(``timing``: the warm images and seconds after it), and the summary records the model's
``load_seconds``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.backbones.loader import load_backbone  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.heldout_scores import snapshot_digest  # noqa: E402
from dbp.eval.perturb import CONDITIONS, apply_condition  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.streams.s1_semantic import (  # noqa: E402
    expected_patch_tokens,
    normalise_batch,
    pool_tokens,
    processor_normalisation,
    register_token_count,
)
from dbp.train.augment import centre_crop  # noqa: E402
from dbp.utils.hf import get_hf_token  # noqa: E402
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402
from dbp.utils.watchdog import start_watchdog  # noqa: E402
from p3_final_preflight import CLEAN_LISTS, FINAL, SUMMARIES  # noqa: E402

LOADERS = 4


class ScoreError(RuntimeError):
    """A model that may not or cannot be scored."""


def read_list(path: Path) -> list[dict[str, str]]:
    """A scoring list written by the pre-flight."""
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_scores(path: Path, rows: list[dict[str, str]], scores: list[float]) -> str:
    """Atomic ``image_id, score`` CSV; returns its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "score"])
        for row, score in zip(rows, scores, strict=True):
            writer.writerow([row["image_id"], f"{score:.6f}"])
    tmp.replace(path)
    return file_sha256(path)


def refuses_rescore(summary_file: Path, limit: int) -> bool:
    """Once only: a real scoring is refused when its summary exists.

    A smoke (``--limit``) writes only to its scratch root and is never a result, so an
    earlier smoke's summary there does not block it (a dry-run's scratch outlives a run).
    """
    return summary_file.is_file() and not limit


def verify_weights(model: dict[str, Any]) -> None:
    """The weights are still the ones the pre-flight hashed."""
    now = {"lora_sha256": file_sha256(Path(model["lora"]) / "adapter_model.safetensors"),
           "head_sha256": file_sha256(Path(model["head"]))}
    if model["kind"] == "snapshot":
        now["snapshot_sha256"] = snapshot_digest(Path(model["weights"]))
    changed = [k for k, v in now.items() if v != model[k]]
    if changed:
        raise ScoreError(f"{model['id']}: {changed} changed since the pre-flight")


def fault_after_lists() -> int | None:
    """Dry-run only: exit right after this many lists are recorded (rehearses a crash)."""
    value = os.environ.get("DBP_FAULT_EXIT_AFTER_LISTS", "")
    return int(value) if value and os.environ.get("DBP_DRYRUN") == "1" else None


def list_record_path(csv_path: Path) -> Path:
    """The per-list record beside a list's CSV (``--resume-lists``)."""
    return csv_path.with_name(csv_path.stem + ".done.json")


def list_identity(model: dict[str, Any], list_sha256: str, backbone: str,
                  condition: str) -> dict[str, Any]:
    """What a kept list must have been scored with."""
    return {"model": model["id"], "lora_sha256": model["lora_sha256"],
            "head_sha256": model["head_sha256"],
            "snapshot_sha256": model.get("snapshot_sha256"), "list_sha256": list_sha256,
            "backbone": backbone, "condition": condition}


def kept_list(csv_path: Path, identity: dict[str, Any]) -> dict[str, Any] | None:
    """A finished list to keep, or None to score it.

    Raises:
        ScoreError: a record exists that does not match this model, list or backbone, or
            whose CSV is not the one it recorded -- never overwritten.
    """
    record_path = list_record_path(csv_path)
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    differ = [k for k, v in identity.items() if record.get(k) != v]
    if differ:
        raise ScoreError(f"{csv_path}: its record was made with a different {differ}; "
                         f"refused, not overwritten")
    if not csv_path.is_file() or file_sha256(csv_path) != record.get("sha256"):
        raise ScoreError(f"{csv_path}: the CSV is missing or not the one its record names")
    return record


def write_list_record(csv_path: Path, identity: dict[str, Any], entry: dict[str, Any]) -> None:
    """Record a finished list (atomic), after its CSV."""
    record_path = list_record_path(csv_path)
    tmp = record_path.with_name(record_path.name + ".tmp")
    tmp.write_text(json.dumps({**identity, **entry}, indent=2) + "\n", encoding="utf-8",
                   newline="\n")
    tmp.replace(record_path)


def score_model(model: dict[str, Any], plan: dict[str, Any], args: argparse.Namespace,
                log: Any) -> dict[str, Any]:
    """Score one model's stage; write its CSVs, then its summary."""
    import torch

    from p2_s2_infer import load_selected_model

    stage_name = plan["stage"]
    out_root = args.out_root or THESIS_DBP
    summary_file = out_root / SUMMARIES / f"{model['id']}_{stage_name}.json"
    if refuses_rescore(summary_file, args.limit):
        raise ScoreError(f"{model['id']}: {stage_name} already scored ({summary_file}); "
                         f"Panel B is scored once per frozen model")
    verify_weights(model)
    loading = time.perf_counter()
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    run = load_run_configs(THESIS_DBP / model["runs_config"], bank).by_id(model["protocol"])
    token = get_hf_token()
    seed_everything(model["seed"])
    backbone = load_backbone(args.backbone, args.device, token)
    n_register = register_token_count(backbone.config)
    n_patch = expected_patch_tokens(run.eval_crop, backbone.patch_size)
    normalisation = processor_normalisation(backbone.repo, backbone.revision, token)
    net, head = load_selected_model(backbone, SimpleNamespace(
        lora=Path(model["lora"]), head=Path(model["head"]), run_dir=Path(model["weights"])),
        args.device)
    amp = args.device.startswith("cuda") and torch.cuda.is_available()
    local = Path(plan["local_root"])
    load_seconds = time.perf_counter() - loading
    first_batch: dict[str, float] = {}

    def crop(row: dict[str, str], condition: str) -> np.ndarray:
        image = load_rgb_uint8(local / row["relative_path"])
        return centre_crop(apply_condition(image, condition), run.eval_crop)

    def score(rows: list[dict[str, str]], condition: str) -> list[float]:
        out: list[float] = []
        began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=LOADERS) as pool:
            for start in range(0, len(rows), args.batch):
                if start == args.batch:   # the first batch (warm-up) is timed apart
                    first_batch["seconds"] = time.perf_counter() - began
                chunk = rows[start: start + args.batch]
                crops = np.stack(list(pool.map(lambda r: crop(r, condition), chunk)))
                batch = normalise_batch(crops, normalisation["image_mean"],
                                        normalisation["image_std"]).to(args.device)
                with torch.no_grad():
                    # fp16 autocast for the backbone only, as p2_s2_infer.py scores.
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        hidden = net(pixel_values=batch).last_hidden_state
                    logits = head(pool_tokens(hidden.float(), n_register, n_patch))
                out.extend(float(v) for v in logits.float().cpu().numpy().ravel())
        if not np.all(np.isfinite(out)):
            raise ScoreError(f"{model['id']}: non-finite scores ({condition})")
        return out

    if stage_name == "clean":
        jobs = [(name, read_list(Path(plan["lists"][name]["path"])), "clean")
                for name in CLEAN_LISTS]
    else:
        test = [r for r in read_list(Path(plan["lists"]["panel_b"]["path"]))
                if r["role"] == "test"]
        jobs = [(condition, test, condition) for condition in plan["robust"]["conditions"]]
    list_sha = {name: file_sha256(Path(plan["lists"][name]["path"])) for name in CLEAN_LISTS
                if name in plan.get("lists", {})}
    results: dict[str, Any] = {}
    kept, unrecorded = [], []
    for name, rows, condition in jobs:
        rows = rows[: args.limit] if args.limit else rows
        path = out_root / FINAL / model["id"] / stage_name / f"{name}.csv"
        source = "panel_b" if stage_name == "robust" else name
        identity = list_identity(model, list_sha.get(source, ""), args.backbone, condition)
        if args.resume_lists and not args.limit:
            record = kept_list(path, identity)
            if record is not None:
                results[name] = {k: record[k] for k in ("rows", "condition", "file", "sha256",
                                                        "seconds", "seconds_per_image",
                                                        "timing")}
                kept.append(name)
                log.info("%s %s %s: KEPT (scored earlier, %d rows; never scored again)",
                         model["id"], stage_name, name, record["rows"])
                continue
            if path.is_file():
                unrecorded.append(name)
                log.warning("%s %s %s: a CSV with no record (a crash between two writes) is "
                            "scored again", model["id"], stage_name, name)
        first_batch.clear()
        started = time.perf_counter()
        scores = score(rows, condition)
        seconds = time.perf_counter() - started
        warm_images = max(0, len(rows) - args.batch)
        timing = {"first_batch_seconds": round(first_batch.get("seconds", seconds), 3),
                  "warm_images": warm_images,
                  "warm_seconds": round(seconds - first_batch["seconds"], 3)
                  if warm_images else 0.0}
        results[name] = {"rows": len(rows), "condition": condition, "file": str(path),
                         "sha256": write_scores(path, rows, scores),
                         "seconds": round(seconds, 1),
                         "seconds_per_image": round(seconds / len(rows), 5), "timing": timing}
        if args.resume_lists and not args.limit:
            write_list_record(path, identity, results[name])
            fault = fault_after_lists()
            if fault is not None and len(results) - len(kept) >= fault:
                log.error("dry-run fault: exiting after %d recorded list(s)", fault)
                os._exit(137)
        log.info("%s %s %s: %d rows in %.1f s (%.4f s/image)", model["id"], stage_name, name,
                 len(rows), seconds, seconds / len(rows))
    from PIL import __version__ as pillow

    summary = {
        "model": model["id"], "stage": stage_name, "protocol": model["protocol"],
        "seed": model["seed"], "step": model.get("step"), "kind": model["kind"],
        "snapshot_sha256": model.get("snapshot_sha256"), "lora_sha256": model["lora_sha256"],
        "head_sha256": model["head_sha256"], "crop": run.eval_crop, "crop_policy": "centre",
        "score": "head logit", "backbone": backbone.name, "revision": backbone.revision,
        "dry_run_random_init": bool(backbone.config.get("dry_run_random_init")),
        "pillow": pillow, "conditions": {c: CONDITIONS[c] for _, _, c in jobs},
        "limit": args.limit or None, "lists": results,
        "panel_b_digest": plan["panel_b"]["digest"],
        "load_seconds": round(load_seconds, 2),
        **({"kept_lists": kept, "unrecorded_lists_scored_again": unrecorded}
           if args.resume_lists else {}),
    }
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = summary_file.with_name(summary_file.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(summary_file)
    return summary


def main() -> int:
    """Score every model in ``--members``; 0 when all were scored."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--members", required=True, help="Comma-separated model ids.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan", type=Path, default=THESIS_DBP / "reports" /
                        "p3_final_plan.json")
    parser.add_argument("--backbone", default="dinov3-vitb16")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Smoke: N rows per list.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Where a smoke writes (never the results tree).")
    parser.add_argument("--deadline-hours", type=float, default=0.0)
    parser.add_argument("--resume-lists", action="store_true",
                        help="Record each finished list; keep recorded lists on a re-run.")
    args = parser.parse_args()
    log = setup_logging("p3_final_score")
    if args.deadline_hours > 0:
        start_watchdog(args.deadline_hours, log)
    if args.limit and args.out_root is None:
        log.error("--limit is a smoke and needs --out-root outside the results tree")
        return 2
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not plan.get("ok") or plan.get("stage") not in ("clean", "robust"):
        log.error("the plan is not ok or has nothing to score: %s", plan.get("stage"))
        return 1
    for model_id in [m for m in args.members.split(",") if m]:
        if model_id not in plan["due"]:
            log.error("%s is not due in this plan (%s)", model_id, plan["stage"])
            return 1
        try:
            score_model(plan["models"][model_id], plan, args, log)
        except ScoreError as error:
            log.error("%s", error)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
