"""P3: per-image held-out scores of a saved S2 snapshot (validation data only, never Panel B).

Approved 2026-10-17 (D2 (ii)). The LOLO training reports kept each held-out AUC but not
the per-image scores the H1-a bootstrap resamples, and H1-c pairs LOLO-f with ``BANK-1``
image by image. This script re-scores a snapshot exactly as the trainer scored it during
training -- the same module (:mod:`dbp.eval.heldout_scores`), keys, crop and batch -- and
writes ``image_id, member, label, score`` plus a sidecar with the snapshot's sha256.

Two modes:

* **one snapshot:** ``--run-dir <outputs/p3/s2/<run id>> --protocol P --seed S --step N``;
* **the LOLO folds at their frozen ``N*(-f)``:** ``--lolo-root <thesis_dbp root holding
  outputs/p3/s2/lolo_f*>`` ``--nstar reports/p3_nstar.json``. Each fold is scored at the
  step the committed selection names, on its held-out members, and the rescored AUC is
  recorded beside the AUC its training report gave at that step.

The held-out members default to every distinct member the run does not train on.
``--limit N`` scores only the first N reals and N reconstructions (pre-flight smoke).

Usage:
    python scripts/p3_score_heldout.py --lolo-root /kaggle/input/.../thesis_dbp \\
        --nstar reports/p3_nstar.json --root /kaggle/working/p1 \\
        --pairs-manifest .../p1_pairs_manifest.csv --splits .../p2_splits.csv \\
        --out-dir reports/p3_heldout_scores
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.backbones.loader import BACKBONES, load_backbone  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.heldout_scores import (  # noqa: E402
    heldout_rows,
    score_heldout,
    score_items,
    snapshot_digest,
    write_scores,
)
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.streams.s1_semantic import (  # noqa: E402
    expected_patch_tokens,
    pool_tokens,
    processor_normalisation,
    register_token_count,
)
from dbp.utils.hf import get_hf_token  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import run_id  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402
from dbp.utils.watchdog import start_watchdog  # noqa: E402

#: The trainer's batch size: the same batches give the same fp16 numerics.
BATCH_SIZE = 16


def snapshot_dir(run_dir: Path, step: int) -> Path:
    """``<run dir>/snapshots/step_NNNNN``."""
    return run_dir / "snapshots" / f"step_{step:05d}"


def score_snapshot(args: argparse.Namespace, protocol: str, seed: int, run_dir: Path,
                   step: int, out: Path, log: Any,
                   reported: float | None = None) -> dict[str, Any]:
    """Score one snapshot's held-out set and write its per-image file."""
    import torch

    from p2_s2_infer import load_selected_model
    from p2_train_s2 import load_reals_index

    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    run = load_run_configs(args.runs_config, bank).by_id(protocol)
    active = set(run.active_members(bank))
    members = sorted(s.key for s in bank.distinct_members() if s.key not in active)
    if not members:
        raise ValueError(f"{protocol} trains on every member: nothing is held out")
    snapshot = snapshot_dir(run_dir, step)
    reals_index = load_reals_index(args.manifests)
    rows = heldout_rows(args.splits, args.pairs_manifest, set(members), set(reals_index))
    reals = sorted({row["image_id"] for row in rows})
    if args.limit:
        reals, rows = reals[: args.limit], [r for r in rows if r["image_id"] in
                                            set(reals[: args.limit])][: args.limit]
    if not rows:
        raise ValueError(f"{protocol}: no validation reconstructions of {members}")

    token = get_hf_token()
    # Seeded exactly as the trainer seeds before it builds the backbone: the pretrained
    # DINOv3 does not depend on it, but the dry-run's random-init backbone does, and a
    # LoRA rescored on a different base would not reproduce its training AUC.
    seed_everything(seed)
    backbone = load_backbone(args.backbone, args.device, token)
    n_register = register_token_count(backbone.config)
    n_patch = expected_patch_tokens(run.eval_crop, backbone.patch_size)
    normalisation = processor_normalisation(backbone.repo, backbone.revision, token)
    model, head = load_selected_model(
        backbone, SimpleNamespace(lora=snapshot / "lora", head=snapshot / "head.pt",
                                  run_dir=snapshot), args.device)

    def forward(batch: Any) -> Any:
        hidden = model(pixel_values=batch).last_hidden_state
        return head(pool_tokens(hidden.float(), n_register, n_patch))

    amp = args.device.startswith("cuda") and torch.cuda.is_available()
    started = time.perf_counter()
    table = score_heldout(reals, rows, args.root, reals_index, lambda items: score_items(
        items, forward, normalisation, run.eval_crop, args.device, amp, BATCH_SIZE))
    seconds = time.perf_counter() - started
    meta = {
        "run_id": run_id(protocol, seed), "protocol": protocol, "seed": seed, "step": step,
        "snapshot": snapshot.relative_to(run_dir).as_posix(),
        "snapshot_sha256": snapshot_digest(snapshot), "held_out_members": members,
        "crop": run.eval_crop, "backbone": backbone.name, "revision": backbone.revision,
        "dry_run_random_init": bool(backbone.config.get("dry_run_random_init")),
        "limit": args.limit or None, "seconds": round(seconds, 1),
        "scored_by": "p3_score_heldout.py (rescored from the snapshot)",
        "reported_heldout_auc": reported,
    }
    write_scores(out, table, meta)
    written = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    diff = None if reported is None or written["heldout_auc"] is None \
        else round(written["heldout_auc"] - reported, 6)
    log.info("%s step %d: %d reals, %d reconstructions in %.1f s; held-out AUC %s "
             "(report %s, diff %s) -> %s", protocol, step, written["reals"],
             written["reconstructions"], seconds, written["heldout_auc"], reported, diff,
             out.name)
    return {"protocol": protocol, "step": step, "csv": out.name,
            "heldout_auc": written["heldout_auc"], "reported": reported, "diff": diff}


def main() -> int:
    """Score one snapshot, or every LOLO fold at its frozen N*(-f)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="One-snapshot mode: the CSV.")
    parser.add_argument("--lolo-root", type=Path, default=None)
    parser.add_argument("--nstar", type=Path, default=THESIS_DBP / "reports" / "p3_nstar.json")
    parser.add_argument("--folds", default="1,2,3,4,5,6,7,8")
    parser.add_argument("--out-dir", type=Path,
                        default=THESIS_DBP / "reports" / "p3_heldout_scores")
    parser.add_argument("--root", required=True, type=Path, help="Corpus root.")
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--manifests", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--runs-config", type=Path,
                        default=THESIS_DBP / "configs" / "p3_runs.yaml")
    parser.add_argument("--backbone", default="dinov3-vitb16", choices=sorted(BACKBONES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--deadline-hours", type=float, default=0.0,
                        help="Hard wall-clock limit: past it the process exits with 2, "
                             "even if a step hangs (0 = none).")
    args = parser.parse_args()
    log = setup_logging("p3_score_heldout")
    log.info("HF_TOKEN present: %s", get_hf_token() is not None)
    if args.deadline_hours > 0:
        start_watchdog(args.deadline_hours, log)

    if args.lolo_root is None:
        if not (args.run_dir and args.protocol and args.step and args.out):
            log.error("one-snapshot mode needs --run-dir, --protocol, --step and --out")
            return 2
        score_snapshot(args, args.protocol, args.seed, args.run_dir, args.step, args.out, log)
        return 0

    selection = json.loads(args.nstar.read_text(encoding="utf-8"))
    results = []
    for fold in [int(f) for f in args.folds.split(",") if f]:
        step = int(selection["nested"][str(fold)]["step"])
        protocol = f"lolo_f{fold}"
        rid = run_id(protocol, 0)
        run_dir = args.lolo_root / "outputs" / "p3" / "s2" / rid
        report = args.lolo_root / "reports" / f"p2_s2_{rid}.json"
        if not snapshot_dir(run_dir, step).is_dir() or not report.is_file():
            log.error("%s: no snapshot at step %d or no training report under %s", protocol,
                      step, args.lolo_root)
            return 1
        curve = json.loads(report.read_text(encoding="utf-8"))["curve"]
        reported = next(p["heldout_auc"] for p in curve if p["step"] == step)
        out = args.out_dir / f"{rid}_step_{step:05d}.csv"
        results.append(score_snapshot(args, protocol, 0, run_dir, step, out, log,
                                      None if args.limit else reported))
    summary = args.out_dir / "p3_lolo_rescore_summary.json"
    tmp = summary.with_name(summary.name + ".tmp")
    tmp.write_text(json.dumps({
        "nstar": args.nstar.as_posix(), "lolo_root": args.lolo_root.as_posix(),
        "limit": args.limit or None, "folds": results}, indent=2) + "\n",
        encoding="utf-8", newline="\n")
    tmp.replace(summary)
    log.info("rescored %d fold(s); summary %s", len(results), summary.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
