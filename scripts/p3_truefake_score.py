"""P3 TrueFake (secondary, descriptive), step 2: score the 4 frozen models, once (one worker
per GPU).

Run by ``p2_run_shards.py`` as ``p3_truefake_score.py --members A,B --device cuda <args>``,
after ``p3_truefake_prepare.py`` has written the plan. For each model in ``--members``:

* the weights are re-hashed and must equal the plan's, then loaded the way every S2
  scorer loads them (``load_selected_model``);
* every row of the scoring list that is neither dropped (disjointness) nor too small for
  the crop is scored with the **deployed protocol** -- the run's centre crop (320), no
  augmentation, the head logit -- from the local copy the pre-flight verified; every
  version of an id is scored independently;
* one CSV (``image_id, score``) at ``reports/p3_truefake_scores/<model>/truefake.csv``, and
  the summary ``reports/p3_truefake/<model>.json`` last, so a summary means done.

**Once only:** a model whose summary exists is refused (exit 1). ``--limit N`` (the smoke)
scores N rows into ``--out-root`` and is never a result. Per-image scores stay private
(git-ignored); only aggregates are committed.
"""

from __future__ import annotations

import argparse
import json
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
from dbp.decoders.registry import load_bank_config  # noqa: E402
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
from p3_final_score import (  # noqa: E402
    ScoreError,
    read_list,
    refuses_rescore,
    verify_weights,
    write_scores,
)
from p3_truefake_prepare import SCORES, SUMMARIES, scorable  # noqa: E402

LOADERS = 4


def scored_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """The rows the protocol scores: not dropped, not too small for the crop, decodable."""
    return [r for r in rows if scorable(r)]


def score_model(model: dict[str, Any], plan: dict[str, Any], args: argparse.Namespace,
                log: Any) -> dict[str, Any]:
    """Score one model on the TrueFake list; write its CSV, then its summary."""
    import torch

    from p2_s2_infer import load_selected_model

    out_root = args.out_root or THESIS_DBP
    summary_file = out_root / SUMMARIES / f"{model['id']}.json"
    if refuses_rescore(summary_file, args.limit):
        raise ScoreError(f"{model['id']}: already scored ({summary_file}); TrueFake is "
                         f"scored once per frozen model")
    verify_weights(model)
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

    def crop(row: dict[str, str]) -> np.ndarray:
        return centre_crop(load_rgb_uint8(local / row["relative_path"]), run.eval_crop)

    rows = scored_rows(read_list(Path(plan["list"]["path"])))
    rows = rows[: args.limit] if args.limit else rows
    scores: list[float] = []
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=LOADERS) as pool:
        for start in range(0, len(rows), args.batch):
            chunk = rows[start: start + args.batch]
            crops = np.stack(list(pool.map(crop, chunk)))
            batch = normalise_batch(crops, normalisation["image_mean"],
                                    normalisation["image_std"]).to(args.device)
            with torch.no_grad():
                # fp16 autocast for the backbone only, as p2_s2_infer.py scores.
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    hidden = net(pixel_values=batch).last_hidden_state
                logits = head(pool_tokens(hidden.float(), n_register, n_patch))
            scores.extend(float(v) for v in logits.float().cpu().numpy().ravel())
    if not np.all(np.isfinite(scores)):
        raise ScoreError(f"{model['id']}: non-finite scores")
    seconds = time.perf_counter() - started
    path = out_root / SCORES / model["id"] / "truefake.csv"
    result = {"rows": len(rows), "file": str(path), "sha256": write_scores(path, rows, scores),
              "seconds": round(seconds, 1), "seconds_per_image": round(seconds / len(rows), 5)}
    log.info("%s truefake: %d rows in %.1f s (%.4f s/image)", model["id"], len(rows), seconds,
             seconds / len(rows))
    from PIL import __version__ as pillow

    summary = {
        "model": model["id"], "stage": "truefake", "protocol": model["protocol"],
        "seed": model["seed"], "step": model.get("step"), "kind": model["kind"],
        "snapshot_sha256": model.get("snapshot_sha256"), "lora_sha256": model["lora_sha256"],
        "head_sha256": model["head_sha256"], "crop": run.eval_crop, "crop_policy": "centre",
        "score": "head logit", "backbone": backbone.name, "revision": backbone.revision,
        "dry_run_random_init": bool(backbone.config.get("dry_run_random_init")),
        "pillow": pillow, "limit": args.limit or None, "lists": {"truefake": result},
        "list_sha256": plan["list"]["sha256"], "sample_sha256": plan["sample"]["sha256"],
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
                        "p3_truefake_plan.json")
    parser.add_argument("--backbone", default="dinov3-vitb16")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--limit", type=int, default=0, help="Smoke: N rows.")
    parser.add_argument("--out-root", type=Path, default=None,
                        help="Where a smoke writes (never the results tree).")
    parser.add_argument("--deadline-hours", type=float, default=0.0)
    args = parser.parse_args()
    log = setup_logging("p3_truefake_score")
    if args.deadline_hours > 0:
        start_watchdog(args.deadline_hours, log)
    if args.limit and args.out_root is None:
        log.error("--limit is a smoke and needs --out-root outside the results tree")
        return 2
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if not plan.get("ok") or plan.get("stage") != "truefake":
        log.error("the plan is not ok or has nothing to score: %s", plan.get("stage"))
        return 1
    for model_id in [m for m in args.members.split(",") if m]:
        if model_id not in plan["due"]:
            log.error("%s is not due in this plan", model_id)
            return 1
        try:
            score_model(plan["models"][model_id], plan, args, log)
        except ScoreError as error:
            log.error("%s", error)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
