"""P2-B4: train the LoRA artifact stream on content-aligned pairs.

One run is one ``(protocol, seed)``. The protocol is a run id from
``configs/p2_runs.yaml``, and it decides exactly one thing: which decoder lineages the
training fakes may come from. Everything else -- backbone, augmentation, optimiser,
batch shape -- is identical across protocols, so a difference between two runs is a
difference in the training bank and nothing else.

**The batch is pairs.** Eight reals and their eight partners, one augmentation draw per
pair, plus ``gen_matched`` rows at a 25% share of the fakes. Splitting a pair across
two steps would let batch statistics differ between the classes, which is the shortcut
the pairing exists to remove.

**The step budget is measured, not assumed.** No committed measurement of a training
step on this backbone exists -- P0's LoRA entry is explicitly a dry run with no
optimizer -- so ``--calibrate`` times real steps and
:func:`dbp.streams.s2_artifact.steps_within_allowance` converts a GPU-hour allowance
into a step ceiling. If the allowance buys fewer than 2,000 steps the run **stops and
asks**: an undertrained S2 would not be a weak result, it would be an invalid
comparison.

Only the LoRA adapters and the head are saved as the *result*; the backbone is
unchanged by definition and re-downloads from its pin.

**Resumable.** Every 500 steps the run writes a checkpoint holding the adapters, the
head, the optimizer moments, the GradScaler, the LR schedule, the augmentation RNG and
the validation curve so far. ``--resume`` (the default) picks the latest one up and
**continues** the step count and the curve rather than restarting them, and it reuses
the step budget recorded in the checkpoint rather than re-calibrating, so a resumed run
cannot quietly acquire a different ceiling from the one it started with.

**Smoke mode** (``--smoke N``) is the Session D pre-flight: N real steps -- forward,
backward, optimiser -- then a checkpoint save, a reload into fresh objects, and a
tensor-by-tensor comparison. It never resumes and never applies the step floor, and it
writes only under the ``--out-dir``/``--reports-dir`` it is given.

**Dry-run seams**, honoured only when ``DBP_DRYRUN=1`` (the local runbook dry-run):
``--dryrun-steps N`` replaces the measured budget with N steps, and
``DBP_FAULT_EXIT_AFTER_STEP=K`` kills the process right after the checkpoint at step K
is written, which is how the dry-run proves that a killed run resumes at step K+1.

**Fixed-step mode** (P3 LOLO, ``--fixed-steps N``; pre-registration section 3). The run
trains for exactly N steps (``N_max`` = 6,000), with no calibration steps and no early
stopping, on a cosine schedule over N. Every ``--heldout-every`` steps (500) it records
the **held-out-lineage AUC** -- the validation reals vs their reconstructions by the
members of the lineages this run excludes, each real and each reconstruction scored once
on a fixed crop -- beside the in-protocol validation AUC, log-loss and TPR at 1% FPR, and
it saves a snapshot of the LoRA adapters and the head (``snapshots/step_NNNNN``). Nothing
is selected inside the run: ``N*`` is chosen after all eight folds, by the nested rule.
``--runs-config`` names the run file (``configs/p3_runs.yaml`` for LOLO); a run whose
config says ``gen_matched: false`` (fold 1) trains without the ``gen_matched`` share.

**The P3 recipe for every non-LOLO run** (``--fixed-steps 6000 --stop-at 5000``; user
decision 2026-10-17, the reading of "same recipe" in pre-registration section 3). The
learning rate follows the **6,000-step** cosine schedule and training **stops at
N* = 5000** -- exactly the model N* was selected on (LOLO step 5000 of 6000). In-protocol
validation runs every ``--heldout-every`` steps; the held-out members (every distinct
member the run does not train on) are scored **once, at the final step**, and their
per-image scores are written to ``reports/p3_heldout_scores/<run id>_step_NNNNN.csv``
with a sidecar naming the snapshot scored and its sha256 (D2 (iii)). Each curve point
records the learning rate the step trained with.

**Real-side degradation** (P3 section 4b; a run config's ``degrade_p``). Each pair -- the
real and its reconstruction, and each ``gen_matched`` real/generation row -- gets one
chain from :mod:`dbp.train.degrade`, identical for both members, on the full 384 px
images before the crop and the augmentation. The chains draw from their **own** random
stream (seeded from ``--seed``), so a degraded run sees exactly the batches of the same
seed without degradation and differs from it only by the chains. Every report records
the chain counts and the measured cost per pair and per batch (the data-loader cost the
plan asks the first smoke to measure).

Usage:
    python scripts/p2_train_s2.py --protocol p_a_sd1 --seed 0 \\
        --root /kaggle/input/.../outputs/p1 --allowance-hours 2.25
    python scripts/p2_train_s2.py --runs-config configs/p3_runs.yaml --protocol lolo_f3 \\
        --seed 0 --root ... --fixed-steps 6000 --heldout-every 500
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.backbones.loader import BACKBONES, load_backbone  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.heldout_scores import (  # noqa: E402
    auc_summary,
    heldout_rows,
    score_heldout,
    score_items,
    snapshot_digest,
    stable_seed,
    write_scores,
)
from dbp.eval.metrics import roc_auc, tpr_at_fpr  # noqa: E402
from dbp.eval.panel import DIAGNOSTIC_ONLY_SETS  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.streams.s1_semantic import (  # noqa: E402
    expected_patch_tokens,
    normalise_batch,
    pool_tokens,
    processor_normalisation,
    register_token_count,
)
from dbp.streams.s2_artifact import (  # noqa: E402
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    StepBudget,
    attach_lora,
    build_head,
    pair_batch_plan,
    steps_within_allowance,
)
from dbp.train.augment import PAIR_SIZE, apply, augment_pair, sample_params  # noqa: E402
from dbp.train.degrade import check_codecs, degrade_pair  # noqa: E402
from dbp.utils.hf import get_hf_token  # noqa: E402
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.run_paths import run_id, s2_run_paths  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

DEFAULT_BACKBONE = "dinov3-vitb16"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 300
BATCH_SIZE = 16
GEN_MATCHED_SHARE = 0.25
EVAL_EVERY = 250
PATIENCE_STEPS = 1000

#: A Kaggle session can die at any moment and the allowance is 2.25 GPU-h, so a
#: crash at step 4,000 would otherwise cost the whole run. The checkpoint carries the
#: optimizer, the scaler, the scheduler and the augmentation RNG as well as the
#: weights: resuming from weights alone would restart Adam's moments and the LR
#: schedule, which is a different training run wearing the same step number.
CHECKPOINT_EVERY = 500

#: Second seed word of the degradation stream: ``default_rng([seed, DEGRADE_STREAM])``.
DEGRADE_STREAM = 417


class TrainingData:
    """Pair rows and unpaired fakes for one protocol, split into train and val."""

    def __init__(
        self, root: Path, splits: Path, pairs: Path, gen_matched: Path,
        members: set[str], usable_reals: set[str], log: Any,
    ) -> None:
        side_of = {}
        with splits.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                side_of[row["row_id"]] = row["split"]

        self.pairs: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
        excluded = 0
        diagnostic_only = 0
        with pairs.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["member"] not in members:
                    continue
                # D-039: a reconstruction of a greyscale real is greyscale, and its
                # real partner is not in the corpus. Keeping it would both reinstate
                # the shortcut and, since the real is absent from reals_index, raise a
                # KeyError the first time this row was sampled mid-training.
                if row["image_id"] not in usable_reals:
                    excluded += 1
                    continue
                # B1: T-ctrl is the near-lossless diagnostic. Training S2 on its
                # reconstructions would make the JPEG-reliance check a statement about
                # its own training data. Counted separately from D-039 so the log says
                # which rule removed which rows.
                if row.get("set_id", "") in DIAGNOSTIC_ONLY_SETS:
                    diagnostic_only += 1
                    continue
                side = side_of.get(row["pair_id"])
                if side:
                    self.pairs[side].append(row)
        if excluded:
            log.info("D-039: %d pair row(s) of flagged reals excluded", excluded)
        if diagnostic_only:
            log.info(
                "diagnostic-only (%s): %d pair row(s) withheld from S2 training (B1)",
                ",".join(sorted(DIAGNOSTIC_ONLY_SETS)), diagnostic_only,
            )

        self.gen: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
        with gen_matched.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                side = side_of.get(row["gen_id"])
                if side:
                    self.gen[side].append(row)

        self.root = root
        self.by_real: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in self.pairs["train"]:
            self.by_real[row["image_id"]].append(row)
        self.train_reals = sorted(self.by_real)
        log.info(
            "protocol data: %d train pair rows over %d reals, %d val pair rows, "
            "%d/%d gen_matched",
            len(self.pairs["train"]), len(self.train_reals), len(self.pairs["val"]),
            len(self.gen["train"]), len(self.gen["val"]),
        )
        if not self.train_reals:
            raise ValueError(
                "no training pairs for this protocol; check --members against the "
                "run config's active_members"
            )

    def real_path(self, row: dict[str, str], reals_index: dict[str, str]) -> Path:
        """Path of the real a pair row was reconstructed from."""
        return self.root / reals_index[row["image_id"]]


def log_loss(truth: list[int], scores: list[float]) -> float:
    """Binary cross-entropy of sigmoid(score), clipped, as a plain float."""
    y = np.asarray(truth, dtype=float)
    p = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=float)))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def lr_factor(step: int, schedule_steps: int) -> float:
    """The LR multiplier after ``step`` scheduler steps: linear warm-up, then cosine.

    The cosine runs over ``schedule_steps``. A P3 run stopped early (``--stop-at``) keeps
    ``schedule_steps`` = 6,000, so its learning rate is the 6,000-step schedule's at every
    step it trains. A plain float: the checkpoint is loaded with ``weights_only=True``,
    which refuses numpy scalars.
    """
    return float(
        min(1.0, (step + 1) / WARMUP_STEPS)
        * 0.5
        * (1.0 + np.cos(np.pi * min(step / schedule_steps, 1.0)))
    )


def fixed_budget(steps: int, allowance_hours: float) -> StepBudget:
    """A budget of exactly ``steps``, for smoke tests and the local dry-run only.

    ``measured_step_seconds`` is 0.0 because nothing was measured, and the report says
    so; the step floor does not apply because neither produces a result.
    """
    return StepBudget(
        allowance_hours=float(allowance_hours), measured_step_seconds=0.0,
        steps=int(steps), ceiling=int(steps), undertrained=False,
    )


def dryrun_enabled() -> bool:
    """Whether the dry-run seams may be used (``DBP_DRYRUN=1``)."""
    return os.environ.get("DBP_DRYRUN") == "1"


def fault_step() -> int | None:
    """The step after whose checkpoint a dry-run kills this process, if any."""
    value = os.environ.get("DBP_FAULT_EXIT_AFTER_STEP", "")
    return int(value) if value and dryrun_enabled() else None


def verify_checkpoint(path: Path, model: Any, head: Any, optimiser: Any, step: int) -> list[str]:
    """Reload a checkpoint into fresh objects and compare it with the live state.

    Returns:
        A list of problems; empty when the checkpoint holds exactly the live state.
    """
    import torch

    saved = torch.load(path, map_location="cpu", weights_only=True)
    problems: list[str] = []
    if saved["step"] != step:
        problems.append(f"step {saved['step']} != {step}")
    live = {k: v for k, v in model.state_dict().items() if "lora" in k}
    if set(live) != set(saved["model"]):
        problems.append("LoRA keys differ")
    problems += [f"LoRA {k} differs" for k in live
                 if k in saved["model"] and not torch.equal(live[k].cpu(), saved["model"][k])]
    fresh_head = type(head)()
    fresh_head.load_state_dict(saved["head"])
    problems += [f"head {k} differs" for k, v in head.state_dict().items()
                 if not torch.equal(v.cpu(), fresh_head.state_dict()[k])]
    fresh_optimiser = torch.optim.AdamW(
        [p for group in optimiser.param_groups for p in group["params"]],
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
    )
    fresh_optimiser.load_state_dict(saved["optimiser"])
    for (_, a), (_, b) in zip(sorted(optimiser.state_dict()["state"].items()),
                              sorted(fresh_optimiser.state_dict()["state"].items()),
                              strict=True):
        for key in a:
            if hasattr(a[key], "shape") and not torch.equal(a[key].cpu(), b[key].cpu()):
                problems.append(f"optimiser state {key} differs")
    return problems


def load_reals_index(manifests: Path) -> dict[str, str]:
    """image_id -> relative_path, over both reals manifests, excluding flagged rows."""
    index: dict[str, str] = {}
    for name in ("p1_reals_manifest_t_real.csv", "p1_reals_manifest_t_ctrl.csv"):
        with (manifests / name).open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("excluded_reason"):
                    continue  # D-039: greyscale reals never enter training
                index[row["image_id"]] = row["relative_path"]
    return index


def main() -> int:
    """Train one (protocol, seed). Returns 0 on success, 3 when undertrained."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--manifests", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--splits", type=Path, default=THESIS_DBP / "reports" / "p2_splits.csv")
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allowance-hours", type=float, default=2.25)
    parser.add_argument("--ceiling", type=int, default=8000)
    parser.add_argument("--calibrate", type=int, default=20, help="Steps to time first.")
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Ignore an existing checkpoint and start from step 0. The old checkpoint "
             "is left on disk; this never deletes one.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "s2",
        help="Root of the per-run directories; this run writes only <out-dir>/<run id>/.",
    )
    parser.add_argument("--reports-dir", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--val-pairs", type=int, default=512,
                        help="Validation pairs scored at each evaluation.")
    parser.add_argument("--smoke", type=int, default=0,
                        help="Pre-flight: N steps, save, reload, compare, exit.")
    parser.add_argument("--dryrun-steps", type=int, default=0,
                        help="Local dry-run only (DBP_DRYRUN=1): a fixed budget of N steps.")
    parser.add_argument("--runs-config", type=Path,
                        default=THESIS_DBP / "configs" / "p2_runs.yaml")
    parser.add_argument("--fixed-steps", type=int, default=0,
                        help="P3: train exactly N steps; no calibration, no early stop.")
    parser.add_argument("--heldout-every", type=int, default=500,
                        help="P3 fixed-step mode: held-out-lineage evaluation and snapshot "
                             "every N steps.")
    parser.add_argument("--stop-at", type=int, default=0,
                        help="P3 recipe: stop at this step of the --fixed-steps schedule "
                             "(6000 schedule, stop at N* = 5000); held-out scored once, at "
                             "the end, per image.")
    args = parser.parse_args()
    log = setup_logging("p2_train_s2")
    if args.dryrun_steps and not dryrun_enabled():
        log.error("--dryrun-steps is a local dry-run seam and needs DBP_DRYRUN=1")
        return 2
    fixed = args.fixed_steps > 0
    if fixed and args.dryrun_steps:
        log.error("--fixed-steps and --dryrun-steps are exclusive")
        return 2
    stop = args.stop_at
    if stop and not (fixed and 0 < stop <= args.fixed_steps):
        log.error("--stop-at needs --fixed-steps and 0 < stop-at <= fixed-steps (got %d, %d)",
                  stop, args.fixed_steps)
        return 2
    if args.smoke:
        # A smoke test proves the loop works; it is never a run to continue.
        args.no_resume = True
        args.eval_every = args.checkpoint_every = args.smoke
        args.heldout_every = args.smoke
    elif fixed:
        # One cadence: in-protocol validation, held-out evaluation, snapshot.
        args.eval_every = args.heldout_every

    import torch
    from torch import nn

    seed_everything(args.seed)
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    runs = load_run_configs(args.runs_config, bank)
    run = runs.by_id(args.protocol)
    members = set(run.active_members(bank))
    log.info("%s", run.report_header(bank))
    # Every distinct member the run does not train on. For a LOLO fold that is exactly
    # the held-out lineage's members; BANK-1 also holds out sd21 and sd_ft_ema.
    held_out = sorted(spec.key for spec in bank.distinct_members() if spec.key not in members)
    if fixed and not stop and not held_out:
        log.error("--fixed-steps without --stop-at (LOLO) needs a run that holds out at "
                  "least one lineage")
        return 2

    token = get_hf_token()
    backbone = load_backbone(args.backbone, args.device, token)
    n_register = register_token_count(backbone.config)
    n_patch = expected_patch_tokens(run.eval_crop, backbone.patch_size)
    normalisation = processor_normalisation(backbone.repo, backbone.revision, token)

    lora = attach_lora(backbone.model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model = lora["model"]
    model.gradient_checkpointing_enable()
    head = build_head(backbone.hidden_size).to(args.device)
    log.info(
        "LoRA on %s: %d trainable of %d (%.3f%%)",
        lora["target_modules"], lora["params_trainable"], lora["params_total"],
        lora["trainable_pct"],
    )

    reals_index = load_reals_index(args.manifests)
    data = TrainingData(
        args.root, args.splits, args.pairs_manifest,
        args.manifests / "p1_gen_matched_manifest.csv", members,
        set(reals_index), log,
    )
    plan = pair_batch_plan(BATCH_SIZE, GEN_MATCHED_SHARE if run.use_gen_matched else 0.0)
    log.info("batch plan: %s%s", plan, "" if run.use_gen_matched
             else " (gen_matched held out by the run config)")
    hold_rows = heldout_rows(args.splits, args.pairs_manifest, set(held_out), set(reals_index)) \
        if fixed else []
    hold_reals = sorted({row["image_id"] for row in hold_rows})
    if args.smoke and hold_reals:
        # The smoke proves the held-out path, not the number: four reals and their
        # reconstructions (BANK-1's full set is ~5,900 images, minutes per smoke per GPU).
        hold_reals = hold_reals[:4]
        hold_rows = [row for row in hold_rows if row["image_id"] in set(hold_reals)]
    if fixed and held_out:
        log.info("held-out lineage(s) %s: members %s; %d validation reals, %d reconstructions",
                 list(run.excluded_lineages), held_out, len(hold_reals), len(hold_rows))
        if not hold_rows:
            log.error("no validation reconstructions of the held-out members")
            return 2

    parameters = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimiser = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    # Its own stream: the batches are those of the same seed without degradation.
    degrade_rng = np.random.default_rng([args.seed, DEGRADE_STREAM])
    degrade_stats: dict[str, Any] = {"pairs": 0, "seconds": 0.0, "ops": {},
                                     "batches": 0, "batch_seconds": 0.0}
    if run.degrade_p > 0:
        check_codecs()
        log.info("degradation: chain probability %.2f per pair (P3 section 4b)", run.degrade_p)

    def degrade(members: list[np.ndarray]) -> list[np.ndarray]:
        """One shared chain for the pair; a no-op when the run does not degrade."""
        if run.degrade_p <= 0:
            return members
        started = time.perf_counter()
        out, chain = degrade_pair(members, degrade_rng, run.degrade_p)
        degrade_stats["seconds"] += time.perf_counter() - started
        degrade_stats["pairs"] += 1
        key = (chain.op or "none") + ("_reshare" if chain.quality2 is not None else "")
        degrade_stats["ops"][key] = degrade_stats["ops"].get(key, 0) + 1
        return out

    def make_batch() -> tuple[Any, Any]:
        """One batch of aligned pairs plus the gen_matched share."""
        started = time.perf_counter()
        images: list[np.ndarray] = []
        labels: list[float] = []
        for _ in range(plan["pairs"]):
            real_id = data.train_reals[int(rng.integers(0, len(data.train_reals)))]
            partner = data.by_real[real_id][int(rng.integers(0, len(data.by_real[real_id])))]
            real = load_rgb_uint8(args.root / reals_index[real_id])
            fake = load_rgb_uint8(args.root / partner["relative_path"])
            augmented, _ = augment_pair(degrade([real, fake]), rng, PAIR_SIZE, run.eval_crop)
            images.extend(augmented)
            labels.extend([0.0, 1.0])
        for _ in range(plan["gen_matched_fakes"]):
            gen = data.gen["train"][int(rng.integers(0, len(data.gen["train"])))]
            real_id = data.train_reals[int(rng.integers(0, len(data.train_reals)))]
            params = sample_params(rng, PAIR_SIZE, run.eval_crop)
            real_img, gen_img = degrade([load_rgb_uint8(args.root / reals_index[real_id]),
                                         load_rgb_uint8(args.root / gen["jpg_path"])])
            images.append(apply(real_img, params))
            labels.append(0.0)
            images.append(apply(gen_img, params))
            labels.append(1.0)
        batch = normalise_batch(
            np.stack(images), normalisation["image_mean"], normalisation["image_std"]
        ).to(args.device)
        degrade_stats["batches"] += 1
        degrade_stats["batch_seconds"] += time.perf_counter() - started
        return batch, torch.tensor(labels, dtype=torch.float32, device=args.device)

    def degrade_summary() -> dict[str, Any]:
        """The chain counts and the measured data-loader cost, for every report."""
        pairs, batches = degrade_stats["pairs"], degrade_stats["batches"]
        return {"p": run.degrade_p, "stream": [args.seed, DEGRADE_STREAM],
                "pairs": pairs, "ops": dict(sorted(degrade_stats["ops"].items())),
                "ms_per_pair": round(1000 * degrade_stats["seconds"] / pairs, 3) if pairs else None,
                "batches": batches,
                "batch_build_ms": round(1000 * degrade_stats["batch_seconds"] / batches, 3)
                if batches else None}

    def forward(batch: Any) -> Any:
        hidden = model(pixel_values=batch).last_hidden_state
        return head(pool_tokens(hidden.float(), n_register, n_patch))

    def train_step() -> float:
        model.train()
        head.train()
        batch, labels = make_batch()
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            loss = criterion(forward(batch), labels)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        return float(loss.item())

    # ---- resume, or calibrate and set the budget ----------------------------
    rid = run_id(args.protocol, args.seed)
    paths = s2_run_paths(args.out_dir, args.reports_dir, rid)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = paths.checkpoint
    checkpoint = None
    if checkpoint_path.is_file() and not args.no_resume:
        # weights_only=True: the checkpoint holds tensors, numbers, strings and plain
        # containers only (the RNG state is a dict of ints), so no pickle is executed.
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
        log.info(
            "resuming from %s at step %d (best val AUC %.4f at step %d)",
            checkpoint_path.name, checkpoint["step"], checkpoint["best_auc"],
            checkpoint["best_step"],
        )

    if checkpoint is not None and int(checkpoint.get("stop_at", 0) or 0) != stop:
        log.error("the checkpoint was written with --stop-at %s, this run asks for %s; a "
                  "resumed run must keep its recipe", checkpoint.get("stop_at", 0), stop)
        return 2
    if checkpoint is not None:
        # The budget is restored, not re-measured: a resumed run that re-calibrated
        # could pick up a different ceiling from the one it started under, and its
        # step count would stop meaning what the report says it means.
        recorded = checkpoint["budget"]
        budget = StepBudget(
            allowance_hours=recorded["allowance_hours"],
            measured_step_seconds=recorded["measured_step_seconds"],
            steps=recorded["steps"], ceiling=recorded["ceiling"],
            undertrained=recorded["undertrained"],
        )
        step_seconds = budget.measured_step_seconds
        log.info(
            "budget restored from the checkpoint: %d steps at %.4f s/step",
            budget.steps, step_seconds,
        )
    elif args.smoke:
        budget = fixed_budget(args.smoke, args.allowance_hours)
        log.info("SMOKE: %d step(s), then save, reload and compare", args.smoke)
    elif fixed:
        # No calibration: calibration steps are real, uncounted training steps, and a
        # fixed-step recipe must train exactly N. The step time is measured in the loop.
        budget = StepBudget(allowance_hours=args.allowance_hours, measured_step_seconds=0.0,
                            steps=args.fixed_steps, ceiling=args.fixed_steps,
                            undertrained=False)
        log.info("FIXED-STEP mode: exactly %d steps; held-out evaluation and snapshot every "
                 "%d; no early stopping", budget.steps, args.heldout_every)
    else:
        log.info("calibrating on %d real steps ...", args.calibrate)
        for _ in range(3):
            train_step()  # warm-up, excluded from the timing
        started = time.perf_counter()
        for _ in range(args.calibrate):
            train_step()
        step_seconds = (time.perf_counter() - started) / args.calibrate
        budget = steps_within_allowance(args.allowance_hours, step_seconds, args.ceiling)
        if args.dryrun_steps:
            budget = fixed_budget(args.dryrun_steps, args.allowance_hours)
            log.info("DRY-RUN: fixed budget of %d step(s), floor not applied", budget.steps)
        log.info(
            "measured %.4f s/step at batch %d -> %d steps inside %.2f GPU-h (ceiling %d)",
            step_seconds, BATCH_SIZE, budget.steps, args.allowance_hours, args.ceiling,
        )

    report_path = paths.report

    def write_report(extra: dict[str, Any]) -> None:
        payload = {
            "run_id": rid,
            "protocol": args.protocol,
            "seed": args.seed,
            "backbone": backbone.name,
            "revision": backbone.revision,
            "crop": run.eval_crop,
            "active_members": sorted(members),
            "excluded_lineages": list(run.excluded_lineages),
            "lora": {k: v for k, v in lora.items() if k != "model"},
            "batch": BATCH_SIZE,
            "batch_plan": plan,
            "optimiser": {
                "lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
                "warmup_steps": WARMUP_STEPS, "schedule": "cosine",
            },
            "step_budget": budget.to_dict(),
            "mode": "smoke" if args.smoke else ("dry-run" if args.dryrun_steps else "train"),
            "degrade": degrade_summary(),
            **extra,
        }
        paths.report_tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths.report_tmp.replace(report_path)

    if budget.undertrained:
        write_report({"status": "STOPPED-UNDERTRAINED"})
        log.error(
            "the allowance buys only %d steps, below the approved floor of %d. An "
            "undertrained S2 would not be a weak result, it would make the "
            "stream comparison invalid. Stopping as agreed -- raise "
            "--allowance-hours or reduce the number of runs.",
            budget.steps, budget.to_dict()["min_steps_per_run"],
        )
        return 3

    # The cosine runs over budget.steps -- the full --fixed-steps schedule even when the
    # run stops early (--stop-at): that is the P3 recipe.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: lr_factor(step, budget.steps))
    last = budget.steps if (args.smoke or not stop) else stop

    def validate() -> float:
        truth, scores = validate_scores()
        return roc_auc(truth, scores)

    def score_paths(items: list[tuple[str, Path]]) -> list[float]:
        """Score images on a fixed, seeded crop with no augmentation, in batches."""
        return score_items(items, forward, normalisation, run.eval_crop, args.device,
                           scaler.is_enabled(), BATCH_SIZE)

    def heldout_table() -> list[dict[str, Any]]:
        """Per-image held-out scores: validation reals vs held-out reconstructions."""
        model.eval()
        head.eval()
        return score_heldout(hold_reals, hold_rows, args.root, reals_index, score_paths)

    def write_heldout_scores(step: int, table: list[dict[str, Any]], snapshot: Path) -> Path:
        """D2 (iii): the final step's per-image held-out scores, with the snapshot's sha256."""
        out = args.reports_dir / "p3_heldout_scores" / f"{rid}_step_{step:05d}.csv"
        write_scores(out, table, {
            "run_id": rid, "protocol": args.protocol, "seed": args.seed, "step": step,
            "schedule_steps": budget.steps, "stop_at": stop or None,
            "snapshot": snapshot.relative_to(paths.run_dir).as_posix(),
            "snapshot_sha256": snapshot_digest(snapshot),
            "held_out_members": held_out, "crop": run.eval_crop,
            "backbone": backbone.name, "revision": backbone.revision,
            "scored_by": "p2_train_s2.py, in training (the trainer's own held-out scoring)",
        })
        return out

    def validate_scores() -> tuple[list[int], list[float]]:
        model.eval()
        head.eval()
        scores: list[float] = []
        truth: list[int] = []
        with torch.no_grad():
            for start in range(0, min(len(data.pairs["val"]), args.val_pairs), BATCH_SIZE):
                rows = data.pairs["val"][start : start + BATCH_SIZE]
                crops, labels = [], []
                for row in rows:
                    params = sample_params(
                        np.random.default_rng(stable_seed(row["pair_id"])),
                        PAIR_SIZE, run.eval_crop,
                        settings={k: 0.0 for k in
                                  ("flip_p", "jpeg_p", "blur_p", "noise_p",
                                   "resize_p", "greyscale_p")},
                    )
                    real_path = args.root / reals_index[row["image_id"]]
                    crops.append(apply(load_rgb_uint8(args.root / row["relative_path"]), params))
                    labels.append(1)
                    crops.append(apply(load_rgb_uint8(real_path), params))
                    labels.append(0)
                batch = normalise_batch(
                    np.stack(crops), normalisation["image_mean"], normalisation["image_std"]
                ).to(args.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
                    scores.extend(forward(batch).float().cpu().tolist())
                truth.extend(labels)
        return truth, scores

    # ---- restore, or start clean --------------------------------------------
    best_auc, best_step, curve, first_step = 0.0, 0, [], 1
    hours_before = 0.0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=False)
        head.load_state_dict(checkpoint["head"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        rng.bit_generator.state = checkpoint["rng"]
        if "rng_degrade" in checkpoint:
            degrade_rng.bit_generator.state = checkpoint["rng_degrade"]
            degrade_stats.update(checkpoint.get("degrade_stats", {}))
        best_auc = checkpoint["best_auc"]
        best_step = checkpoint["best_step"]
        curve = list(checkpoint["curve"])
        first_step = checkpoint["step"] + 1
        hours_before = checkpoint["gpu_hours"]
        log.info(
            "continuing at step %d of %d; %d curve point(s) carried over",
            first_step, budget.steps, len(curve),
        )

    def save_checkpoint(step: int, hours: float) -> None:
        """Write the full training state atomically, so a crash mid-write loses nothing."""
        tmp = paths.checkpoint_tmp
        torch.save(
            {
                "step": step,
                "model": {k: v for k, v in model.state_dict().items() if "lora" in k},
                "head": head.state_dict(),
                "optimiser": optimiser.state_dict(),
                "scaler": scaler.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": rng.bit_generator.state,
                "rng_degrade": degrade_rng.bit_generator.state,
                "degrade_stats": dict(degrade_stats),
                "best_auc": best_auc,
                "best_step": best_step,
                "curve": curve,
                "gpu_hours": hours,
                "budget": budget.to_dict(),
                "stop_at": stop,
                "protocol": args.protocol,
                "seed": args.seed,
                "train_seconds": train_seconds,
            },
            tmp,
        )
        tmp.replace(checkpoint_path)

    started = time.perf_counter()
    train_seconds = float(checkpoint.get("train_seconds", 0.0)) if checkpoint is not None \
        else 0.0
    step = first_step - 1
    heldout_scores_file: Path | None = None
    for step in range(first_step, last + 1):
        step_started = time.perf_counter()
        lr_now = float(optimiser.param_groups[0]["lr"])    # the LR this step trains with
        loss = train_step()
        scheduler.step()
        elapsed = hours_before + (time.perf_counter() - started) / 3600.0

        train_seconds += time.perf_counter() - step_started
        if fixed and (step % args.eval_every == 0 or step == last):
            final = step == last
            truth, val_scores = validate_scores()
            point = {"step": step, "loss": round(loss, 5), "lr": lr_now,
                     "val_auc": round(float(roc_auc(truth, val_scores)), 5),
                     "val_logloss": round(log_loss(truth, val_scores), 5),
                     "val_tpr_at_1pct_fpr": round(tpr_at_fpr(truth, val_scores, 0.01)[0], 5)}
            # LOLO: held-out evaluation and a snapshot at every evaluation. A P3-recipe run
            # (--stop-at) selects nothing, so it scores the held-out set and snapshots once,
            # at its final step.
            table = heldout_table() if held_out and (not stop or final) else None
            if table is not None:
                point.update(auc_summary(table))
            curve.append(point)
            if not stop or final:
                snapshot = paths.run_dir / "snapshots" / f"step_{step:05d}"
                model.save_pretrained(str(snapshot / "lora"))
                torch.save(head.state_dict(), snapshot / "head.tmp.pt")
                (snapshot / "head.tmp.pt").replace(snapshot / "head.pt")
                if table is not None and stop:
                    heldout_scores_file = write_heldout_scores(step, table, snapshot)
            # The latest weights are also "the model" for tooling that expects one; no
            # selection happens inside the run.
            model.save_pretrained(str(paths.lora))
            torch.save(head.state_dict(), paths.head_tmp)
            paths.head_tmp.replace(paths.head)
            log.info("step %5d  lr %.3e  loss %.4f  val AUC %.4f  held-out AUC %s %s", step,
                     lr_now, loss, point["val_auc"], point.get("heldout_auc"),
                     point.get("heldout_auc_by_member", ""))
        elif not fixed and (step % args.eval_every == 0 or step == last):
            # A plain float: the checkpoint is loaded with weights_only=True, which refuses
            # numpy scalars (caught by the Session D smoke pre-flight).
            auc = float(validate())
            curve.append({"step": step, "loss": round(loss, 5), "val_auc": round(auc, 5)})
            log.info("step %5d  loss %.4f  val AUC %.4f", step, loss, auc)
            # The first evaluation always saves: a run whose AUC never rose above 0.0
            # (or was NaN on a one-class batch) must still leave a model to score with.
            if auc > best_auc or not paths.head.is_file():
                best_auc, best_step = auc, step
                model.save_pretrained(str(paths.lora))
                torch.save(head.state_dict(), paths.head_tmp)
                paths.head_tmp.replace(paths.head)
            elif step - best_step >= PATIENCE_STEPS:
                log.info("early stop: %d steps without improvement", step - best_step)
                save_checkpoint(step, elapsed)
                break

        if step % args.checkpoint_every == 0:
            save_checkpoint(step, elapsed)
            log.info("  checkpoint at step %d (%.2f GPU-h so far)", step, elapsed)
            if fault_step() == step:
                log.warning("DRY-RUN FAULT: killing the process after step %d", step)
                os._exit(137)

    hours = hours_before + (time.perf_counter() - started) / 3600.0
    save_checkpoint(step, hours)
    if args.smoke and fixed and held_out and \
            not any(p.get("heldout_auc") is not None for p in curve):
        log.error("SMOKE FAILED: the held-out evaluation did not run")
        return 1
    if args.smoke:
        # Always write the selected-model files here, so the inference smoke that follows
        # reads this smoke's weights and never a stale copy from an earlier pre-flight.
        model.save_pretrained(str(paths.lora))
        torch.save(head.state_dict(), paths.head_tmp)
        paths.head_tmp.replace(paths.head)
        problems = verify_checkpoint(checkpoint_path, model, head, optimiser, step)
        if not (paths.lora / "adapter_config.json").is_file() or not paths.head.is_file():
            problems.append("the selected model (LoRA + head) was not saved")
        write_report({"status": "SMOKE-FAILED" if problems else "SMOKE-OK",
                      "steps_run": step, "problems": problems})
        if problems:
            log.error("SMOKE FAILED: %s", "; ".join(problems))
            return 1
        log.info("SMOKE PASSED: %d step(s), checkpoint saved, reloaded and identical", step)
        return 0
    if fixed:
        steps_done = curve[-1]["step"] if curve else step
        write_report(
            {
                "status": "OK" if steps_done == last else "INCOMPLETE",
                "mode_detail": ("P3 recipe: the fixed-steps schedule stopped at --stop-at; "
                                "held-out scored per image at the final step"
                                if stop else "fixed-steps (P3 LOLO); no selection inside the run"),
                "runs_config": args.runs_config.name,
                "fixed_steps": budget.steps,
                "schedule_steps": budget.steps,
                "stop_at": stop or None,
                "excluded_members": list(run.excluded_members),
                "heldout_scores": (heldout_scores_file.relative_to(args.reports_dir).as_posix()
                                   if heldout_scores_file else None),
                "heldout_every": args.heldout_every,
                "held_out_lineages": list(run.excluded_lineages),
                "held_out_members": held_out,
                "heldout_reals": len(hold_reals),
                "heldout_reconstructions": len(hold_rows),
                "gen_matched_used": run.use_gen_matched,
                "steps_run": steps_done,
                "resumed_from_step": first_step if checkpoint is not None else None,
                "curve": curve,
                "snapshots": [p["step"] for p in curve],
                "measured_step_seconds_train_loop": round(train_seconds / max(step, 1), 5),
                "gpu_hours": round(hours, 3),
                "checkpoint_every": args.checkpoint_every,
                "checkpoint": checkpoint_path.name,
                "determinism": "seeds recorded; CUDA nondeterminism means no bit-exactness "
                               "claim",
            }
        )
        log.info("done: %d fixed steps in %.2f GPU-h; held-out AUC at the last step %s",
                 steps_done, hours, curve[-1].get("heldout_auc") if curve else None)
        return 0
    write_report(
        {
            "status": "OK",
            "steps_run": curve[-1]["step"] if curve else step,
            "resumed_from_step": first_step if checkpoint is not None else None,
            "best_val_auc": round(best_auc, 5),
            "best_step": best_step,
            "val_curve": curve,
            "gpu_hours": round(hours, 3),
            "checkpoint_every": args.checkpoint_every,
            "checkpoint": checkpoint_path.name,
            "determinism": "seeds recorded; CUDA nondeterminism means no bit-exactness claim",
        }
    )
    log.info(
        "done: best val AUC %.4f at step %d in %.2f GPU-h (%d curve point(s))",
        best_auc, best_step, hours, len(curve),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
