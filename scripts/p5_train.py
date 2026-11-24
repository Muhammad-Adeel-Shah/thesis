"""P5 training (docs/P5_PREREGISTRATION.md section 3): frame models, feature caches, temporal heads.

Three stages, each one resumable job under the AWS session wrapper:

- ``--stage frame`` -- ``S2V-F`` / ``S2V-1`` / ``LOLO-v``: **the S2 recipe per frame**, reusing
  P3's building blocks unchanged (DINOv3 ViT-B/16 + LoRA r = 8, the head, token pooling, AMP,
  batch 16 = 8 aligned pairs, AdamW LR 1e-4 wd 0.01, warm-up 300, the 6,000-step cosine
  schedule stopped at ``N*`` = 5,000, augmentation, and the S2-D degradation chain at
  p = 0.35 on its own stream). A step draws a real, one of its pairs, and ONE frame index
  t in 1..24 for both clips (frame 0 of a causal VAE is never used; PROPOSED in the prereg).
  The clips are the stored pair files (``p5_build_pairs.py``): the degraded versions where the
  pair was degraded, else the real's PNG frames and the format-matched reconstruction.
  Checkpoint every 500 steps; the final model is saved as P3 saves it (``lora/`` +
  ``head.pt``, weights only), so P3's scoring loader reads it unchanged.
- ``--stage cache`` -- the frozen frame model's pooled features and frame logits for frames
  0..16 of every training pair clip and every validation clip (all bank members), as
  safetensors with a JSON index.
- ``--stage temporal`` -- ``S2V-T``: per-frame features ``f_t`` and deltas ``f_t - f_{t-1}``,
  t = 1..16, into a 2-layer transformer head of width 256, trained on the cached features
  (AdamW LR 3e-4, wd 0.01, batch 64 clips, 3,000 steps, seeded; PROPOSED in the prereg).
  Then the validation report: in-protocol AUC and the held-out-lineage AUC (V1-a), for the
  frame model (mean logit, frames 1..16) and the temporal head.

Runs: ``s2v_f`` (all 6 members), ``s2v_1`` (wan21 only), ``lolo_<v>`` for the F1 folds
(v in wan21, hunyuan, ltx090; sd15_pf always in). Nothing is selected on panel data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

BANK = ("wan21", "ltx090", "sd15_pf", "hunyuan", "cogvx", "mochi")
ALWAYS_IN = "sd15_pf"
LOLO_FOLDS = ("wan21", "hunyuan", "ltx090")   # F1 fired 2026-11-23; HunyuanVideo kept 2026-11-24
FRAMES_SCORED = range(1, 17)                  # frames 1..16 of a 17-frame window
TRAIN_FRAMES = (1, 24)                        # t drawn in 1..24 of a 25-frame clip
DEGRADE_P, DEGRADE_STREAM = 0.35, 417
SCHEDULE_STEPS, STOP_AT, CHECKPOINT_EVERY = 6000, 5000, 500
TEMPORAL = {"width": 256, "layers": 2, "heads": 4, "lr": 3e-4, "wd": 0.01, "batch": 64,
            "steps": 3000}


def run_members(run: str) -> set[str]:
    """The bank members a run trains on."""
    if run == "s2v_f":
        return set(BANK)
    if run == "s2v_1":
        return {"wan21"}
    if run.startswith("lolo_") and run[5:] in LOLO_FOLDS:
        return set(BANK) - {run[5:]}
    raise SystemExit(f"unknown run {run!r}: s2v_f, s2v_1 or lolo_<{'|'.join(LOLO_FOLDS)}>")


def load_pairs(path: Path) -> list[dict[str, Any]]:
    """The pair records, the LAST line per pair id (a rebuilt pair supersedes its old line)."""
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            latest[rec["pair_id"]] = rec
    return [latest[k] for k in sorted(latest)]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClipStore:
    """Frames of the stored clips: PNG frames of a real, or a decoded mp4 (exact)."""

    CACHE_CLIPS = 96                       # about 0.7 GB of decoded 25 x 320 x 320 clips

    def __init__(self, reals_root: Path, pairs_root: Path) -> None:
        self.reals_root, self.pairs_root = reals_root, pairs_root
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _decoded(self, relative: str) -> np.ndarray:
        from dbp.data.video import decode_clip

        if relative in self._cache:
            self._cache.move_to_end(relative)
            return self._cache[relative]
        frames = decode_clip((self.pairs_root / relative).read_bytes())
        self._cache[relative] = frames
        if len(self._cache) > self.CACHE_CLIPS:
            self._cache.popitem(last=False)
        return frames

    def frames(self, rec: dict[str, Any], side: str, use_degraded: bool,
               diagnostic: bool = False) -> np.ndarray:
        """``side`` "real" or "fake"; the degraded files only when the pair has them; the
        validation diagnostic's re-encoded copy with ``diagnostic``."""
        from dbp.data.video import read_png_clip

        if diagnostic:
            return self._decoded(rec["real_diag" if side == "real" else "recon_diag"]["path"])
        if use_degraded and rec.get("degraded"):
            return self._decoded(rec["real_deg" if side == "real" else "recon_deg"]["path"])
        if side == "real":
            return read_png_clip(self.reals_root / rec["real_clip"])
        return self._decoded(rec["fm"]["path"])


def backbone_parts(device: str) -> dict[str, Any]:
    from dbp.backbones.loader import load_backbone
    from dbp.streams.s1_semantic import (
        expected_patch_tokens,
        processor_normalisation,
        register_token_count,
    )
    from dbp.utils.hf import get_hf_token

    token = get_hf_token()
    backbone = load_backbone("dinov3-vitb16", device, token)
    return {"backbone": backbone, "n_register": register_token_count(backbone.config),
            "n_patch": expected_patch_tokens(320, backbone.patch_size),
            "norm": processor_normalisation(backbone.repo, backbone.revision, token)}


def stage_frame(args: argparse.Namespace) -> int:
    import torch
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from torch import nn

    from dbp.streams.s1_semantic import normalise_batch, pool_tokens
    from dbp.streams.s2_artifact import LORA_ALPHA, LORA_DROPOUT, LORA_R, attach_lora, build_head
    from dbp.train.augment import augment_pair
    from dbp.train.degrade import check_codecs, degrade_pair
    from dbp.utils.seed import seed_everything
    from p2_train_s2 import BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, lr_factor

    members = run_members(args.run)
    pairs = [r for r in load_pairs(args.pairs) if r["split"] == "train" and r["member"] in members]
    if not pairs:
        raise SystemExit(f"{args.run}: no training pairs for {sorted(members)}")
    by_real: dict[str, list[dict[str, Any]]] = {}
    for rec in pairs:
        by_real.setdefault(rec["sample_id"], []).append(rec)
    reals = sorted(by_real)
    run_dir = args.out_dir / f"{args.run}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    parts = backbone_parts(args.device)
    lora = attach_lora(parts["backbone"].model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model = lora["model"]
    model.gradient_checkpointing_enable()
    head = build_head(parts["backbone"].hidden_size).to(args.device)
    params = [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
    optimiser = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: lr_factor(step, args.schedule_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=args.device.startswith("cuda"))
    criterion = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(args.seed)
    degrade_rng = np.random.default_rng([args.seed, DEGRADE_STREAM])
    check_codecs()
    store = ClipStore(args.reals_root, args.pairs_root)
    checkpoint_path = run_dir / "checkpoint.pt"
    step, curve = 0, []
    if checkpoint_path.is_file():
        ck = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        set_peft_model_state_dict(model, ck["lora"])
        head.load_state_dict(ck["head"])
        optimiser.load_state_dict(ck["optimiser"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        rng.bit_generator.state, degrade_rng.bit_generator.state = ck["rng"], ck["degrade_rng"]
        step, curve = int(ck["step"]), list(ck["curve"])
        print(f"resumed {args.run} seed {args.seed} at step {step}", flush=True)

    def batch() -> tuple[Any, Any]:
        images, labels = [], []
        for _ in range(BATCH_SIZE // 2):
            real_id = reals[int(rng.integers(0, len(reals)))]
            rec = by_real[real_id][int(rng.integers(0, len(by_real[real_id])))]
            t = int(rng.integers(TRAIN_FRAMES[0], TRAIN_FRAMES[1] + 1))
            pair = [store.frames(rec, "real", True)[t], store.frames(rec, "fake", True)[t]]
            pair, _ = degrade_pair(pair, degrade_rng, DEGRADE_P)
            augmented, _ = augment_pair(pair, rng, 320, 320)
            images.extend(augmented)
            labels.extend([0.0, 1.0])
        x = normalise_batch(np.stack(images), parts["norm"]["image_mean"],
                            parts["norm"]["image_std"]).to(args.device)
        return x, torch.tensor(labels, dtype=torch.float32, device=args.device)

    started = time.time()
    while step < args.stop_at:
        model.train()
        head.train()
        x, y = batch()
        optimiser.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=scaler.is_enabled()):
            hidden = model(pixel_values=x).last_hidden_state
            loss = criterion(head(pool_tokens(hidden.float(), parts["n_register"],
                                              parts["n_patch"])), y)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()
        step += 1
        if step % 100 == 0 or step == args.stop_at:
            curve.append({"step": step, "loss": round(float(loss.item()), 5),
                          "lr": scheduler.get_last_lr()[0]})
            print(f"{args.run} seed {args.seed} step {step} loss {float(loss.item()):.4f} "
                  f"({time.time() - started:.0f} s)", flush=True)
        if step % args.checkpoint_every == 0 or step == args.stop_at:
            tmp = checkpoint_path.with_name("checkpoint.tmp")
            torch.save({"lora": get_peft_model_state_dict(model), "head": head.state_dict(),
                        "optimiser": optimiser.state_dict(), "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(), "rng": rng.bit_generator.state,
                        "degrade_rng": degrade_rng.bit_generator.state, "step": step,
                        "curve": curve}, tmp)
            tmp.replace(checkpoint_path)
    model.save_pretrained(str(run_dir / "lora"))
    tmp = run_dir / "head.tmp.pt"
    torch.save(head.state_dict(), tmp)
    tmp.replace(run_dir / "head.pt")
    report = {"run": args.run, "seed": args.seed, "members": sorted(members), "steps": step,
              "schedule_steps": args.schedule_steps, "pairs": len(pairs), "reals": len(reals),
              "lora_sha256": sha256_file(run_dir / "lora" / "adapter_model.safetensors"),
              "head_sha256": sha256_file(run_dir / "head.pt"), "curve": curve,
              "seconds": round(time.time() - started, 1)}
    write_json(args.reports / f"p5_frame_{args.run}_seed{args.seed}.json", report)
    print(json.dumps({k: report[k] for k in ("run", "seed", "steps", "lora_sha256",
                                             "head_sha256")}), flush=True)
    return 0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def cache_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Every clip the temporal stage and the validation report need, once each."""
    members = run_members(args.run)
    items, seen_val_reals = [], set()
    for rec in load_pairs(args.pairs):
        if rec["split"] == "train" and rec["member"] in members:
            items.append({"key": f"{rec['pair_id']}:real", "rec": rec, "side": "real",
                          "degraded": True, "label": 0, "split": "train", "member": rec["member"]})
            items.append({"key": f"{rec['pair_id']}:fake", "rec": rec, "side": "fake",
                          "degraded": True, "label": 1, "split": "train", "member": rec["member"]})
        elif rec["split"] == "validation":          # every bank member: V1-a needs them all
            if rec["sample_id"] not in seen_val_reals:
                seen_val_reals.add(rec["sample_id"])
                items.append({"key": f"val:{rec['sample_id']}:real", "rec": rec, "side": "real",
                              "degraded": False, "label": 0, "split": "validation",
                              "member": ""})
            items.append({"key": f"val:{rec['pair_id']}:fake", "rec": rec, "side": "fake",
                          "degraded": False, "label": 1, "split": "validation",
                          "member": rec["member"]})
            # the diagnostic: the same pair re-encoded (its own real copy per pair)
            for side, label in (("real", 0), ("fake", 1)):
                items.append({"key": f"valdiag:{rec['pair_id']}:{side}", "rec": rec,
                              "side": side, "degraded": False, "diagnostic": True,
                              "label": label, "split": "validation_reencoded",
                              "member": rec["member"] if label else ""})
    return items


def stage_cache(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import save_file

    from dbp.streams.s1_semantic import normalise_batch, pool_tokens
    from p2_s2_infer import load_selected_model

    run_dir = args.out_dir / f"{args.run}_seed{args.seed}"
    if (run_dir / "features.safetensors").is_file() and (run_dir / "features_index.json").is_file():
        print(f"cache {args.run} seed {args.seed}: present, skipped (resume)", flush=True)
        return 0
    parts = backbone_parts(args.device)
    model, head = load_selected_model(parts["backbone"], SimpleNamespace(
        lora=run_dir / "lora", head=run_dir / "head.pt", run_dir=run_dir), args.device)
    items = cache_items(args)
    store = ClipStore(args.reals_root, args.pairs_root)
    feats, logits = [], []
    amp = args.device.startswith("cuda") and torch.cuda.is_available()
    started = time.time()
    with torch.no_grad():
        for i, item in enumerate(items):
            frames = store.frames(item["rec"], item["side"], item["degraded"],
                                  item.get("diagnostic", False))[:17]
            x = normalise_batch(frames, parts["norm"]["image_mean"],
                                parts["norm"]["image_std"]).to(args.device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                hidden = model(pixel_values=x).last_hidden_state
            pooled = pool_tokens(hidden.float(), parts["n_register"], parts["n_patch"])
            feats.append(pooled.cpu().to(torch.float16))
            logits.append(head(pooled).float().view(-1).cpu())
            if (i + 1) % 1000 == 0:
                print(f"  cached {i + 1}/{len(items)} clips, {time.time() - started:.0f} s",
                      flush=True)
    out = run_dir / "features.safetensors"
    tmp = out.with_name("features.tmp.safetensors")
    save_file({"features": torch.stack(feats), "logits": torch.stack(logits)}, str(tmp))
    tmp.replace(out)
    index = [{k: item[k] for k in ("key", "label", "split", "member")} for item in items]
    write_json(run_dir / "features_index.json", index)
    print(f"cache {args.run} seed {args.seed}: {len(items)} clips x 17 frames, "
          f"{time.time() - started:.0f} s", flush=True)
    return 0


def temporal_head(dim: int) -> Any:
    """``[B, 16, 2*dim]`` (f_t, f_t - f_{t-1}) -> one logit per clip."""
    import torch
    from torch import nn

    class Head(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = TEMPORAL["width"]
            self.proj = nn.Linear(2 * dim, width)
            self.pos = nn.Parameter(torch.zeros(1, len(FRAMES_SCORED), width))
            layer = nn.TransformerEncoderLayer(width, TEMPORAL["heads"], 2 * width, 0.1,
                                               batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, TEMPORAL["layers"])
            self.out = nn.Linear(width, 1)

        def forward(self, x: Any) -> Any:
            return self.out(self.encoder(self.proj(x) + self.pos).mean(dim=1)).view(-1)

    return Head()


def temporal_inputs(features: Any) -> Any:
    """``[N, 17, D]`` per-frame features -> ``[N, 16, 2D]`` (f_t, f_t - f_{t-1}), t = 1..16."""
    import torch

    f = features.float()
    return torch.cat([f[:, 1:17], f[:, 1:17] - f[:, 0:16]], dim=2)


def stage_temporal(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import load_file, save_file

    from dbp.eval.metrics import roc_auc

    run_dir = args.out_dir / f"{args.run}_seed{args.seed}"
    cached = load_file(str(run_dir / "features.safetensors"))
    index = json.loads((run_dir / "features_index.json").read_text(encoding="utf-8"))
    x_all = temporal_inputs(cached["features"])
    labels = torch.tensor([float(i["label"]) for i in index])
    train = [i for i, row in enumerate(index) if row["split"] == "train"]
    train_real = [i for i in train if index[i]["label"] == 0]
    train_fake = [i for i in train if index[i]["label"] == 1]
    torch.manual_seed(args.seed)
    rng = np.random.default_rng([args.seed, 1])
    head = temporal_head(cached["features"].shape[2])
    optimiser = torch.optim.AdamW(head.parameters(), lr=TEMPORAL["lr"],
                                  weight_decay=TEMPORAL["wd"])
    criterion = torch.nn.BCEWithLogitsLoss()
    steps = args.temporal_steps
    for step in range(1, steps + 1):
        half = TEMPORAL["batch"] // 2
        pick = (list(rng.choice(train_real, half)) + list(rng.choice(train_fake, half)))
        head.train()
        loss = criterion(head(x_all[pick]), labels[pick])
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if step % 500 == 0 or step == steps:
            print(f"temporal {args.run} seed {args.seed} step {step} loss {loss.item():.4f}",
                  flush=True)
    out = run_dir / "temporal_head.safetensors"
    tmp = out.with_name("temporal_head.tmp.safetensors")
    save_file({k: v.contiguous() for k, v in head.state_dict().items()}, str(tmp))
    tmp.replace(out)
    head.eval()
    with torch.no_grad():
        temporal = head(x_all).tolist()
    frame = cached["logits"][:, 1:17].mean(dim=1).tolist()
    members = run_members(args.run)
    def auc(scores: list[float], fake_members: set[str],
            split: str = "validation") -> float | None:
        reals = [i for i, r in enumerate(index) if r["split"] == split and r["label"] == 0]
        fakes = [i for i, r in enumerate(index) if r["split"] == split
                 and r["label"] == 1 and r["member"] in fake_members]
        if not fakes or not reals:
            return None
        ids = reals + fakes
        return round(roc_auc([index[i]["label"] for i in ids], [scores[i] for i in ids]), 4)

    val_real = [i for i, r in enumerate(index) if r["split"] == "validation" and r["label"] == 0]

    held_out = set(BANK) - members
    report = {"run": args.run, "seed": args.seed, "members": sorted(members),
              "held_out": sorted(held_out), "temporal_steps": steps,
              "temporal_head_sha256": sha256_file(out), "validation_reals": len(val_real),
              "in_protocol_auc": {"frame": auc(frame, members), "temporal": auc(temporal, members)},
              "held_out_auc": {"frame": auc(frame, held_out), "temporal": auc(temporal, held_out)},
              "per_member_auc": {m: {"frame": auc(frame, {m}), "temporal": auc(temporal, {m})}
                                 for m in BANK},
              # the diagnostic (user, 2026-11-24): the same validation pairs re-encoded
              "reencoded_diagnostic": {
                  "in_protocol_auc": {"frame": auc(frame, members, "validation_reencoded"),
                                      "temporal": auc(temporal, members, "validation_reencoded")},
                  "held_out_auc": {"frame": auc(frame, held_out, "validation_reencoded"),
                                   "temporal": auc(temporal, held_out, "validation_reencoded")}}}
    write_json(args.reports / f"p5_temporal_{args.run}_seed{args.seed}.json", report)
    print(json.dumps({k: report[k] for k in ("run", "seed", "in_protocol_auc", "held_out_auc")}),
          flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--stage", required=True, choices=["frame", "cache", "temporal"])
    parser.add_argument("--run", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pairs", type=Path, required=True, help="p5_pairs.jsonl")
    parser.add_argument("--pairs-root", type=Path, required=True)
    parser.add_argument("--reals-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports" / "p5_train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--schedule-steps", type=int, default=SCHEDULE_STEPS)
    parser.add_argument("--stop-at", type=int, default=STOP_AT)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--temporal-steps", type=int, default=TEMPORAL["steps"])
    args = parser.parse_args()
    run_members(args.run)
    from dbp.utils.seed import seed_everything

    seed_everything(args.seed)
    stages = {"frame": stage_frame, "cache": stage_cache, "temporal": stage_temporal}
    return stages[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main())
