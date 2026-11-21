"""P5 pilot (EXPLORATORY; docs/P5_PLAN.md section 8, approved 2026-11-21).

Question: does the frozen image ``S2-D`` (P3, ViT-B seeds 0 and 1), applied per frame,
already catch video-VAE reconstructions of 200 Pexels clips by three VAEs?

Fixed before it runs (user, 2026-11-21):
- it may change bank membership **only for a technical failure** (OOM, crash, a wrong
  frame count, non-determinism), never because a VAE is "already caught";
- "already caught" = on the **format-matched** view, clip AUC >= 0.95, TPR >= 80% and
  real FPR <= 15% at S2-D's committed k = 0 union threshold;
- the Qwen-Image vs Wan2.1 VAE weight comparison is reported; if the decoders are
  identical, S2-D's Wan results are flagged "seen decoder".

Phases (each a ``p2_run_shards.py`` worker: ``--members A,B --device cpu|cuda``):
- ``smoke``: every bank and reserve member on 3 clips -- frame count, PSNR, s/clip, peak
  VRAM, re-run determinism, fp16 parity (>= 55 dB at every clip, no NaN/Inf);
- ``lineage`` (CPU): Qwen-Image's VAE vs wan21's, tensor by tensor;
- ``reconstruct``: the pilot VAEs on every kept clip -> lossless PNG clips;
- ``score``: S2-D per frame (frames 1..24) on reals and reconstructions, in two views --
  ``raw`` and ``h264`` (both clips re-encoded identically: libx264 CRF 23, medium,
  yuv420p, native fps);
- ``summary`` (CPU, never fails): the tables, the reading, the archive.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.video import encode_roundtrip, read_png_clip, write_png_clip  # noqa: E402
from dbp.decoders.reconstruct import (  # noqa: E402
    FP16_BUILD_PSNR_THRESHOLD_DB,
    fp16_build_allowed,
    max_abs_diff,
    psnr_uint8,
    reconstruct_per_frame,
    reconstruct_video_kl,
    video_to_tensor,
    video_to_uint8,
)
from dbp.decoders.registry import load_video_bank  # noqa: E402

PILOT_VAES = ("wan21", "hunyuan", "ltx090")
SCORED_FRAMES = range(1, 25)          # frame 0 of a causal VAE is image-like: never scored
VIEWS = {"raw": None, "h264": {"codec": "libx264", "crf": 23, "preset": "medium"}}
CAUGHT = {"clip_auc": 0.95, "tpr": 0.80, "fpr": 0.15, "view": "h264"}
OUT = Path("reports") / "p5_pilot"


def kept_clips(manifest: Path) -> list[dict[str, str]]:
    """The pilot pool's kept clips, in manifest order."""
    with manifest.open(encoding="utf-8", newline="") as handle:
        return [r for r in csv.DictReader(handle) if r["status"] == "kept"]


def write_json(path: Path, payload: Any) -> None:
    """Atomic JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def peak_mb(device: str) -> float | None:
    import torch

    return round(torch.cuda.max_memory_allocated() / 2**20, 1) \
        if device == "cuda" and torch.cuda.is_available() else None


def reset_peak(device: str) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def fp16_clip(member16: Any, clip: np.ndarray) -> tuple[np.ndarray, bool]:
    """A separately loaded fp16 member on one clip, through the protocol functions; and
    whether the float output was finite before quantisation. (The fp32 member is never
    cast: a round trip through fp16 would leave rounded weights behind.)"""
    import torch

    x = video_to_tensor(clip, member16.device, torch.float16)
    y = (reconstruct_per_frame(member16.model, x) if member16.spec.kind == "vae_kl_per_frame"
         else reconstruct_video_kl(member16.model, x))
    return video_to_uint8(y), bool(torch.isfinite(y).all())


def phase_smoke(args: argparse.Namespace, bank: Any, clips: list[dict[str, str]]) -> None:
    from dbp.decoders.loaders import load_video_member

    sample = clips[: args.smoke_clips]
    for key in args.members.split(","):
        spec, out = bank.by_key(key), args.reports / f"smoke_{key}.json"
        record: dict[str, Any] = {"member": key, "role": spec.role, "device": args.device,
                                  "clips": [c["sample_id"] for c in sample]}
        try:
            reset_peak(args.device)
            began = time.perf_counter()
            member = load_video_member(spec, args.device, "fp32")
            record["load_seconds"] = round(time.perf_counter() - began, 2)
            record["dry_run_random_init"] = member.dry_run_random_init
            rows, refs = [], []
            for c in sample:
                clip = read_png_clip(args.reals_root / c["clip_dir"])
                reset_peak(args.device)
                t = time.perf_counter()
                recon = member.reconstruct(clip)
                rows.append({"sample_id": c["sample_id"],
                             "seconds": round(time.perf_counter() - t, 3),
                             "psnr_db": round(psnr_uint8(clip, recon), 3),
                             "peak_mb": peak_mb(args.device), "shape": list(recon.shape)})
                refs.append((clip, recon))
            again = member.reconstruct(refs[0][0])
            record["determinism_max_abs_delta"] = max_abs_diff(refs[0][1], again)
            member.free()
            fp16: dict[int, float | None] = {}
            if args.device == "cuda":
                member16 = load_video_member(spec, args.device, "fp16")
                for i, (clip, ref) in enumerate(refs):
                    half, finite = fp16_clip(member16, clip)
                    fp16[i] = round(psnr_uint8(ref, half), 3) if finite else None
                member16.free()
            allowed, reason = (fp16_build_allowed(fp16) if fp16
                               else (False, "not measured (CPU)"))
            record.update({"status": "OK" if record["determinism_max_abs_delta"] == 0
                           else "FAILED: non-deterministic", "per_clip": rows,
                           "fp16_psnr_db": fp16, "fp16_build_allowed": allowed,
                           "fp16_reason": reason,
                           "fp16_threshold_db": FP16_BUILD_PSNR_THRESHOLD_DB,
                           "warm_seconds_per_clip": round(float(np.mean(
                               [r["seconds"] for r in rows[1:]] or [rows[0]["seconds"]])), 3),
                           "peak_mb_max": max((r["peak_mb"] or 0) for r in rows) or None})
        except Exception as exc:   # recorded per member: a technical failure is a result
            record.update({"status": f"FAILED: {type(exc).__name__}: {str(exc)[:400]}",
                           "peak_mb_at_failure": peak_mb(args.device)})
        write_json(out, record)
        print(f"smoke {key}: {record['status']}", flush=True)


def phase_lineage(args: argparse.Namespace, bank: Any) -> None:
    """Qwen-Image's VAE against wan21's, tensor by tensor (CPU, fp32)."""
    import torch

    from dbp.decoders.loaders import load_member, load_video_member
    from dbp.decoders.registry import load_bank_config

    wan = load_video_member(bank.by_key("wan21"), "cpu", "fp32")
    if os.environ.get("DBP_DRYRUN") == "1":
        other, source = load_video_member(bank.by_key("wan21"), "cpu", "fp32").model, \
            "DRY-RUN: a second wan21 stand-in (the comparison code is tested, not the weights)"
    else:
        image_bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
        spec = next(s for s in image_bank.members if s.key == "qwen_image")
        other, source = load_member(spec, "cpu", "fp32").model, \
            f"{spec.repo}@{spec.revision}/{spec.subfolder}"
    out: dict[str, Any] = {"a": "wan21", "b": source, "parts": {}}
    for part in ("encoder", "decoder", "quant_conv", "post_quant_conv"):
        a, b = getattr(wan.model, part, None), getattr(other, part, None)
        if a is None or b is None:
            out["parts"][part] = {"present": [a is not None, b is not None]}
            continue
        sa, sb = a.state_dict(), b.state_dict()
        common = sorted(set(sa) & set(sb))
        same_shape = [k for k in common if sa[k].shape == sb[k].shape]
        exact = bf16 = fp16 = 0
        worst = 0.0
        for k in same_shape:
            x, y = sa[k].float(), sb[k].float()
            exact += bool(torch.equal(x, y))
            bf16 += bool(torch.equal(x.bfloat16().float(), y))
            fp16 += bool(torch.equal(x.half().float(), y))
            worst = max(worst, float((x - y).abs().max()) if x.numel() else 0.0)
        n = len(sa)
        out["parts"][part] = {
            "tensors_a": n, "tensors_b": len(sb), "names_equal": set(sa) == set(sb),
            "shapes_equal": len(same_shape) == n == len(sb), "identical_fp32": exact,
            "identical_after_bf16_rounding": bf16, "identical_after_fp16_rounding": fp16,
            "max_abs_diff": worst}
    dec = out["parts"].get("decoder", {})
    n = dec.get("tensors_a", -1)
    out["decoder_identical"] = bool(dec.get("shapes_equal")) and max(
        dec.get("identical_fp32", 0), dec.get("identical_after_bf16_rounding", 0),
        dec.get("identical_after_fp16_rounding", 0)) == n
    out["flag"] = ("SEEN DECODER: image S2-D's Wan-family results are flagged 'seen decoder'"
                   if out["decoder_identical"] else "decoders differ")
    write_json(args.reports / "lineage_qwen_vs_wan21.json", out)
    print("lineage:", out["flag"], flush=True)


def phase_reconstruct(args: argparse.Namespace, bank: Any, clips: list[dict[str, str]]) -> None:
    from dbp.decoders.loaders import load_video_member

    for key in args.members.split(","):
        smoke = args.reports / f"smoke_{key}.json"
        status = json.loads(smoke.read_text(encoding="utf-8")).get("status") \
            if smoke.is_file() else "no smoke record"
        record = args.reports / f"recon_{key}.jsonl"
        if status != "OK":
            write_json(args.reports / f"recon_{key}_skipped.json",
                       {"member": key, "reason": f"smoke status: {status}"})
            print(f"reconstruct {key}: SKIPPED ({status})", flush=True)
            continue
        done = set()
        if record.is_file():
            done = {json.loads(line)["sample_id"] for line in
                    record.read_text(encoding="utf-8").splitlines() if line.strip()}
        member = load_video_member(bank.by_key(key), args.device, "fp32")
        root = args.recon_root / key
        for c in clips:
            if c["sample_id"] in done:
                continue
            target = root / c["sample_id"]
            if target.exists():   # written but not recorded: a crash between two writes
                shutil.rmtree(target)
            clip = read_png_clip(args.reals_root / c["clip_dir"])
            t = time.perf_counter()
            recon = member.reconstruct(clip)
            seconds = time.perf_counter() - t
            written = write_png_clip(recon, target)
            row = {"sample_id": c["sample_id"], "seconds": round(seconds, 3),
                   "psnr_db": round(psnr_uint8(clip, recon), 3), **written}
            with record.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        member.free()
        print(f"reconstruct {key}: complete", flush=True)


def load_s2d(model: dict[str, Any], device: str) -> tuple[Any, ...]:
    """The frozen S2-D model on P3's final-scoring path (the same functions, unchanged)."""
    import torch

    from dbp.backbones.loader import load_backbone
    from dbp.data.manifest import file_sha256
    from dbp.streams.s1_semantic import (
        expected_patch_tokens,
        processor_normalisation,
        register_token_count,
    )
    from dbp.utils.hf import get_hf_token
    from dbp.utils.seed import seed_everything
    from p2_s2_infer import load_selected_model

    lora, head = Path(model["lora"]), Path(model["head"])
    now = {"lora_sha256": file_sha256(lora / "adapter_model.safetensors"),
           "head_sha256": file_sha256(head)}
    if now != {k: model[k] for k in now}:
        raise RuntimeError(f"{model['id']}: weights differ from the record: {now}")
    token = get_hf_token()
    seed_everything(int(model["seed"]))
    backbone = load_backbone("dinov3-vitb16", device, token)
    norm = processor_normalisation(backbone.repo, backbone.revision, token)
    net, head_net = load_selected_model(backbone, SimpleNamespace(
        lora=lora, head=head, run_dir=lora.parent), device)
    amp = device.startswith("cuda") and torch.cuda.is_available()
    return net, head_net, norm, register_token_count(backbone.config), \
        expected_patch_tokens(320, backbone.patch_size), amp, backbone


def phase_score(args: argparse.Namespace, expect: dict[str, Any],
                clips: list[dict[str, str]]) -> None:
    import torch

    from dbp.streams.s1_semantic import normalise_batch, pool_tokens

    sources = ["real", *[k for k in PILOT_VAES if (args.reports / f"recon_{k}.jsonl").is_file()]]
    clips = clips[: args.limit] if args.limit else clips
    out_dir = args.reports / ("score_smoke" if args.limit else "scores")
    for model_id in args.members.split(","):
        model = next(m for m in expect["models"] if m["id"] == model_id)
        model = {**model, "lora": str(args.s2d_root / model["lora"]),
                 "head": str(args.s2d_root / model["head"])}
        net, head, norm, n_reg, n_patch, amp, backbone = load_s2d(model, args.device)
        target = out_dir / f"{model_id}.csv"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        began, frames_scored = time.perf_counter(), 0
        with tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["sample_id", "source", "view", "frame", "logit"])
            for c in clips:
                for source in sources:
                    root = args.reals_root / c["clip_dir"] if source == "real" else \
                        args.recon_root / source / c["sample_id"]
                    if not root.is_dir():
                        continue
                    clip = read_png_clip(root)
                    for view, setting in VIEWS.items():
                        frames = clip if setting is None else encode_roundtrip(
                            clip, setting["codec"], setting["crf"], c["fps_fraction"] or 30,
                            setting["preset"])[0]
                        picked = np.stack([frames[i] for i in SCORED_FRAMES])
                        batch = normalise_batch(picked, norm["image_mean"],
                                                norm["image_std"]).to(args.device)
                        with torch.no_grad():
                            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                                hidden = net(pixel_values=batch).last_hidden_state
                            logits = head(pool_tokens(hidden.float(), n_reg, n_patch))
                        values = logits.float().cpu().numpy().ravel()
                        if not np.all(np.isfinite(values)):
                            raise RuntimeError(f"{model_id}: non-finite logits")
                        for frame, value in zip(SCORED_FRAMES, values, strict=True):
                            writer.writerow([c["sample_id"], source, view, frame,
                                             f"{float(value):.6f}"])
                        frames_scored += len(values)
        tmp.replace(target)
        write_json(out_dir / f"{model_id}.json", {
            "model": model_id, "lora_sha256": model["lora_sha256"],
            "head_sha256": model["head_sha256"], "backbone": backbone.name,
            "revision": backbone.revision, "normalisation": norm.get("source", "processor"),
            "dry_run_random_init": bool(backbone.config.get("dry_run_random_init")),
            "sources": sources, "views": VIEWS, "frames": list(SCORED_FRAMES),
            "clips": len(clips), "frames_scored": frames_scored,
            "seconds": round(time.perf_counter() - began, 1)})
        print(f"score {model_id}: {frames_scored} frames", flush=True)


def summarise(args: argparse.Namespace, expect: dict[str, Any]) -> dict[str, Any]:
    from dbp.eval.metrics import roc_auc

    thresholds = json.loads((THESIS_DBP / expect["thresholds"]["path"]).read_text("utf-8"))
    rng = np.random.default_rng(2026)
    summary: dict[str, Any] = {"exploratory": True, "rule_already_caught": CAUGHT,
                               "smoke": {}, "reconstruct": {}, "scores": {}}
    for path in sorted(args.reports.glob("smoke_*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        summary["smoke"][s["member"]] = {k: s.get(k) for k in (
            "status", "role", "load_seconds", "warm_seconds_per_clip", "peak_mb_max",
            "determinism_max_abs_delta", "fp16_build_allowed", "fp16_reason")} | {
            "psnr_db": [r["psnr_db"] for r in s.get("per_clip", [])]}
    for path in sorted(args.reports.glob("recon_*.jsonl")):
        rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line]
        summary["reconstruct"][path.stem[6:]] = {
            "clips": len(rows),
            "psnr_db_mean": round(float(np.mean([r["psnr_db"] for r in rows])), 3),
            "seconds_per_clip_mean": round(float(np.mean([r["seconds"] for r in rows])), 3),
            "png_bytes_per_clip_mean": int(np.mean([r["bytes"] for r in rows]))}
    lineage = args.reports / "lineage_qwen_vs_wan21.json"
    summary["lineage"] = json.loads(lineage.read_text("utf-8")) if lineage.is_file() else None
    seen = bool(summary["lineage"] and summary["lineage"].get("decoder_identical"))
    for model in expect["models"]:
        path = args.reports / "scores" / f"{model['id']}.csv"
        if not path.is_file():
            continue
        thr = float(thresholds["descriptive"][model["id"]]["threshold_union"])
        table: dict[tuple[str, str, str], list[float]] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for r in csv.DictReader(handle):
                table.setdefault((r["source"], r["view"], r["sample_id"]), []).append(
                    float(r["logit"]))
        result: dict[str, Any] = {"threshold_union": thr}
        for view in VIEWS:
            reals = {k[2]: v for k, v in table.items() if k[0] == "real" and k[1] == view}
            if not reals:
                continue
            real_clip = np.array([np.mean(v) for v in reals.values()])
            real_frames = np.concatenate([np.array(v) for v in reals.values()])
            per_view: dict[str, Any] = {"real_fpr_clip": float(np.mean(real_clip >= thr)),
                                        "real_fpr_frame": float(np.mean(real_frames >= thr)),
                                        "n_real_clips": len(real_clip)}
            for vae in PILOT_VAES:
                fakes = {k[2]: v for k, v in table.items() if k[0] == vae and k[1] == view}
                if not fakes:
                    continue
                ids = sorted(set(fakes) & set(reals))
                fc = np.array([np.mean(fakes[i]) for i in ids])
                rc = np.array([np.mean(reals[i]) for i in ids])
                y = np.r_[np.zeros(len(rc)), np.ones(len(fc))]
                auc = roc_auc(y, np.r_[rc, fc])
                boots = []
                for _ in range(1000):   # paired: resample clip ids
                    pick = rng.integers(0, len(ids), len(ids))
                    boots.append(roc_auc(y, np.r_[rc[pick], fc[pick]]))
                ff = np.concatenate([np.array(fakes[i]) for i in ids])
                rf = np.concatenate([np.array(reals[i]) for i in ids])
                entry = {"n_pairs": len(ids), "clip_auc": round(auc, 4),
                         "clip_auc_ci95": [round(float(np.quantile(boots, q)), 4)
                                           for q in (0.025, 0.975)],
                         "frame_auc": round(roc_auc(np.r_[np.zeros(len(rf)), np.ones(len(ff))],
                                                    np.r_[rf, ff]), 4),
                         "tpr_clip": float(np.mean(fc >= thr)),
                         "fpr_clip_paired_reals": float(np.mean(rc >= thr))}
                if view == CAUGHT["view"]:
                    entry["already_caught"] = bool(
                        entry["clip_auc"] >= CAUGHT["clip_auc"] and entry["tpr_clip"] >=
                        CAUGHT["tpr"] and per_view["real_fpr_clip"] <= CAUGHT["fpr"])
                if vae == "wan21" and seen:
                    entry["flag"] = "seen decoder (Qwen-Image VAE == wan21 VAE)"
                per_view[vae] = entry
            result[view] = per_view
        summary["scores"][model["id"]] = result
    return summary


def phase_summary(args: argparse.Namespace, expect: dict[str, Any]) -> int:
    """Never fails the run: any error is written into the summary and printed."""
    try:
        summary = summarise(args, expect)
    except Exception as exc:
        summary = {"error": f"{type(exc).__name__}: {exc}"}
    write_json(args.reports / "p5_pilot_summary.json", summary)
    print(json.dumps(summary, indent=1, default=str)[:6000])
    if args.zip is not None:
        try:
            args.zip.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(f"{args.zip}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
                for folder in (args.reports, args.reports.parent / "p5_reals"):
                    for path in sorted(folder.rglob("*")):
                        if path.is_file() and path.suffix in (".json", ".jsonl", ".csv", ".txt",
                                                              ".log", ".md"):
                            archive.write(path, path.relative_to(args.reports.parent.parent))
            print(f"archive: {args.zip}.zip")
        except OSError as exc:
            print(f"archive NOT written: {exc}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--phase", required=True,
                        choices=["smoke", "lineage", "reconstruct", "score", "summary"])
    parser.add_argument("--members", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--reals-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--recon-root", type=Path, default=None)
    parser.add_argument("--s2d-root", type=Path, default=None)
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / OUT)
    parser.add_argument("--smoke-clips", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="score: a smoke on N clips")
    parser.add_argument("--zip", type=Path, default=None)
    args = parser.parse_args()
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    if args.phase == "summary":
        return phase_summary(args, expect)
    bank = load_video_bank(THESIS_DBP / "configs" / "video_bank.yaml")
    if args.phase == "lineage":
        phase_lineage(args, bank)
        return 0
    clips = kept_clips(args.manifest)
    if not clips:
        raise SystemExit(f"{args.manifest}: no kept clips")
    if args.phase == "smoke":
        phase_smoke(args, bank, clips)
    elif args.phase == "reconstruct":
        phase_reconstruct(args, bank, clips)
    else:
        phase_score(args, expect, clips)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
