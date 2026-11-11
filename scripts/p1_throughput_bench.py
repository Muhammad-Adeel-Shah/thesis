"""P1-B1: measure what the T4 actually does, before committing to a 12-hour build.

Everything the P1 budget rests on was extrapolated from P0, which measured one image
per crop, at batch 1, in fp32 only. Three of those extrapolations are guesses:

1. **fp16 speed.** P0 ran fp16 only to check parity, never to time it.
2. **Batching.** P0 never ran a batch larger than 1.
3. **Generation.** P0 never ran a UNet at all, so the cost of B6 is unknown, not
   merely uncertain. ``--time-generation N`` measures it on N images before anything
   is committed to.

It also decides fp16 eligibility for the build, which is a stricter question than the
P0 smoke test asked: 55 dB at **every** crop measured, not 45 dB at one
(``FP16_BUILD_PSNR_THRESHOLD_DB``). A pair built in fp16 is permanent, and a member
that is fine at 512 and marginal at 1024 must not be trusted at either.

Writes ``reports/p1_throughput.csv`` and ``reports/p1_backbone_throughput.json``.
No dataset is written here, but the pin check still runs first: a bench against a
member that no longer resolves would report a blank row as a measurement.

Usage:
    python scripts/p1_throughput_bench.py --device cuda --crops 384 --batches 1,2,4
    python scripts/p1_throughput_bench.py --device cuda --time-generation 10
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.decoders.loaders import MemberLoadError, load_member  # noqa: E402
from dbp.decoders.reconstruct import (  # noqa: E402
    FP16_BUILD_PSNR_THRESHOLD_DB,
    fp16_build_allowed,
    psnr_uint8,
    verify_diffusers_api,
)
from dbp.decoders.registry import DecoderSpec, load_bank_config  # noqa: E402
from dbp.utils.hf import get_hf_token  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402
from p0_verify_pins import PinVerificationError, require_pins  # noqa: E402

#: Reconstructions discarded before timing (issue I-009).
WARMUPS = 3

#: Timed repeats; the median is reported so one scheduler hiccup cannot set the budget.
REPEATS = 5

CSV_COLUMNS = [
    "key",
    "lineage",
    "kind",
    "crop",
    "precision",
    "batch",
    "ms_per_img",
    "cpu_ms_per_img",
    "images_per_s",
    "peak_gpu_mb",
    "fp16_parity_psnr_db",
    "fp16_parity_source",
    "fp16_build_allowed",
    "fp16_build_reason",
    "status",
    "detail",
]


@dataclass(frozen=True)
class Timing:
    """One timed measurement, in milliseconds per image."""

    ms_per_image: float
    peak_gpu_mb: float | None

    @property
    def images_per_second(self) -> float:
        return 1000.0 / self.ms_per_image if self.ms_per_image > 0 else float("inf")


def synthetic_image(size: int, seed: int = 2026) -> np.ndarray:
    """A deterministic, photo-like test image.

    Structured rather than white noise: a VAE's cost does not depend on the content,
    but its *residual* does, and a flat or noise image would make the parity PSNR
    meaningless.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float64)
    base = (
        118.0
        + 52.0 * np.sin(xs / 13.0)
        + 41.0 * np.cos(ys / 19.0)
        + 26.0 * np.sin((xs + ys) / 29.0)
    )
    stack = np.stack(
        [base, base * 0.88 + 17.0, base * 0.72 + 39.0 + rng.normal(0, 2.5, base.shape)], axis=-1
    )
    return np.clip(stack, 0, 255).astype(np.uint8)


def time_member(
    member: Any, image: np.ndarray, batch: int, is_cuda: bool, device: torch.device
) -> Timing:
    """Time ``batch`` reconstructions and return the per-image cost.

    The batch is run as ``batch`` successive single-image reconstructions rather than
    as one stacked tensor. ``BankMember.reconstruct`` takes one HWC image, and giving
    it a batch dimension would be a second reconstruction path -- exactly what the
    protocol forbids. What this measures is therefore the throughput the build scripts
    will actually see, which is the number the budget needs.
    """
    for _ in range(WARMUPS):
        member.reconstruct(image)

    if is_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    samples: list[float] = []
    for _ in range(REPEATS):
        if is_cuda:
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(batch):
            member.reconstruct(image)
        if is_cuda:
            torch.cuda.synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0 / batch)

    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if is_cuda else None
    return Timing(ms_per_image=statistics.median(samples), peak_gpu_mb=peak)


def bench_member(
    spec: DecoderSpec,
    crops: list[int],
    batches: list[int],
    device: torch.device,
    cache_dir: Path,
    log: Any,
) -> list[dict[str, Any]]:
    """Benchmark one member across crops, precisions and batch sizes."""
    is_cuda = device.type == "cuda"
    base = {"key": spec.key, "lineage": spec.lineage, "kind": spec.kind}
    rows: list[dict[str, Any]] = []

    # fp32 first: it is the reference for the fp16 parity judgement.
    parity_by_crop: dict[int, float | None] = {}
    reference: dict[int, np.ndarray] = {}

    for precision in ("fp32", "fp16"):
        if precision == "fp16" and not is_cuda:
            log.info("%-14s fp16 skipped on CPU (different kernels, different numerics)", spec.key)
            continue
        try:
            member = load_member(spec, device=device, precision=precision, cache_dir=cache_dir)
        except MemberLoadError as exc:
            message = str(exc)[:300]
            log.error("%-14s %s load failed: %s", spec.key, precision, message)
            rows.append({**base, "precision": precision, "status": "ERROR", "detail": message})
            if precision == "fp32":
                return rows
            continue

        for crop in crops:
            image = synthetic_image(crop)
            try:
                recon = member.reconstruct(image)
                if precision == "fp32":
                    reference[crop] = recon
                else:
                    parity_by_crop[crop] = (
                        psnr_uint8(reference[crop], recon) if crop in reference else None
                    )

                for batch in batches:
                    timing = time_member(member, image, batch, is_cuda, device)
                    row = {
                        **base,
                        "crop": crop,
                        "precision": precision,
                        "batch": batch,
                        "images_per_s": round(timing.images_per_second, 3),
                        "peak_gpu_mb": (
                            round(timing.peak_gpu_mb, 1)
                            if timing.peak_gpu_mb is not None
                            else "PENDING-GPU"
                        ),
                        "status": "OK",
                    }
                    if is_cuda:
                        row["ms_per_img"] = round(timing.ms_per_image, 2)
                        row["cpu_ms_per_img"] = ""
                    else:
                        row["ms_per_img"] = ""
                        row["cpu_ms_per_img"] = round(timing.ms_per_image, 2)
                    rows.append(row)
                    log.info(
                        "%-14s crop=%-5d %-4s batch=%-2d  %8.2f ms/img  %6.2f img/s",
                        spec.key,
                        crop,
                        precision,
                        batch,
                        timing.ms_per_image,
                        timing.images_per_second,
                    )
            except torch.cuda.OutOfMemoryError:
                rows.append(
                    {
                        **base,
                        "crop": crop,
                        "precision": precision,
                        "status": f"OOM@{crop}",
                        "detail": "CUDA OOM; tiling is forbidden so no fallback was attempted",
                    }
                )
                log.error("%-14s OOM at crop %d (no tiling fallback by design)", spec.key, crop)
                torch.cuda.empty_cache()
            except Exception as exc:
                rows.append(
                    {
                        **base,
                        "crop": crop,
                        "precision": precision,
                        "status": "ERROR",
                        "detail": f"{type(exc).__name__}: {exc}"[:300],
                    }
                )
                log.error("%-14s crop=%d ERROR %s", spec.key, crop, exc)
                log.debug(traceback.format_exc())

        member.free()
        del member
        if is_cuda:
            torch.cuda.empty_cache()

    if parity_by_crop:
        allowed, reason = fp16_build_allowed(parity_by_crop)
        for row in rows:
            row["fp16_build_allowed"] = allowed
            row["fp16_build_reason"] = reason
            # Provenance, because the number differs from P0's by design and the
            # difference caused a review query. P0 measures parity on skimage
            # photographs at crops 512/1024; this measures it on the synthetic image
            # at the build crop. Neither is wrong, but a CSV that does not say which
            # it is invites exactly that confusion (I-040).
            row["fp16_parity_source"] = (
                f"synthetic_image(size={row.get('crop')}, seed=2026), "
                f"threshold {FP16_BUILD_PSNR_THRESHOLD_DB:.0f} dB at every crop"
            )
            crop = row.get("crop")
            if crop in parity_by_crop and parity_by_crop[crop] is not None:
                row["fp16_parity_psnr_db"] = round(float(parity_by_crop[crop]), 3)
        log.info(
            "%-14s fp16 for the BUILD: %s (%s; threshold %.0f dB at every crop)",
            spec.key,
            "ALLOWED" if allowed else "REFUSED",
            reason,
            FP16_BUILD_PSNR_THRESHOLD_DB,
        )
    return rows


def bench_backbones(device: torch.device, size: int, log: Any) -> dict[str, Any]:
    """Measure both backbones at the P2 input size, fp16 autocast on CUDA."""
    from dbp.backbones.loader import BACKBONES, BackboneLoadError, forward_once, load_backbone

    results: dict[str, Any] = {}
    for name in BACKBONES:
        try:
            backbone = load_backbone(name, device=device, token=get_hf_token())
        except BackboneLoadError as exc:
            results[name] = {"status": "BLOCKED-ACCESS", "detail": str(exc)[:300]}
            log.error("%-14s backbone unavailable: %s", name, str(exc)[:200])
            continue
        measured = forward_once(
            backbone.model,
            size=size,
            device=device,
            use_fp16_autocast=device.type == "cuda",
            warmups=WARMUPS,
        )
        measured["status"] = "OK"
        results[name] = measured
        log.info(
            "%-14s %d tokens, %.2f ms/forward, peak %s MB",
            name,
            measured["num_tokens"],
            measured["forward_ms"],
            measured["peak_gpu_mb"],
        )
        del backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return results


def time_generation(count: int, steps: int, device: torch.device, log: Any) -> dict[str, Any]:
    """Time SD1.5 text-to-image, the unmeasured half of the P1 budget.

    The pipeline is loaded from the same repo and the same pinned revision as the
    ``sd15`` bank member, so the generator and the probe are provably the same
    checkpoint and no second pin is introduced.

    The safety checker is left **enabled**. It costs time, and that time belongs in the
    budget, because B6 runs with it enabled.
    """
    from diffusers import StableDiffusionPipeline

    spec = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml").by_key("sd15")
    log.info("loading %s @ %s for generation timing", spec.repo, spec.revision[:12])
    pipeline = StableDiffusionPipeline.from_pretrained(
        spec.repo,
        revision=spec.revision,
        use_safetensors=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        token=get_hf_token(),
    )
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    if pipeline.safety_checker is None:
        log.warning("this checkpoint ships no safety checker; B6 must state that")

    generator = torch.Generator(device=device).manual_seed(2026)
    prompt = "a photograph of a city street in the late afternoon"

    # One discarded generation: the first absorbs compilation and allocator growth.
    pipeline(prompt, num_inference_steps=steps, generator=generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    samples: list[float] = []
    for index in range(count):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        pipeline(prompt, num_inference_steps=steps, generator=generator)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = (time.perf_counter() - start) * 1000.0
        samples.append(elapsed)
        log.info("generation %d/%d: %.0f ms", index + 1, count, elapsed)

    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    median = statistics.median(samples)
    del pipeline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "repo": spec.repo,
        "revision": spec.revision,
        "n_timed": count,
        "steps": steps,
        "median_ms_per_image": round(median, 1),
        "min_ms_per_image": round(min(samples), 1),
        "max_ms_per_image": round(max(samples), 1),
        "peak_gpu_mb": round(peak, 1) if peak is not None else None,
        "gpu_hours_per_1000": round((median / 1000.0) * 1000 / 3600.0, 3),
        "safety_checker": "enabled",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-B1 throughput and budget bench.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crops", default="384", help="Comma-separated crop sizes.")
    parser.add_argument("--batches", default="1,2,4", help="Comma-separated batch sizes.")
    parser.add_argument("--members", default="", help="Comma-separated keys; default is all.")
    parser.add_argument("--backbone-size", type=int, default=336)
    parser.add_argument("--no-backbones", action="store_true")
    parser.add_argument(
        "--time-generation",
        type=int,
        default=0,
        help="Time this many SD1.5 generations to size B6. 0 skips it.",
    )
    parser.add_argument("--generation-steps", type=int, default=25)
    parser.add_argument("--skip-pin-check", action="store_true", help="Diagnostics only.")
    args = parser.parse_args()

    log = setup_logging("p1_throughput", log_dir=THESIS_DBP / "outputs" / "logs")
    log.info("seed: %s", seed_everything(2026, deterministic=True))

    api = verify_diffusers_api()
    if not all(api.values()):
        log.error("diffusers output attributes changed: %s", api)
        return 2
    log.info("diffusers API check: all %d attributes present", len(api))

    if not args.skip_pin_check:
        try:
            require_pins(printer=log.info)
        except PinVerificationError as exc:
            log.error("%s", exc)
            return 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log.error("--device cuda requested but no CUDA device is present.")
        return 2
    log.info("device: %s", device)

    config = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    specs = config.distinct_members()
    if args.members:
        wanted = {m.strip() for m in args.members.split(",") if m.strip()}
        specs = [s for s in specs if s.key in wanted]
    log.info("benchmarking %d distinct members: %s", len(specs), [s.key for s in specs])
    log.info("aliases skipped: %s", config.aliases() or "none")

    crops = [int(c) for c in args.crops.split(",") if c.strip()]
    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    cache_dir = THESIS_DBP / ".cache" / "weights"

    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.extend(bench_member(spec, crops, batches, device, cache_dir, log))

    csv_path = THESIS_DBP / "reports" / "p1_throughput.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    findings: dict[str, Any] = {
        "device": str(device),
        "crops": crops,
        "batches": batches,
        "warmups_discarded": WARMUPS,
        "timed_repeats": REPEATS,
        "statistic": "median",
        "backbones": None,
        "generation": None,
    }
    if not args.no_backbones:
        findings["backbones"] = bench_backbones(device, args.backbone_size, log)
    if args.time_generation > 0:
        findings["generation"] = time_generation(
            args.time_generation, args.generation_steps, device, log
        )

    json_path = THESIS_DBP / "reports" / "p1_backbone_throughput.json"
    json_path.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", json_path)

    ok = sum(1 for r in rows if r.get("status") == "OK")
    log.info("summary: %d/%d rows OK", ok, len(rows))
    failed = [r for r in rows if r.get("status") not in ("OK", None)]
    for row in failed:
        log.warning("  %-14s %s %s", row["key"], row.get("status"), row.get("detail", ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
