"""P0 decoder-bank smoke test.

Proves that every bank member loads and reconstructs correctly, deterministically and
within the T4's limits, before any dataset is built. Bad numerics here (fp16 overflow,
tiling seams, an accidental resize) would silently corrupt every later experiment.

For each member and crop size it records, per section 6.4.9:
``key, revision, dtype, crop, PSNR, LPIPS-vgg, mean|r|, ms_per_img, cpu_ms_per_img,
peak_gpu_mb, fp16_allowed, deterministic, status``.

``ms_per_img`` is reserved for GPU timings. On CPU it stays empty and the measurement
goes in ``cpu_ms_per_img``, so a CPU number can never be mistaken for a T4 number.

Timings discard ``--warmups`` reconstructions first (issue I-009). Without that, the
first member measured absorbs CUDA context creation, cuDNN autotuning and the initial
allocator growth: in the P0 T4 run ``sd15`` reported 2029 ms at crop 512 against a peer
median of 761 ms, which is a property of the schedule, not of the decoder.

``--compare-to`` re-reads an earlier CSV and asserts the deterministic columns
(``psnr_db``, ``lpips_vgg``, ``mean_abs_residual``) are unchanged. That is the
regression check against the real pinned weights, complementing the offline digests in
``tests/test_reconstruct_regression.py``. Timing and memory columns are excluded: they
are measurements of the machine, not of the protocol.

Usage:
    python scripts/p0_bank_smoke_test.py --device cpu --crops 512
    python scripts/p0_bank_smoke_test.py --device cuda --crops 512,1024
    # Regression against the P0 T4 run:
    python scripts/p0_bank_smoke_test.py --device cuda --crops 512,1024
        --compare-to reports/kaggle_20260920/p0_bank_smoke.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

THESIS_DBP = Path(__file__).resolve().parents[1]
REPO_ROOT = THESIS_DBP.parent
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.decoders.loaders import BankMember, MemberLoadError, load_member  # noqa: E402
from dbp.decoders.reconstruct import (  # noqa: E402
    fp16_verdict,
    max_abs_diff,
    prepare_input,
    psnr_uint8,
    verify_diffusers_api,
)
from dbp.decoders.registry import DecoderSpec, load_bank_config  # noqa: E402
from dbp.utils.io import center_crop, residual_int16, save_png_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402

#: Reconstructions discarded before the timed one (issue I-009).
TIMING_WARMUPS = 3

#: Columns a re-run must reproduce exactly. Deliberately excludes every timing and
#: memory column, which depend on the machine rather than on the protocol.
REGRESSION_COLUMNS = ("psnr_db", "lpips_vgg", "mean_abs_residual", "max_abs_delta_rerun")

CSV_COLUMNS = [
    "key",
    "lineage",
    "tier",
    "kind",
    "revision",
    "dtype",
    "crop",
    "psnr_db",
    "lpips_vgg",
    "mean_abs_residual",
    "ms_per_img",
    "cpu_ms_per_img",
    "peak_gpu_mb",
    "fp16_allowed",
    "fp16_psnr_db",
    "deterministic",
    "max_abs_delta_rerun",
    "status",
    "detail",
]

#: Fixed amplification for the saved residual visualisation, so members are comparable.
RESIDUAL_AMPLIFICATION = 10


def load_smoke_images(crops: list[int], log: Any) -> dict[int, tuple[str, np.ndarray]]:
    """Return one public-domain skimage test image per requested crop size.

    Only ``skimage.data`` samples are used (guardrail #8). A crop size with no source
    image large enough is skipped rather than satisfied by resizing.
    """
    from skimage import data

    candidates: list[tuple[str, Any]] = [
        ("astronaut", data.astronaut),
        ("retina", data.retina),
        ("coffee", data.coffee),
        ("chelsea", data.chelsea),
    ]
    loaded: list[tuple[str, np.ndarray]] = []
    for name, fn in candidates:
        try:
            arr = np.asarray(fn(), dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 3:
                loaded.append((name, arr))
                log.info("smoke image %-10s %s", name, arr.shape)
        except Exception as exc:
            log.warning("smoke image %-10s unavailable: %s", name, type(exc).__name__)

    chosen: dict[int, tuple[str, np.ndarray]] = {}
    for crop in crops:
        fitting = [(n, a) for n, a in loaded if a.shape[0] >= crop and a.shape[1] >= crop]
        if not fitting:
            log.warning("crop %d: no skimage sample is large enough; skipping (never resize)", crop)
            continue
        name, arr = fitting[0]
        chosen[crop] = (name, center_crop(arr, crop))
        log.info("crop %4d -> %s centre-cropped to %dx%d", crop, name, crop, crop)
    return chosen


def build_lpips(device: torch.device, log: Any) -> Any | None:
    """Instantiate LPIPS-VGG, or return None with an explanation logged."""
    try:
        import lpips

        model = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()
        log.info("LPIPS-vgg ready on %s", device)
        return model
    except Exception as exc:
        log.warning(
            "LPIPS unavailable (%s: %s). lpips_vgg will be recorded as NOT-RUN.",
            type(exc).__name__,
            str(exc)[:160],
        )
        return None


@torch.no_grad()
def lpips_distance(model: Any, a: np.ndarray, b: np.ndarray, device: torch.device) -> float:
    """LPIPS between two uint8 images. LPIPS expects NCHW in [-1, 1]."""
    def prep(arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
        return (t.to(device=device, dtype=torch.float32) / 127.5) - 1.0

    return float(model(prep(a), prep(b)).item())


def timed_reconstruct(
    member: BankMember, img: np.ndarray, is_cuda: bool, warmups: int = 0
) -> tuple[np.ndarray, float]:
    """Reconstruct once and return the output with the wall time in milliseconds.

    Args:
        member: The loaded bank member.
        img: ``uint8`` HWC input.
        is_cuda: Whether to synchronise around the measurement.
        warmups: Reconstructions to run and discard first. The reconstruction is
            deterministic, so a warm-up cannot change the result -- only the clock.
    """
    for _ in range(warmups):
        member.reconstruct(img)
    if is_cuda:
        torch.cuda.synchronize()
    start = time.perf_counter()
    out = member.reconstruct(img)
    if is_cuda:
        torch.cuda.synchronize()
    return out, (time.perf_counter() - start) * 1000.0


def save_triplet(
    key: str, crop: int, x: np.ndarray, recon: np.ndarray, out_dir: Path
) -> None:
    """Save the input, the reconstruction and an amplified |residual| as PNGs."""
    residual = residual_int16(x, recon)
    amplified = np.clip(np.abs(residual) * RESIDUAL_AMPLIFICATION, 0, 255).astype(np.uint8)
    save_png_uint8(x, out_dir / f"{key}_{crop}_input.png")
    save_png_uint8(recon, out_dir / f"{key}_{crop}_recon.png")
    save_png_uint8(amplified, out_dir / f"{key}_{crop}_residual_x{RESIDUAL_AMPLIFICATION}.png")


def run_member(
    spec: DecoderSpec,
    images: dict[int, tuple[str, np.ndarray]],
    device: torch.device,
    cache_dir: Path,
    out_dir: Path,
    lpips_model: Any | None,
    log: Any,
    decoder_hashes: dict[str, str],
    warmups: int = TIMING_WARMUPS,
) -> list[dict[str, Any]]:
    """Run every crop size for one member and return its CSV rows."""
    is_cuda = device.type == "cuda"
    rows: list[dict[str, Any]] = []

    base = {
        "key": spec.key,
        "lineage": spec.lineage,
        "tier": spec.tier,
        "kind": spec.kind,
        "revision": (spec.revision or "")[:12],
    }

    try:
        member = load_member(spec, device=device, precision="fp32", cache_dir=cache_dir)
    except MemberLoadError as exc:
        message = str(exc)
        status = "BLOCKED-ACCESS" if _is_access_error(message) else "ERROR"
        log.error("%-14s %s: %s", spec.key, status, message[:200])
        return [{**base, "dtype": "fp32", "crop": "", "status": status, "detail": message[:300]}]

    digest = member.decoder_state_dict_sha256()
    if digest:
        decoder_hashes[spec.key] = digest

    for crop, (_source, img) in sorted(images.items()):
        row: dict[str, Any] = {**base, "dtype": "fp32", "crop": crop}
        try:
            if is_cuda:
                torch.cuda.reset_peak_memory_stats(device)

            recon_a, ms_a = timed_reconstruct(member, img, is_cuda, warmups=warmups)
            recon_b, _ = timed_reconstruct(member, img, is_cuda)

            x = prepare_input(img, spec.crop_multiple)
            delta = max_abs_diff(recon_a, recon_b)
            residual = residual_int16(x, recon_a)

            row["psnr_db"] = round(psnr_uint8(x, recon_a), 3)
            row["mean_abs_residual"] = round(float(np.mean(np.abs(residual))), 4)
            row["max_abs_delta_rerun"] = delta
            row["deterministic"] = bool(delta <= 1)
            if is_cuda:
                row["ms_per_img"] = round(ms_a, 1)
                row["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
            else:
                row["cpu_ms_per_img"] = round(ms_a, 1)
                row["ms_per_img"] = ""
                row["peak_gpu_mb"] = "PENDING-GPU"

            if lpips_model is not None:
                row["lpips_vgg"] = round(lpips_distance(lpips_model, x, recon_a, device), 5)
            else:
                row["lpips_vgg"] = "NOT-RUN"

            save_triplet(spec.key, crop, x, recon_a, out_dir)

            # fp16 parity. Only meaningful on CUDA: CPU fp16 uses different kernels and
            # would not reproduce the overflow behaviour the threshold is testing for.
            if is_cuda:
                row.update(_fp16_parity(spec, img, device, cache_dir, recon_a, log))
            else:
                row["fp16_allowed"] = "PENDING-GPU"
                row["fp16_psnr_db"] = "PENDING-GPU"

            row["status"] = "OK"
            log.info(
                "%-14s crop=%-5d PSNR=%7.3f dB  mean|r|=%.4f  det=%s  %s",
                spec.key,
                crop,
                row["psnr_db"],
                row["mean_abs_residual"],
                row["deterministic"],
                f"{ms_a:.0f} ms ({'gpu' if is_cuda else 'cpu'})",
            )
        except torch.cuda.OutOfMemoryError:
            # Protocol step 11: record and continue. Never fall back to tiling.
            row["status"] = f"OOM@{crop}"
            row["detail"] = "CUDA OOM; tiling is forbidden so no fallback was attempted"
            log.error("%-14s OOM at crop %d (no tiling fallback by design)", spec.key, crop)
            torch.cuda.empty_cache()
        except Exception as exc:
            row["status"] = "ERROR"
            row["detail"] = f"{type(exc).__name__}: {exc}"[:300]
            log.error("%-14s crop=%d ERROR %s", spec.key, crop, row["detail"])
            log.debug(traceback.format_exc())
        rows.append(row)

    member.free()
    del member
    if is_cuda:
        torch.cuda.empty_cache()
    return rows


def _fp16_parity(
    spec: DecoderSpec,
    img: np.ndarray,
    device: torch.device,
    cache_dir: Path,
    recon_fp32: np.ndarray,
    log: Any,
) -> dict[str, Any]:
    """Load the member in fp16, reconstruct, and judge parity against fp32."""
    try:
        member16 = load_member(spec, device=device, precision="fp16", cache_dir=cache_dir)
        recon16 = member16.reconstruct(img)
        member16.free()
        del member16
        torch.cuda.empty_cache()
    except Exception as exc:
        log.warning("%-14s fp16 run failed: %s", spec.key, f"{type(exc).__name__}: {exc}"[:160])
        recon16 = None
    verdict = fp16_verdict(recon_fp32, recon16)
    if not verdict["fp16_allowed"]:
        log.warning("%-14s fp16 NOT allowed: %s", spec.key, verdict["fp16_reason"])
    return {
        "fp16_allowed": verdict["fp16_allowed"],
        "fp16_psnr_db": verdict["fp16_psnr_db"],
    }


def _is_access_error(message: str) -> bool:
    """Whether a load failure was caused by gated-repo denial rather than a bug."""
    lowered = message.lower()
    return any(
        token in lowered
        for token in ("gatedrepo", "401", "403", "awaiting a review", "access to model")
    )


def find_identical_decoders(hashes: dict[str, str]) -> list[list[str]]:
    """Group member keys whose decoder weights are byte-identical."""
    groups: dict[str, list[str]] = {}
    for key, digest in hashes.items():
        groups.setdefault(digest, []).append(key)
    return [sorted(keys) for keys in groups.values() if len(keys) > 1]


def compare_to_baseline(
    rows: list[dict[str, Any]], baseline_csv: Path
) -> tuple[list[str], int]:
    """Check the deterministic columns against an earlier run of this script.

    Only ``(key, crop)`` pairs present in both files are compared, so a run over a
    subset of the bank -- or a baseline predating a new member -- is not an error. The
    comparison is on the CSV text, not on floats re-parsed and re-rounded: the file is
    the artefact the thesis cites, and two runs that write different text are different
    runs regardless of how close the numbers are.

    Args:
        rows: The rows this run produced.
        baseline_csv: An earlier ``p0_bank_smoke.csv``.

    Returns:
        ``(mismatch_messages, pairs_compared)``. An empty list means no regression.

    Raises:
        FileNotFoundError: If the baseline does not exist. A silently skipped
            regression check is worse than no regression check.
    """
    if not baseline_csv.is_file():
        raise FileNotFoundError(f"--compare-to baseline not found: {baseline_csv}")

    with baseline_csv.open("r", encoding="utf-8", newline="") as handle:
        baseline = {(r["key"], r["crop"]): r for r in csv.DictReader(handle)}

    mismatches: list[str] = []
    compared = 0
    for row in rows:
        identity = (str(row.get("key", "")), str(row.get("crop", "")))
        old = baseline.get(identity)
        if old is None:
            continue
        compared += 1
        for column in REGRESSION_COLUMNS:
            was = old.get(column, "")
            now = "" if row.get(column) is None else str(row.get(column, ""))
            if was != now:
                mismatches.append(
                    f"{identity[0]} crop={identity[1]} {column}: was {was!r}, now {now!r}"
                )
    return mismatches, compared


def main() -> int:
    parser = argparse.ArgumentParser(description="P0 decoder-bank smoke test.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--crops", default="512", help="Comma-separated crop sizes, e.g. 512,1024")
    parser.add_argument("--members", default="", help="Comma-separated keys; default is all.")
    parser.add_argument("--tiers", default="core,optional")
    parser.add_argument(
        "--no-lpips", action="store_true", help="Skip LPIPS (avoids a 553 MB VGG16 download)."
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=TIMING_WARMUPS,
        help="Reconstructions discarded before the timed one (issue I-009).",
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help=(
            "An earlier p0_bank_smoke.csv. Every deterministic column must match for "
            "every (key, crop) present in both; any difference is a regression and "
            "exits non-zero."
        ),
    )
    args = parser.parse_args()
    if args.warmups < 0:
        parser.error("--warmups cannot be negative")

    log = setup_logging("p0_bank_smoke", log_dir=THESIS_DBP / "outputs" / "logs")
    seed_info = seed_everything(2026, deterministic=True)
    log.info("seed: %s", seed_info)

    api = verify_diffusers_api()
    log.info("diffusers API check: %s", api)
    if not all(api.values()):
        log.error("diffusers output attribute names have changed: %s", api)
        log.error("Refusing to run: the reconstruction protocol cannot be trusted.")
        return 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log.error("--device cuda requested but torch.cuda.is_available() is False.")
        return 2
    log.info("device: %s", device)

    config = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    specs = config.select(tiers=tiers)
    if args.members:
        wanted = {m.strip() for m in args.members.split(",") if m.strip()}
        specs = [s for s in specs if s.key in wanted]
    log.info("members: %s", [s.key for s in specs])

    crops = [int(c) for c in args.crops.split(",") if c.strip()]
    images = load_smoke_images(crops, log)
    if not images:
        log.error("no usable smoke image for crops %s", crops)
        return 2

    cache_dir = THESIS_DBP / ".cache" / "weights"
    out_dir = THESIS_DBP / "outputs" / "p0"
    out_dir.mkdir(parents=True, exist_ok=True)

    lpips_model = None if args.no_lpips else build_lpips(device, log)

    rows: list[dict[str, Any]] = []
    decoder_hashes: dict[str, str] = {}
    for spec in specs:
        rows.extend(
            run_member(
                spec,
                images,
                device,
                cache_dir,
                out_dir,
                lpips_model,
                log,
                decoder_hashes,
                warmups=args.warmups,
            )
        )

    csv_path = THESIS_DBP / "reports" / "p0_bank_smoke.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLUMNS})
    log.info("wrote %s (%d rows)", csv_path, len(rows))

    identical = find_identical_decoders(decoder_hashes)
    findings = {
        "device": str(device),
        "crops": crops,
        "smoke_images": {str(c): name for c, (name, _) in images.items()},
        "diffusers_api_check": api,
        "decoder_state_dict_sha256": decoder_hashes,
        "byte_identical_decoder_groups": identical,
        "note": (
            "Members listed in the same group share byte-identical decoder weights and "
            "are therefore ONE lineage, not two, for the leave-one-lineage-out protocol."
        ),
    }
    findings_path = THESIS_DBP / "reports" / "p0_decoder_identity.json"
    findings_path.write_text(json.dumps(findings, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s", findings_path)
    if identical:
        for group in identical:
            log.warning("byte-identical decoders: %s", group)
    else:
        log.info("no byte-identical decoder pairs among the members that loaded")

    ok = sum(1 for r in rows if r.get("status") == "OK")
    log.info("summary: %d/%d rows OK", ok, len(rows))
    for row in rows:
        if row.get("status") != "OK":
            log.warning("  %-14s %-8s %s", row["key"], row.get("status"), row.get("detail", ""))

    if args.compare_to is not None:
        mismatches, compared = compare_to_baseline(rows, args.compare_to)
        log.info("regression check against %s: %d (key, crop) pairs", args.compare_to, compared)
        if compared == 0:
            log.error("regression check compared nothing; the baseline shares no rows with "
                      "this run, so it proves nothing.")
            return 1
        if mismatches:
            log.error("REGRESSION: %d deterministic value(s) changed", len(mismatches))
            for message in mismatches:
                log.error("  %s", message)
            return 1
        log.info("regression check PASSED: every deterministic column is unchanged")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
