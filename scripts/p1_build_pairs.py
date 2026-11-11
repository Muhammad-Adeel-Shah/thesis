"""P1-B5: reconstruct every real image through every distinct bank member.

This is the dataset the method is built on: for each real image, one reconstruction per
decoder lineage, so a detector can be trained on the *pattern across decoders* rather
than on any single decoder's artefacts.

Four properties the implementation guarantees:

**Aliases are skipped by the registry, never by name.** The member list comes from
``BankConfig.distinct_members()``. ``sd_ft_mse`` has byte-identical decoder weights to
``sd21`` (D-016), so reconstructing both would store the same image twice under two
names and double that lineage's weight in training. No key is written out by hand
anywhere in this file.

**It is resumable, because a 12-hour session is a hard deadline.** Work is member-major
and chunked: a member is loaded once, its images are processed in chunks, and each
chunk's manifest rows are appended only after its PNGs are on disk. A rerun reads the
manifest, skips finished images and continues. A killed run can leave a PNG with no
row; the rerun simply redoes it, which is safe because the reconstruction is
deterministic.

**fp16 is used only where it was measured to be safe.** The precision per member comes
from ``reports/p1_throughput.csv`` and the rule in ``fp16_build_allowed``: 55 dB parity
at every crop measured. Without that file the run is fp32 throughout, which is slower
and always correct.

**Sharding splits members, not images.** Two GPUs each load about half the bank once,
rather than both loading all of it. ``--shard i --num-shards n`` assigns members
round-robin, so each shard gets a mix of cheap and expensive members.

**P3 step 4a reuses it unchanged in substance** (the 4,000 Open Images reals through the
same 12 members, at the same per-member precision, read from the committed P1 bench
``reports/kaggle_20260926/p1_throughput.csv``). ``--name`` keeps its manifests and summary
apart from P1's (``p3_oi_pairs_manifest_<key>.csv``); ``--cache-dir`` puts downloaded
weights outside the output directory; ``--limit`` builds the pre-flight sample; each
member's summary records the bytes it wrote, so the output size is measured.

Usage:
    python scripts/p1_build_pairs.py --reals-manifest reports/p1_reals_manifest_t_real.csv
    python scripts/p1_build_pairs.py --shard 0 --num-shards 2 --device cuda:0
    python scripts/p1_build_pairs.py --merge
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.manifest import (  # noqa: E402
    append_rows,
    completed_ids,
    file_sha256,
    read_manifest,
    write_manifest,
)
from dbp.decoders.loaders import MemberLoadError, load_member  # noqa: E402
from dbp.decoders.reconstruct import (  # noqa: E402
    max_abs_diff,
    prepare_input,
    psnr_uint8,
    verify_diffusers_api,
)
from dbp.decoders.registry import DecoderSpec, load_bank_config  # noqa: E402
from dbp.utils.io import load_rgb_uint8, residual_int16, save_png_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402
from p0_verify_pins import PinVerificationError, require_pins  # noqa: E402

MANIFEST_COLUMNS = (
    "pair_id",
    "image_id",
    "set_id",
    "member",
    "lineage",
    "kind",
    "revision",
    "precision",
    "crop",
    "relative_path",
    "sha256",
    "psnr_db",
    "mean_abs_residual",
    "max_abs_residual",
    "ms_per_img",
)

#: Images per chunk. Small enough that a killed session loses little, large enough that
#: the manifest append is not the bottleneck.
CHUNK = 250


def member_precision(spec: DecoderSpec, throughput_csv: Path, log: Any) -> str:
    """Decide fp32 or fp16 for one member from the measured parity.

    The decision is read from the bench output rather than recomputed, so the build and
    the report cannot disagree. Anything unmeasured is fp32.
    """
    if not throughput_csv.is_file():
        return "fp32"
    with throughput_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("key") != spec.key:
                continue
            verdict = str(row.get("fp16_build_allowed", "")).strip().lower()
            if verdict in ("true", "1", "yes"):
                log.info("%-14s fp16 allowed for the build: %s", spec.key,
                         row.get("fp16_build_reason", ""))
                return "fp16"
            if verdict in ("false", "0", "no"):
                log.info("%-14s fp32 forced: %s", spec.key, row.get("fp16_build_reason", ""))
                return "fp32"
    return "fp32"


def select_shard(specs: list[DecoderSpec], shard: int, num_shards: int) -> list[DecoderSpec]:
    """Round-robin the member list, so each shard gets cheap and expensive members."""
    if num_shards < 1 or not 0 <= shard < num_shards:
        raise ValueError(f"invalid shard {shard} of {num_shards}")
    return [spec for index, spec in enumerate(specs) if index % num_shards == shard]


def load_reals(manifests: list[Path], reals_root: Path, log: Any) -> list[dict[str, str]]:
    """Read the real-image manifests and resolve each row to a readable file."""
    rows: list[dict[str, str]] = []
    for manifest in manifests:
        if not manifest.is_file():
            raise FileNotFoundError(f"reals manifest not found: {manifest}")
        for row in read_manifest(manifest):
            path = reals_root / row["set_id"] / f"{row['image_id']}.png"
            if not path.is_file():
                raise FileNotFoundError(
                    f"{manifest.name} lists {row['image_id']} but {path} is missing. "
                    f"Rebuild the reals before building pairs."
                )
            row["_path"] = str(path)
            rows.append(row)
        log.info("%s: %d images", manifest.name, len(read_manifest(manifest)))
    return rows


def build_for_member(
    spec: DecoderSpec,
    reals: list[dict[str, str]],
    device: torch.device,
    cache_dir: Path,
    out_root: Path,
    throughput_csv: Path,
    log: Any,
    name: str = "p1_pairs",
) -> dict[str, Any]:
    """Reconstruct every real through one member, resuming if a manifest already exists."""
    manifest_path = THESIS_DBP / "reports" / f"{name}_manifest_{spec.key}.csv"
    done = completed_ids(manifest_path)
    todo = [row for row in reals if row["image_id"] not in done]
    if done:
        log.info("%-14s resuming: %d done, %d to go", spec.key, len(done), len(todo))
    if not todo:
        log.info("%-14s already complete", spec.key)
        return {"member": spec.key, "written": 0, "already_done": len(done), "status": "OK"}

    precision = member_precision(spec, throughput_csv, log)
    try:
        member = load_member(spec, device=device, precision=precision, cache_dir=cache_dir)
    except MemberLoadError as exc:
        log.error("%-14s load failed: %s", spec.key, str(exc)[:300])
        return {
            "member": spec.key,
            "written": 0,
            "already_done": len(done),
            "status": "BLOCKED-ACCESS" if "gated" in str(exc).lower() else "ERROR",
            "detail": str(exc)[:300],
        }

    member_dir = out_root / spec.key
    member_dir.mkdir(parents=True, exist_ok=True)
    is_cuda = device.type == "cuda"

    written = 0
    bytes_written = 0
    failures: list[dict[str, str]] = []
    chunk_rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index, real in enumerate(todo, start=1):
        try:
            source = load_rgb_uint8(real["_path"])
            if is_cuda:
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            recon = member.reconstruct(source)
            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - begin) * 1000.0

            cropped = prepare_input(source, spec.crop_multiple)
            residual = residual_int16(cropped, recon)
            destination = member_dir / f"{real['image_id']}.png"
            save_png_uint8(recon, destination)
            bytes_written += destination.stat().st_size

            chunk_rows.append(
                {
                    "pair_id": f"{real['image_id']}__{spec.key}",
                    "image_id": real["image_id"],
                    "set_id": real["set_id"],
                    "member": spec.key,
                    "lineage": spec.lineage,
                    "kind": spec.kind,
                    "revision": (spec.revision or spec.sha256 or "")[:12],
                    "precision": precision,
                    "crop": recon.shape[0],
                    "relative_path": destination.relative_to(out_root.parent).as_posix(),
                    "sha256": file_sha256(destination),
                    "psnr_db": round(psnr_uint8(cropped, recon), 4),
                    "mean_abs_residual": round(float(np.mean(np.abs(residual))), 5),
                    "max_abs_residual": max_abs_diff(cropped, recon),
                    "ms_per_img": round(elapsed_ms, 2) if is_cuda else "",
                }
            )
        except torch.cuda.OutOfMemoryError:
            failures.append(
                {"image_id": real["image_id"], "error": "CUDA OOM (no tiling by design)"}
            )
            log.error("%-14s OOM on %s", spec.key, real["image_id"])
            torch.cuda.empty_cache()
        except Exception as exc:
            failures.append(
                {"image_id": real["image_id"], "error": f"{type(exc).__name__}: {exc}"[:200]}
            )
            log.error("%-14s failed on %s: %s", spec.key, real["image_id"], exc)

        if len(chunk_rows) >= CHUNK or index == len(todo):
            # Rows are appended only after their PNGs exist, so the manifest never
            # describes a file that is not there.
            append_rows(chunk_rows, manifest_path, MANIFEST_COLUMNS)
            written += len(chunk_rows)
            rate = written / max(time.perf_counter() - started, 1e-9)
            log.info(
                "%-14s %d/%d written (%.2f img/s, %.1f min remaining)",
                spec.key,
                written,
                len(todo),
                rate,
                (len(todo) - written) / max(rate, 1e-9) / 60.0,
            )
            chunk_rows = []

    member.free()
    del member
    if is_cuda:
        torch.cuda.empty_cache()

    return {
        "member": spec.key,
        "lineage": spec.lineage,
        "precision": precision,
        "written": written,
        "bytes_written": bytes_written,
        "already_done": len(done),
        "failures": failures,
        "manifest": manifest_path.name,
        "status": "OK" if not failures else "PARTIAL",
    }


def merge_manifests(log: Any, name: str = "p1_pairs") -> Path:
    """Concatenate the per-member manifests into the single pairs manifest."""
    shards = sorted((THESIS_DBP / "reports").glob(f"{name}_manifest_*.csv"))
    rows: list[dict[str, str]] = []
    for shard in shards:
        shard_rows = read_manifest(shard)
        rows.extend(shard_rows)
        log.info("%s: %d rows", shard.name, len(shard_rows))
    rows.sort(key=lambda r: (r.get("image_id", ""), r.get("member", "")))
    destination = THESIS_DBP / "reports" / f"{name}_manifest.csv"
    write_manifest(rows, destination, MANIFEST_COLUMNS)
    log.info("wrote %s (%d rows from %d members)", destination, len(rows), len(shards))
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-B5 aligned reconstruction pairs.")
    parser.add_argument(
        "--reals-manifest",
        type=Path,
        action="append",
        default=None,
        help="Repeatable. Defaults to both t_real and t_ctrl manifests.",
    )
    parser.add_argument(
        "--reals-root", type=Path, default=THESIS_DBP / "outputs" / "p1" / "reals"
    )
    parser.add_argument(
        "--out-root", type=Path, default=THESIS_DBP / "outputs" / "p1" / "pairs"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--members", default="", help="Comma-separated keys; default is all.")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--throughput-csv", type=Path, default=THESIS_DBP / "reports" / "p1_throughput.csv"
    )
    parser.add_argument("--merge", action="store_true", help="Only merge existing manifests.")
    parser.add_argument("--name", default="p1_pairs",
                        help="Manifest/summary name stem (P3 4a: p3_oi_pairs).")
    parser.add_argument("--cache-dir", type=Path, default=THESIS_DBP / ".cache" / "weights",
                        help="Downloaded weights (the Real-ESRGAN .pth); /tmp on Kaggle.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Pre-flight sample: only the first N reals.")
    parser.add_argument("--skip-pin-check", action="store_true", help="Diagnostics only.")
    args = parser.parse_args()

    log = setup_logging(f"p1_build_pairs_s{args.shard}", log_dir=THESIS_DBP / "outputs" / "logs")

    if args.merge:
        merge_manifests(log, args.name)
        return 0

    log.info("seed: %s", seed_everything(2026, deterministic=True))
    api = verify_diffusers_api()
    if not all(api.values()):
        log.error("diffusers output attributes changed: %s", api)
        return 2

    if not args.skip_pin_check:
        try:
            require_pins(printer=log.info)
        except PinVerificationError as exc:
            log.error("%s", exc)
            return 2

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log.error("--device %s requested but no CUDA device is present.", args.device)
        return 2
    log.info("device: %s", device)

    config = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    specs = config.distinct_members()
    if args.members:
        wanted = {m.strip() for m in args.members.split(",") if m.strip()}
        specs = [s for s in specs if s.key in wanted]
    specs = select_shard(specs, args.shard, args.num_shards)
    log.info(
        "shard %d/%d -> %d members: %s",
        args.shard,
        args.num_shards,
        len(specs),
        [s.key for s in specs],
    )
    log.info("aliases skipped by the registry: %s", config.aliases() or "none")

    manifests = args.reals_manifest or [
        THESIS_DBP / "reports" / "p1_reals_manifest_t_real.csv",
        THESIS_DBP / "reports" / "p1_reals_manifest_t_ctrl.csv",
    ]
    try:
        reals = load_reals([Path(m) for m in manifests], args.reals_root, log)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 2
    if args.limit:
        reals = reals[: args.limit]
    log.info("%d real images x %d members = %d pairs", len(reals), len(specs),
             len(reals) * len(specs))

    args.out_root.mkdir(parents=True, exist_ok=True)
    summaries = [
        build_for_member(spec, reals, device, args.cache_dir, args.out_root,
                         args.throughput_csv, log, args.name)
        for spec in specs
    ]

    # Jobs given explicit member lists (P3 4a) all run as shard 0, so their summaries are
    # named by their members; a sharded P1 run keeps its shard name.
    tag = "+".join(s.key for s in specs) if args.members else f"shard{args.shard}"
    summary_path = THESIS_DBP / "reports" / f"{args.name}_summary_{tag}.json"
    summary_path.write_text(
        json.dumps(
            {
                "shard": args.shard,
                "num_shards": args.num_shards,
                "device": str(device),
                "n_reals": len(reals),
                "limit": args.limit or None,
                "throughput_csv": str(args.throughput_csv),
                "members": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log.info("wrote %s", summary_path)

    failed = [s for s in summaries if s["status"] not in ("OK",)]
    for entry in failed:
        log.warning("  %-14s %s %s", entry["member"], entry["status"], entry.get("detail", ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
