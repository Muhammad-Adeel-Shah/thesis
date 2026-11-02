"""P2-B5: run every image through all twelve decoders and cache the probe features.

This is the dominant GPU cost of P2. Two things keep it affordable:

* the bank is **fixed at test time**, so features are computed once per image and every
  protocol's head reuses them -- computing them per protocol would triple the bill;
* it shards by member exactly as the P1 pair build did, so a T4 x2 session halves the
  wall clock. ``--members`` selects a shard; ``--shard i --num-shards n`` splits the
  bank automatically.

The loop is member-outer, image-inner: a member is loaded once, used for every image,
then freed. Loading twelve decoders per image would spend most of the session moving
weights.

**Resumable, and verified rather than assumed.** Each member writes
``p2_probe_<key>.npz`` plus a sidecar ``p2_probe_<key>.done.json`` holding that file's
SHA-256, its row count and the crop it was built at. A re-run skips a member only when
the sidecar exists **and** the file still hashes to what the sidecar says **and** the
image list has not changed underneath it; anything else is recomputed and the reason is
logged. A half-written ``.npz`` from a killed session therefore costs one member, not a
silent shard of garbage that every downstream head would then read as real residuals.
The sidecar is per member on purpose: a single shared manifest would be written
concurrently by both T4 shards and one would lose.

Reconstruction goes through ``dbp.decoders.reconstruct`` only, via
:class:`dbp.decoders.loaders.BankMember` -- never resized, never tiled, ``mode()`` not
``sample()``, and at the D-016 precision the member is recorded with. The crop is 320
centred (D-045), which is a different crop from the 384 the P1 pairs were built at, so
these PSNRs are **not** expected to reproduce the pair manifests'.

``--calibrate N`` runs N images through one member and reports the measured per-image
cost including LPIPS, then exits. That number was never measured in P0 or P1 and the
budget carries it only as contingency, so it is measured before the full run rather
than assumed.

Usage:
    python scripts/p2_probe_features.py --images reports/p2_probe_images.csv \\
        --root /kaggle/input/.../outputs/p1 --device cuda --shard 0 --num-shards 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.decoders.loaders import load_member  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.streams.s3_probe import FEATURE_NAMES, member_features  # noqa: E402
from dbp.train.augment import CROP_SIZE, centre_crop  # noqa: E402
from dbp.utils.io import load_rgb_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402


def read_image_list(path: Path) -> list[dict[str, str]]:
    """Rows of ``image_id,relative_path,label,view`` naming what to probe."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "relative_path"}
    if rows and not required <= set(rows[0]):
        raise ValueError(f"image list needs {sorted(required)}, has {sorted(rows[0])}")
    return rows


def timing_report_name(tag: str) -> str:
    """File name of one job's timing report under ``reports/``.

    Two parallel jobs both ran as ``shard 0/1`` in the S3 recovery and both wrote
    ``p2_probe_timing_shard0.json``; the later one replaced the other's record
    (I-053). A runner therefore passes a tag naming the job and its device.
    """
    if not tag or any(c in tag for c in "/\\ "):
        raise ValueError(f"timing tag must be a non-empty plain name, got {tag!r}")
    return f"p2_probe_timing_{tag}.json"


def image_list_digest(rows: list[dict[str, str]]) -> str:
    """SHA-256 over an image list's ids and paths, in order.

    Over the ids and paths, not the file bytes: what matters is that a run would probe
    the same images in the same order as the run that wrote a cache. Every sidecar
    records this value, and the restore and pre-flight steps recompute it with this
    function rather than a copy of the formula, so the three can never disagree.
    """
    return hashlib.sha256(
        "\n".join(f"{r['image_id']}\t{r['relative_path']}" for r in rows).encode()
    ).hexdigest()


def build_lpips(device: str) -> Any:
    """The LPIPS-vgg network AEROBLADE uses, loaded once per process."""
    import lpips
    import torch

    net = lpips.LPIPS(net="vgg").to(torch.device(device))
    net.eval()
    return net


def lpips_distance(net: Any, a_uint8: np.ndarray, b_uint8: np.ndarray, device: str) -> float:
    """LPIPS-vgg between two uint8 crops, on the [-1, 1] range the network expects."""
    import torch

    def tensor(image: np.ndarray) -> Any:
        array = image.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        return float(net(tensor(a_uint8), tensor(b_uint8)).item())


def shard_is_complete(
    out_path: Path, expected_rows: int, crop: int, images_digest: str, log: Any
) -> bool:
    """Whether a member's cache can be trusted and skipped.

    Four conditions, and all of them have to hold. "The file exists" is not one of
    them: a session killed mid-write leaves a file that exists, and a head reading it
    would treat truncated or garbage residuals as real ones.

    Args:
        out_path: The member's ``.npz``.
        expected_rows: How many images this run would compute.
        crop: The crop this run uses.
        images_digest: SHA-256 over the image list, so a changed list forces a rebuild.
        log: Logger.

    Returns:
        True only when the cache is present, intact and built for this exact run.
    """
    sidecar = out_path.with_suffix(".done.json")
    if not out_path.is_file():
        return False
    if not sidecar.is_file():
        log.info("  %s: file present but no sidecar -- recomputing", out_path.name)
        return False
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    actual = file_sha256(out_path)
    if actual != record.get("sha256"):
        log.info("  %s: sha256 differs from the sidecar -- recomputing", out_path.name)
        return False
    if int(record.get("rows", -1)) != expected_rows:
        log.info(
            "  %s: sidecar says %s rows, this run needs %d -- recomputing",
            out_path.name, record.get("rows"), expected_rows,
        )
        return False
    if int(record.get("crop", -1)) != crop:
        log.info(
            "  %s: built at crop %s, this run is at %d -- recomputing",
            out_path.name, record.get("crop"), crop,
        )
        return False
    if record.get("images_sha256") != images_digest:
        log.info("  %s: the image list changed -- recomputing", out_path.name)
        return False
    return True


def write_sidecar(out_path: Path, rows: int, crop: int, images_digest: str, member: str) -> None:
    """Record what a finished shard contains, so the next run can verify it."""
    sidecar = out_path.with_suffix(".done.json")
    tmp = sidecar.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "member": member,
                "sha256": file_sha256(out_path),
                "rows": rows,
                "crop": crop,
                "images_sha256": images_digest,
                "bytes": out_path.stat().st_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    tmp.replace(sidecar)


def select_members(bank: Any, members: str | None, shard: int, num_shards: int) -> list[Any]:
    """Distinct members for this shard, in a stable order."""
    distinct = list(bank.distinct_members())
    if members:
        wanted = {m.strip() for m in members.split(",") if m.strip()}
        unknown = wanted - {spec.key for spec in distinct}
        if unknown:
            raise ValueError(f"unknown member(s): {sorted(unknown)}")
        return [spec for spec in distinct if spec.key in wanted]
    return [spec for index, spec in enumerate(distinct) if index % num_shards == shard]


def resolve_roots(root: Path, panel_root: Path | None) -> dict[str, Path]:
    """Map an image list's ``root`` column onto real directories.

    A list may name two: ``corpus`` for the reals, pairs and generations, and ``panel``
    for the frozen evaluation panel, which is materialised from evaluation sets that
    are not in the corpus. A row with no ``root`` value falls back to the corpus, which
    is what a list written before the column existed means.
    """
    roots = {"corpus": root, "": root}
    if panel_root is not None:
        roots["panel"] = panel_root
    return roots


def row_path(row: dict[str, str], roots: dict[str, Path]) -> Path:
    """Absolute path of one image-list row.

    Raises:
        KeyError: If the row names a root that was not supplied. Falling back to the
            corpus root would read a panel path against the wrong directory and fail
            later with a confusing message.
    """
    name = str(row.get("root", "") or "")
    if name not in roots:
        raise KeyError(
            f"row {row.get('image_id')!r} names root {name!r}, which was not given; "
            f"pass --{name}-root"
        )
    return roots[name] / row["relative_path"]


def main() -> int:
    """Cache probe features for one shard of the bank. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path,
                        help="Corpus root: reals, pairs, generated, gen_matched.")
    parser.add_argument(
        "--panel-root", type=Path, default=None,
        help="Frozen panel root from p2_materialise_panel.py. Required when the "
             "image list contains rows with root=panel.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--members", default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--tag", default=None,
        help="Names this job's timing report. Defaults to shard<i>; a parallel runner "
             "must pass a distinct tag per job (I-053).",
    )
    parser.add_argument("--crop", type=int, default=CROP_SIZE)
    parser.add_argument(
        "--calibrate", type=int, default=0,
        help="Measure per-image cost on N images with one member, then exit.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "probe"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/tmp/dbp_weights"),
        help="Real-ESRGAN weights cache. Never inside /kaggle/working (section 3).",
    )
    args = parser.parse_args()
    log = setup_logging("p2_probe_features")

    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    specs = select_members(bank, args.members, args.shard, args.num_shards)
    images = read_image_list(args.images)
    log.info(
        "shard %d/%d: %d member(s) x %d image(s) at crop %d",
        args.shard, args.num_shards, len(specs), len(images), args.crop,
    )
    log.info("members: %s", [spec.key for spec in specs])

    images_digest = image_list_digest(images)
    log.info("image-list digest: %s", images_digest[:16])

    roots = resolve_roots(args.root, args.panel_root)
    log.info("roots: %s", {k: str(v) for k, v in roots.items() if k})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    net = build_lpips(args.device)

    # Throughput table for the P1 members, so precision follows what P1 recorded.
    precision_of = _precision_from_throughput(log)

    summaries: list[dict[str, Any]] = []
    skipped: list[str] = []
    for spec in specs:
        precision = precision_of.get(spec.key, "fp32")
        out_path = args.out_dir / f"p2_probe_{spec.key}.npz"
        if not args.calibrate and shard_is_complete(
            out_path, len(images), args.crop, images_digest, log
        ):
            log.info("%s: cache verified, skipping", spec.key)
            skipped.append(spec.key)
            continue

        log.info("loading %s (%s, %s)", spec.key, spec.kind, precision)
        member = load_member(spec, args.device, precision, args.cache_dir)

        subset = images[: args.calibrate] if args.calibrate else images
        features = np.zeros((len(subset), len(FEATURE_NAMES)), dtype=np.float32)
        ids: list[str] = []
        started = time.perf_counter()
        for index, row in enumerate(subset):
            crop = centre_crop(load_rgb_uint8(row_path(row, roots)), args.crop)
            reconstruction = member.reconstruct(crop)
            # sr_gan and the VAEs crop to different multiples, so compare on the
            # overlap rather than assuming the shapes match.
            height = min(crop.shape[0], reconstruction.shape[0])
            width = min(crop.shape[1], reconstruction.shape[1])
            left, right = crop[:height, :width], reconstruction[:height, :width]
            features[index] = member_features(
                left, right, lpips_distance(net, left, right, args.device)
            )
            ids.append(str(row["image_id"]))
            if index and index % 200 == 0:
                rate = (time.perf_counter() - started) / index
                log.info("  %s %d/%d  %.3f s/image", spec.key, index, len(subset), rate)

        elapsed = time.perf_counter() - started
        per_image = elapsed / max(len(subset), 1)
        log.info(
            "%s: %d image(s) in %.1f s = %.4f s/image (reconstruct + LPIPS)",
            spec.key, len(subset), elapsed, per_image,
        )
        summaries.append(
            {
                "member": spec.key,
                "kind": spec.kind,
                "lineage": spec.lineage,
                "precision": precision,
                "images": len(subset),
                "seconds_per_image": round(per_image, 5),
                "crop": args.crop,
            }
        )
        if not args.calibrate:
            tmp = out_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                tmp,
                features=features,
                image_ids=np.array(ids),
                feature_names=np.array(FEATURE_NAMES),
                member=spec.key,
            )
            tmp.replace(out_path)
            write_sidecar(out_path, len(subset), args.crop, images_digest, spec.key)
            log.info("wrote %s and its sidecar", out_path.name)
        member.free()

    report = {
        "shard": args.shard,
        "num_shards": args.num_shards,
        "crop": args.crop,
        "images": len(images),
        "calibration_only": bool(args.calibrate),
        "images_sha256": images_digest,
        "skipped_verified": skipped,
        "members": summaries,
        "projected_full_run_gpu_hours": round(
            sum(s["seconds_per_image"] for s in summaries) * len(images) / 3600.0, 3
        )
        if summaries
        else None,
    }
    name = timing_report_name(args.tag or f"shard{args.shard}")
    path = THESIS_DBP / "reports" / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    log.info("wrote reports/%s", name)
    if skipped:
        log.info("%d member(s) skipped with a verified cache: %s", len(skipped), skipped)
    if args.calibrate:
        log.info(
            "calibration only -- nothing cached. Projected for these members over all "
            "%d images: %.2f GPU-h", len(images), report["projected_full_run_gpu_hours"] or 0.0
        )
    return 0


def _precision_from_throughput(log: Any) -> dict[str, str]:
    """Per-member precision, taken from P1's recorded ``fp16_build_allowed``.

    S3 must reconstruct at the same precision the pairs were built at, or the residual
    statistics of a training fake and a test image would come from different numerics.
    """
    path = THESIS_DBP / "reports" / "kaggle_20260926" / "p1_throughput.csv"
    if not path.is_file():
        log.info("no throughput table found; every member runs fp32 (conservative)")
        return {}
    allowed: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            ok = str(row.get("fp16_build_allowed", "")).strip().lower() in ("true", "1", "yes")
            allowed[row["key"]] = "fp16" if ok else "fp32"
    log.info(
        "precision from P1: %s",
        {k: v for k, v in sorted(allowed.items()) if v == "fp32"} or "all fp16",
    )
    return allowed


if __name__ == "__main__":
    raise SystemExit(main())
