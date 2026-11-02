"""P2-B3: cache frozen DINOv3 features for S1 (and for S2's inference path).

The backbone is loaded once, frozen, and run over a list of images at 320 centre crop.
The feature is ``[CLS || mean patch]`` with the **register tokens excluded**, and the
register count is read from the loaded model's own config -- DINOv3 has four, DINOv2
none, and a hard-coded number would quietly average four register vectors into the
patch mean.

Features are cached as fp16 ``.npz`` beside their image ids, so the probe, the stacker
and every re-score read the same numbers rather than recomputing them. At 1,536 dims
that is about 3 KB an image.

Training features come from a **fixed centre crop**, not from augmented views. That is
a recorded limitation of S1 (section 6.2 of the plan): it is what makes the frozen
stream cheap, and it means S1's robustness numbers are read as "frozen features of an
unaugmented view", not as "a model that saw degradations".

Usage:
    python scripts/p2_extract_s1.py --images reports/p2_s1_images.csv \\
        --root /kaggle/input/.../outputs/p1 --device cuda --batch 32
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.backbones.loader import BACKBONES, load_backbone  # noqa: E402
from dbp.streams.s1_semantic import (  # noqa: E402
    expected_patch_tokens,
    normalise_batch,
    pool_tokens,
    processor_normalisation,
    register_token_count,
)
from dbp.train.augment import CROP_SIZE, centre_crop  # noqa: E402
from dbp.utils.hf import get_hf_token  # noqa: E402
from dbp.utils.io import load_rgb_uint8, reencode_jpeg  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: The backbone approved for P2 (Phase A, A3): ViT-B/16, because no committed
#: measurement licenses ViT-L/16 and Phase A was not allowed to take one.
DEFAULT_BACKBONE = "dinov3-vitb16"


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
    """Cache S1 features for one image list. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path,
                        help="Corpus root: reals, pairs, generated, gen_matched.")
    parser.add_argument(
        "--panel-root", type=Path, default=None,
        help="Frozen panel root from p2_materialise_panel.py. Required when the "
             "image list contains rows with root=panel.",
    )
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE, choices=sorted(BACKBONES))
    parser.add_argument("--device", default="cuda",
                        help="cuda, cpu, or auto (cuda if available).")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--crop", type=int, default=CROP_SIZE)
    parser.add_argument("--tag", default="panel", help="Names the output file.")
    parser.add_argument(
        "--reencode-jpeg", type=int, default=None, metavar="QUALITY",
        help="Format control (X1): round-trip every full image through JPEG at QUALITY, "
             "4:4:4, before the crop. Off by default; P2's caches never used it.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N rows (the pre-flight smoke test).")
    parser.add_argument(
        "--out-dir", type=Path, default=THESIS_DBP / "outputs" / "p2" / "s1"
    )
    args = parser.parse_args()
    log = setup_logging("p2_extract_s1")

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("device auto -> %s", args.device)
    token = get_hf_token()
    log.info("HF_TOKEN present: %s", token is not None)

    backbone = load_backbone(args.backbone, args.device, token)
    n_register = register_token_count(backbone.config)
    n_patch = expected_patch_tokens(args.crop, backbone.patch_size)
    log.info(
        "%s @ %s: hidden %d, patch %d, %d register token(s), %d patch token(s) at crop %d",
        backbone.name, backbone.revision[:12], backbone.hidden_size,
        backbone.patch_size, n_register, n_patch, args.crop,
    )

    normalisation = processor_normalisation(backbone.repo, backbone.revision, token)
    log.info("normalisation from %s", normalisation["source"])

    with args.images.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]
    log.info("%d image(s) to encode%s", len(rows),
             f", re-encoded as JPEG q{args.reencode_jpeg} 4:4:4" if args.reencode_jpeg else "")

    def load(row: dict[str, str]) -> np.ndarray:
        image = load_rgb_uint8(row_path(row, roots))
        return reencode_jpeg(image, args.reencode_jpeg) if args.reencode_jpeg else image

    roots = resolve_roots(args.root, args.panel_root)
    log.info("roots: %s", {k: str(v) for k, v in roots.items() if k})

    features = np.zeros((len(rows), 2 * backbone.hidden_size), dtype=np.float16)
    ids: list[str] = []
    started = time.perf_counter()
    use_amp = args.device.startswith("cuda")

    for start in range(0, len(rows), args.batch):
        chunk = rows[start : start + args.batch]
        crops = np.stack(
            [
                centre_crop(load(row), args.crop)
                for row in chunk
            ]
        )
        batch = normalise_batch(
            crops, normalisation["image_mean"], normalisation["image_std"]
        ).to(args.device)
        with torch.no_grad():
            # fp16 autocast, never bf16: the T4 is sm_75 and has no bf16 tensor cores.
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                hidden = backbone.model(pixel_values=batch).last_hidden_state
            pooled = pool_tokens(hidden.float(), n_register, n_patch)
        features[start : start + len(chunk)] = pooled.cpu().numpy().astype(np.float16)
        ids.extend(str(row["image_id"]) for row in chunk)
        if start and (start // args.batch) % 20 == 0:
            rate = (time.perf_counter() - started) / start
            log.info("  %d/%d  %.4f s/image", start, len(rows), rate)

    elapsed = time.perf_counter() - started
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"p2_s1_{args.backbone}_{args.tag}.npz"
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, features=features, image_ids=np.array(ids))
    tmp.replace(out)

    report = {
        "backbone": backbone.name,
        "repo": backbone.repo,
        "revision": backbone.revision,
        "crop": args.crop,
        "batch": args.batch,
        "num_register_tokens": n_register,
        "num_patch_tokens": n_patch,
        "feature_dim": int(features.shape[1]),
        "pooling": "[CLS || mean patch], register tokens excluded",
        "normalisation": normalisation,
        "images": len(rows),
        "seconds": round(elapsed, 2),
        "seconds_per_image": round(elapsed / max(len(rows), 1), 5),
        "dtype": "float16",
        "cache": out.name,
        "train_crop_policy": "fixed centre crop, not augmented (recorded S1 limitation)",
        "device": args.device,
        "limit": args.limit,
        "reencode": (None if not args.reencode_jpeg else
                     {"format": "JPEG", "quality": args.reencode_jpeg, "subsampling": "4:4:4",
                      "applied": "to the full image, before the centre crop"}),
    }
    path = THESIS_DBP / "reports" / f"p2_s1_extract_{args.tag}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)

    log.info(
        "%d features of dim %d in %.1f s (%.4f s/image) -> %s",
        len(rows), features.shape[1], elapsed, report["seconds_per_image"], out.name,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
