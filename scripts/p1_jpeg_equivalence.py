"""P1-B2: is a JPEG written on Kaggle the same JPEG as one written here? (I-010)

libjpeg is not one program. libjpeg-turbo, mozjpeg and the IJG reference differ in how
they round the DCT and in which optimisations they apply, so the same array encoded at
the same quality on two machines can differ byte for byte, and sometimes pixel for
pixel after decoding. That matters here for one specific reason: the JPEG-matched views
(CLAUDE.md 6b) re-encode images, and if that encode is not portable then the evaluation
set is a property of the machine that built it.

The run produces, for a fixed set of deterministic test images and a grid of quality
and chroma-subsampling settings:

- the SHA-256 of the encoded bytes,
- the encoded size,
- the SHA-256 of the decoded pixels, and the max and mean absolute pixel difference
  against the source,

together with the Pillow, libjpeg and platform versions. Run it on both machines and
give the second run ``--compare-to`` the first run's JSON: it prints exactly which
cells differ, in bytes and in pixels, and exits non-zero if any do.

The two failure modes are very different and the report separates them. Different bytes
with identical decoded pixels is a container or entropy-coding difference and is
harmless as long as the matched view is built in one place. Different decoded pixels is
a numerical difference, and then the matched view must be built on Kaggle only -- which
is the rule I-010 already states, and this makes it evidence rather than caution.

Usage:
    python scripts/p1_jpeg_equivalence.py --out reports/p1_jpeg_equivalence.json
    python scripts/p1_jpeg_equivalence.py --compare-to reports/p1_jpeg_equivalence.json
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import PIL
from PIL import Image, features

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.jpeg import SUBSAMPLING_LABELS, encode_jpeg_bytes, jpeg_metadata  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

#: The qualities the P1 plan requires, plus 95 which is where most camera JPEGs sit.
QUALITIES = (50, 75, 90, 95)

#: Test-image sizes. 383 is deliberately not a multiple of 16: chroma subsampling pads
#: the final MCU, and that padding is one of the places encoders differ.
IMAGE_SIZES = ((256, 256), (383, 257))


def test_image(height: int, width: int, seed: int) -> np.ndarray:
    """A deterministic image with smooth gradients, hard edges and fine texture.

    All three matter: gradients expose DCT rounding, edges expose ringing differences,
    and fine texture exposes the quantisation of high-frequency coefficients. A single
    flat or noise image would hide two of the three.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)

    gradient = 40.0 + 170.0 * (xs / max(width - 1, 1))
    rings = 30.0 * np.sin(np.sqrt((xs - width / 2) ** 2 + (ys - height / 2) ** 2) / 4.0)
    texture = rng.normal(0.0, 9.0, size=(height, width))

    luma = np.clip(gradient + rings + texture, 0, 255)
    # Hard edges: a few saturated blocks, which is where ringing shows up.
    luma[height // 8 : height // 3, width // 8 : width // 3] = 255.0
    luma[height // 2 : height // 2 + height // 6, width // 2 : width // 2 + width // 6] = 0.0

    stack = np.stack(
        [
            luma,
            np.clip(luma * 0.82 + 24.0, 0, 255),
            np.clip(luma * 0.65 + 48.0, 0, 255),
        ],
        axis=-1,
    )
    return stack.astype(np.uint8)


def sha256_bytes(payload: bytes) -> str:
    """Hex SHA-256 of a byte string."""
    return hashlib.sha256(payload).hexdigest()


def sha256_array(array: np.ndarray) -> str:
    """Hex SHA-256 of an array's raw bytes, in C order."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def environment() -> dict[str, Any]:
    """Everything that could plausibly change the encoded bytes."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "pillow": PIL.__version__,
        "jpeg_feature": features.check("jpg"),
        "libjpeg_version": features.version("jpg"),
        "jpg_2000": features.version("jpg_2000"),
        "numpy": np.__version__,
    }


def run_grid(log: Any) -> list[dict[str, Any]]:
    """Encode every (image, quality, subsampling) cell and describe the result."""
    rows: list[dict[str, Any]] = []
    for index, (height, width) in enumerate(IMAGE_SIZES):
        source = test_image(height, width, seed=2026 + index)
        source_sha = sha256_array(source)
        for quality in QUALITIES:
            for subsampling, label in sorted(SUBSAMPLING_LABELS.items()):
                payload = encode_jpeg_bytes(source, quality, subsampling)
                with Image.open(io.BytesIO(payload)) as handle:
                    decoded = np.asarray(handle.convert("RGB"), dtype=np.uint8)

                difference = np.abs(decoded.astype(np.int16) - source.astype(np.int16))
                rows.append(
                    {
                        "cell": f"{height}x{width}_q{quality}_{label.replace(':', '')}",
                        "height": height,
                        "width": width,
                        "quality": quality,
                        "subsampling": label,
                        "source_sha256": source_sha,
                        "encoded_bytes": len(payload),
                        "encoded_sha256": sha256_bytes(payload),
                        "decoded_sha256": sha256_array(decoded),
                        "decoded_max_abs_diff": int(difference.max()),
                        "decoded_mean_abs_diff": round(float(difference.mean()), 6),
                    }
                )
                log.info(
                    "%-28s %7d bytes  bytes:%s  pixels:%s",
                    rows[-1]["cell"],
                    rows[-1]["encoded_bytes"],
                    rows[-1]["encoded_sha256"][:12],
                    rows[-1]["decoded_sha256"][:12],
                )
    return rows


def round_trip_check(log: Any) -> dict[str, Any]:
    """Confirm that what this machine writes, it also reads back consistently.

    A machine that cannot reproduce its own quality setting would make every cross-
    machine comparison uninterpretable, so it is checked before the comparison.
    """
    source = test_image(128, 128, seed=99)
    outcomes = []
    for quality in QUALITIES:
        payload = encode_jpeg_bytes(source, quality, 0)
        buffer = io.BytesIO(payload)
        buffer.seek(0)
        temporary = THESIS_DBP / "outputs" / "p1" / f"_jpeg_probe_q{quality}.jpg"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_bytes(payload)
        estimated = jpeg_metadata(temporary)["jpeg_quality"]
        temporary.unlink()
        outcomes.append({"set": quality, "estimated": estimated, "match": estimated == quality})
        log.info("round trip q=%d -> estimated %s", quality, estimated)
    return {"per_quality": outcomes, "all_match": all(o["match"] for o in outcomes)}


def compare(current: dict[str, Any], baseline: dict[str, Any], log: Any) -> dict[str, Any]:
    """Diff two runs cell by cell, separating byte from pixel differences."""
    baseline_rows = {row["cell"]: row for row in baseline["cells"]}
    byte_differences: list[dict[str, Any]] = []
    pixel_differences: list[dict[str, Any]] = []
    missing: list[str] = []

    for row in current["cells"]:
        other = baseline_rows.get(row["cell"])
        if other is None:
            missing.append(row["cell"])
            continue
        if row["source_sha256"] != other["source_sha256"]:
            raise RuntimeError(
                f"{row['cell']}: the two runs did not start from the same source image. "
                f"The comparison would be meaningless."
            )
        if row["encoded_sha256"] != other["encoded_sha256"]:
            byte_differences.append(
                {
                    "cell": row["cell"],
                    "this_sha256": row["encoded_sha256"],
                    "other_sha256": other["encoded_sha256"],
                    "this_bytes": row["encoded_bytes"],
                    "other_bytes": other["encoded_bytes"],
                }
            )
        if row["decoded_sha256"] != other["decoded_sha256"]:
            pixel_differences.append(
                {
                    "cell": row["cell"],
                    "this_max_abs_diff_vs_source": row["decoded_max_abs_diff"],
                    "other_max_abs_diff_vs_source": other["decoded_max_abs_diff"],
                    "this_mean_abs_diff_vs_source": row["decoded_mean_abs_diff"],
                    "other_mean_abs_diff_vs_source": other["decoded_mean_abs_diff"],
                }
            )

    if pixel_differences:
        verdict = "PIXELS-DIFFER"
        meaning = (
            "The two libjpeg builds decode to different pixels. Every JPEG-matched view "
            "must be built in ONE place (Kaggle), and the build machine must be recorded "
            "in the data card."
        )
    elif byte_differences:
        verdict = "BYTES-DIFFER-PIXELS-MATCH"
        meaning = (
            "Container or entropy-coding differences only; the decoded images agree. "
            "Matched views are still built in one place, but a rebuild elsewhere would "
            "produce the same pixels."
        )
    else:
        verdict = "IDENTICAL"
        meaning = "The two machines produce byte-identical JPEGs at every tested cell."

    log.info("comparison verdict: %s", verdict)
    log.info("%s", meaning)
    for entry in byte_differences:
        log.warning(
            "  bytes differ: %-28s %s vs %s (%d vs %d bytes)",
            entry["cell"],
            entry["this_sha256"][:12],
            entry["other_sha256"][:12],
            entry["this_bytes"],
            entry["other_bytes"],
        )
    for entry in pixel_differences:
        log.error("  PIXELS differ: %s", entry["cell"])
    for cell in missing:
        log.warning("  cell not in the baseline: %s", cell)

    return {
        "verdict": verdict,
        "meaning": meaning,
        "cells_compared": len(current["cells"]) - len(missing),
        "cells_missing_from_baseline": missing,
        "byte_differences": byte_differences,
        "pixel_differences": pixel_differences,
        "this_environment": current["environment"],
        "other_environment": baseline["environment"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-B2 JPEG encoder equivalence (I-010).")
    parser.add_argument(
        "--out", type=Path, default=THESIS_DBP / "reports" / "p1_jpeg_equivalence.json"
    )
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="An equivalence JSON from the other machine. Exits non-zero on any difference.",
    )
    args = parser.parse_args()

    log = setup_logging("p1_jpeg_equivalence", log_dir=THESIS_DBP / "outputs" / "logs")
    env = environment()
    log.info("Pillow %s, libjpeg %s, %s", env["pillow"], env["libjpeg_version"], env["platform"])

    report: dict[str, Any] = {
        "environment": env,
        "qualities": list(QUALITIES),
        "subsampling": {str(code): label for code, label in sorted(SUBSAMPLING_LABELS.items())},
        "image_sizes": [list(size) for size in IMAGE_SIZES],
        "round_trip": round_trip_check(log),
        "cells": run_grid(log),
    }

    exit_code = 0
    if args.compare_to is not None:
        if not args.compare_to.is_file():
            log.error("--compare-to file not found: %s", args.compare_to)
            return 2
        baseline = json.loads(args.compare_to.read_text(encoding="utf-8"))
        report["comparison"] = compare(report, baseline, log)
        exit_code = 0 if report["comparison"]["verdict"] == "IDENTICAL" else 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s (%d cells)", args.out, len(report["cells"]))

    if not report["round_trip"]["all_match"]:
        log.error("this machine cannot recover its own JPEG quality setting")
        return 2
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
