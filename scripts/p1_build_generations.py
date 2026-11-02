"""P1-B6: generate the SD1.5 fake-image set.

The pairs from B5 teach a detector what a decoder does to a *real* image. They cannot
teach it what a fully generated image looks like, because a reconstruction still
carries the original photograph's content and noise. This set supplies the other half:
images that never were a photograph.

Decisions worth stating:

**Same checkpoint as the probe.** The pipeline is loaded from the ``sd15`` bank
member's repo at the same pinned revision, so the generator and the bank member are
provably the same weights and no second pin enters the project.

**Prompts come from the real corpus, not from a prompt list.** COCO captions describe
the same distribution of scenes as the COCO reals, so the fake and real sets differ in
provenance rather than in subject matter. A detector trained against fakes of a
different subject distribution learns the subject, not the generator.

**Seeded per image, from the image id.** Reproducible, and independent of the order
images are generated in, so a resumed run produces exactly the images an uninterrupted
one would.

**The safety checker stays enabled.** Flagged outputs are discarded and counted rather
than written; the count goes in the report, so the set size is explained.

Usage:
    python scripts/p1_build_generations.py --captions
        /kaggle/input/coco-2017-dataset/annotations/captions_train2017.json --limit 2500
"""

from __future__ import annotations

import argparse
import hashlib
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

from dbp.data.jpeg import jpeg_metadata  # noqa: E402
from dbp.data.manifest import (  # noqa: E402
    BASE_COLUMNS,
    append_rows,
    completed_ids,
    file_sha256,
    read_manifest,
    write_manifest,
)
from dbp.data.phash import find_duplicate_groups, phash_from_array, phash_hex  # noqa: E402
from dbp.utils.io import center_crop, save_png_uint8  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from dbp.utils.seed import seed_everything  # noqa: E402
from p0_verify_pins import PinVerificationError, require_pins  # noqa: E402

MANIFEST_COLUMNS = (
    *BASE_COLUMNS,
    "generator",
    "revision",
    "prompt_sha256",
    "prompt",
    "seed",
    "steps",
    "guidance_scale",
    "native_size",
    "crop",
    "phash",
)

#: SD1.5's native training resolution. Generating at anything else changes the
#: artefacts, so this is fixed rather than exposed as a convenience flag.
NATIVE_SIZE = 512

CHUNK = 50


def load_captions(path: Path, limit: int, log: Any) -> list[tuple[str, str]]:
    """Read COCO captions and return ``(caption_id, text)``, deterministically ordered.

    One caption per image: COCO carries about five per image, and using several would
    give one scene several near-identical generations.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    annotations = payload.get("annotations")
    if not annotations:
        raise ValueError(f"{path} has no 'annotations'; is this a COCO captions file?")

    first_per_image: dict[int, dict[str, Any]] = {}
    for record in annotations:
        image_id = int(record["image_id"])
        if image_id not in first_per_image or record["id"] < first_per_image[image_id]["id"]:
            first_per_image[image_id] = record

    ordered = sorted(first_per_image.values(), key=lambda r: int(r["image_id"]))
    captions = [(f"coco_{int(r['image_id']):012d}", str(r["caption"]).strip()) for r in ordered]
    captions = [(cid, text) for cid, text in captions if text]
    log.info("%d captions available, taking %d", len(captions), min(limit, len(captions)))
    return captions[:limit]


def seed_for(caption_id: str) -> int:
    """A per-image seed derived from the caption id, stable across processes."""
    digest = hashlib.blake2b(caption_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31 - 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="P1-B6 SD1.5 generation set.")
    parser.add_argument("--captions", type=Path, required=True, help="COCO captions JSON.")
    parser.add_argument("--limit", type=int, default=2500)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--crop", type=int, default=384, help="Centre-crop after generation.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out-root", type=Path, default=THESIS_DBP / "outputs" / "p1" / "generated"
    )
    parser.add_argument("--set-id", default="g_sd15")
    parser.add_argument("--skip-pin-check", action="store_true", help="Diagnostics only.")
    args = parser.parse_args()

    log = setup_logging("p1_build_generations", log_dir=THESIS_DBP / "outputs" / "logs")
    log.info("seed: %s", seed_everything(2026, deterministic=True))

    if args.crop > NATIVE_SIZE:
        log.error("--crop %d exceeds the native size %d; cropping cannot enlarge (D-001)",
                  args.crop, NATIVE_SIZE)
        return 2
    if not args.captions.is_file():
        log.error("captions file not found: %s", args.captions)
        return 2

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

    from diffusers import StableDiffusionPipeline

    from dbp.decoders.registry import load_bank_config
    from dbp.utils.hf import get_hf_token

    spec = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml").by_key("sd15")
    log.info("generator: %s @ %s (the sd15 bank member's own pin)", spec.repo, spec.revision)

    pipeline = StableDiffusionPipeline.from_pretrained(
        spec.repo,
        revision=spec.revision,
        use_safetensors=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        token=get_hf_token(),
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)
    safety_checker_state = "enabled" if pipeline.safety_checker is not None else "absent"
    log.info("safety checker: %s", safety_checker_state)

    out_dir = args.out_root / args.set_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = THESIS_DBP / "reports" / "p1_gen_manifest.csv"
    done = completed_ids(manifest_path)
    if done:
        log.info("resuming: %d images already generated", len(done))

    captions = load_captions(args.captions, args.limit, log)
    chunk_rows: list[dict[str, Any]] = []
    written = 0
    flagged = 0
    failures: list[dict[str, str]] = []
    started = time.perf_counter()

    for index, (caption_id, text) in enumerate(captions, start=1):
        if caption_id in done:
            continue
        seed = seed_for(caption_id)
        generator = torch.Generator(device=device).manual_seed(seed)
        try:
            result = pipeline(
                text,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=NATIVE_SIZE,
                width=NATIVE_SIZE,
                generator=generator,
                output_type="np",
            )
        except Exception as exc:
            failures.append(
                {"caption_id": caption_id, "error": f"{type(exc).__name__}: {exc}"[:200]}
            )
            log.error("generation failed for %s: %s", caption_id, exc)
            continue

        nsfw = getattr(result, "nsfw_content_detected", None)
        if nsfw and bool(nsfw[0]):
            flagged += 1
            log.info("safety checker flagged %s; discarded", caption_id)
            continue

        # output_type="np" gives float in [0, 1]; quantise exactly as the protocol does.
        image = np.clip(np.asarray(result.images[0], dtype=np.float32), 0.0, 1.0)
        image_uint8 = (image * 255.0).round().astype(np.uint8)
        cropped = center_crop(image_uint8, args.crop)

        destination = out_dir / f"{caption_id}.png"
        save_png_uint8(cropped, destination)
        metadata = jpeg_metadata(destination)

        chunk_rows.append(
            {
                "image_id": caption_id,
                "set_id": args.set_id,
                "label": "fake",
                "relative_path": destination.relative_to(args.out_root.parent).as_posix(),
                "width": args.crop,
                "height": args.crop,
                "format": metadata["format"],
                "is_jpeg": metadata["is_jpeg"],
                "jpeg_quality": metadata["jpeg_quality"],
                "subsampling": metadata["subsampling"],
                "sha256": file_sha256(destination),
                "generator": "SD1.5",
                "revision": spec.revision,
                "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "prompt": text.replace("\n", " ")[:300],
                "seed": seed,
                "steps": args.steps,
                "guidance_scale": args.guidance_scale,
                "native_size": NATIVE_SIZE,
                "crop": args.crop,
                "phash": phash_hex(phash_from_array(cropped)),
            }
        )

        if len(chunk_rows) >= CHUNK or index == len(captions):
            append_rows(chunk_rows, manifest_path, MANIFEST_COLUMNS)
            written += len(chunk_rows)
            rate = written / max(time.perf_counter() - started, 1e-9)
            log.info(
                "%d written (%.3f img/s, %.1f min remaining)",
                written,
                rate,
                (len(captions) - len(done) - written) / max(rate, 1e-9) / 60.0,
            )
            chunk_rows = []

    if chunk_rows:
        append_rows(chunk_rows, manifest_path, MANIFEST_COLUMNS)
        written += len(chunk_rows)

    rows = read_manifest(manifest_path) if manifest_path.is_file() else []
    hashes = {row["image_id"]: int(row["phash"], 16) for row in rows if row.get("phash")}
    duplicate_groups = find_duplicate_groups(hashes)
    if duplicate_groups:
        log.warning(
            "%d near-duplicate generation group(s); recorded, not dropped -- duplicates "
            "among generations are a property of the generator and belong in the report",
            len(duplicate_groups),
        )

    # Rewrite the manifest atomically once the run is complete, so the final artefact
    # is a single well-formed file rather than a sequence of appends.
    write_manifest(rows, manifest_path, MANIFEST_COLUMNS)

    summary = {
        "set_id": args.set_id,
        "generator": "SD1.5",
        "repo": spec.repo,
        "revision": spec.revision,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "native_size": NATIVE_SIZE,
        "crop": args.crop,
        "requested": args.limit,
        "written_this_run": written,
        "total_in_manifest": len(rows),
        "safety_checker": safety_checker_state,
        "safety_checker_flagged_and_discarded": flagged,
        "failures": failures,
        "near_duplicate_groups": duplicate_groups,
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.is_file() else None,
    }
    summary_path = THESIS_DBP / "reports" / "p1_gen_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("wrote %s and %s", manifest_path, summary_path)
    log.info(
        "generated %d, flagged and discarded %d, failed %d",
        written,
        flagged,
        len(failures),
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
