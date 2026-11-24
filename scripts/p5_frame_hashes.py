"""P5 training-side frame hashes for the panel leak checks (prereg section 2.5; user 2026-11-24).

Run on the AWS instance in the pairs session (CPU): the pHash (``dbp.data.phash``) of EVERY
frame of every kept Pexels clip of the training pool -- train, validation and the
threshold-held-out reals -- as stored (25 frames, centre crop 320). Only the hashes leave
the instance (a small report); the clips never do. On Kaggle, the panel's keyframe hashes and
each panel fake's first frame are checked against them (sha256 / pHash <= 6 bits); a match is
recorded and reported separately (P3's rule), never silently dropped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))


def hash_clip(job: tuple[str, str, str, str]) -> dict[str, object]:
    """Every frame's pHash (hex) of one stored clip."""
    from dbp.data.phash import phash_from_array, phash_hex
    from dbp.data.video import read_png_clip

    sample_id, video_id, split, clip_dir = job
    frames = read_png_clip(Path(clip_dir))
    return {"sample_id": sample_id, "video_id": video_id, "split": split,
            "phash": [phash_hex(phash_from_array(f)) for f in frames]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-table", type=Path, required=True)
    parser.add_argument("--reals-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="p5_train_frame_hashes.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with args.splits_table.open(encoding="utf-8", newline="") as handle:
        split_of = {r["sample_id"]: r["split"] for r in csv.DictReader(handle)}
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        jobs = [(r["sample_id"], r["video_id"], split_of.get(r["sample_id"]),
                 str(args.reals_root / r["clip_dir"]))
                for r in csv.DictReader(handle) if r["status"] == "kept"]
    missing = [j[0] for j in jobs if j[2] is None]
    if missing:
        raise SystemExit(f"{len(missing)} kept clips have no split, e.g. {missing[:3]}")
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(hash_clip, jobs, chunksize=16))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(args.out.name + ".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(json.dumps({"clips": len(rows), "frames": sum(len(r["phash"]) for r in rows),
                      "splits": {s: sum(r["split"] == s for r in rows)
                                 for s in ("train", "validation", "threshold")},
                      "sha256": digest}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
