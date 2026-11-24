"""P5 training pairs (docs/P5_PREREGISTRATION.md section 2.4): encoded once, written, hashed.

For every train/validation clip and every bank member that reconstructed it:

1. **Format match** (all pairs). The reconstruction is clean PNG while its real was decoded
   from H.264, so the reconstruction is re-encoded with libx264 (preset medium) at the CRF
   whose bits-per-pixel is closest to its real SOURCE clip's (``bit_rate / (W x H x fps)``,
   from the Pexels manifest). x264 is byte-deterministic, so the CRF search is exact.
   Stored: ``fm.mp4``.
2. **The re-encode degradation** (TRAIN pairs only, exactly p = 0.80, user 2026-11-22/23):
   both clips of the pair -- the real (its PNG frames) and the format-matched
   reconstruction -- are re-encoded with one shared draw: H.264 with p = 0.7 else H.265,
   CRF uniform over 18-35 (integers). Stored: ``real_deg.mp4`` and ``recon_deg.mp4``.
   x265 is not byte-deterministic (user 2026-11-24, option a): each clip is encoded ONCE,
   written and hashed; the hashes in the record are the data.

The draw of a pair is ``default_rng(stable seed of "2026:p5-pairs:<sample_id>:<member>")``,
three numbers always consumed, so it does not depend on order, workers or resumes.
Validation pairs are never degraded: they are the V1-a held-out set. As a DIAGNOSTIC only
(user, 2026-11-24), every validation pair also gets a re-encoded copy with its own draw's
codec and CRF (``real_diag.mp4`` / ``recon_diag.mp4``); it is scored beside the plain one and
never trained on.

Resumable: one JSON line per pair in ``<reports>/p5_pairs.jsonl``; a pair whose record exists
and whose files still hash to it is skipped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.video import decode_clip, encode_clip, read_png_clip  # noqa: E402

#: docs/P5_PREREGISTRATION.md section 2.4 (user, 2026-11-22/23)
P_REENCODE = 0.80
H264_SHARE = 0.70
CRF_RANGE = (18, 35)                  # inclusive
FM_CODEC, FM_PRESET = "libx264", "medium"
SEED_PREFIX = "2026:p5-pairs:"
CRF_MIN, CRF_MAX = 0, 51


def pair_draw(sample_id: str, member: str) -> dict[str, Any]:
    """The pre-declared degradation draw of one pair (independent of order and workers)."""
    digest = hashlib.sha256(f"{SEED_PREFIX}{sample_id}:{member}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    u_apply, u_codec = (float(x) for x in rng.random(2))
    crf = int(rng.integers(CRF_RANGE[0], CRF_RANGE[1] + 1))
    return {"degrade": u_apply < P_REENCODE,
            "codec": "libx264" if u_codec < H264_SHARE else "libx265", "crf": crf}


def source_bpp(row: dict[str, str]) -> tuple[float, str]:
    """Bits per pixel per frame of the real's SOURCE video, and where it came from."""
    width, height = int(row["width"]), int(row["height"])
    fps = float(row["fps"])
    bit_rate = float(row.get("bit_rate") or 0)
    if bit_rate > 0:
        return bit_rate / (width * height * fps), "bit_rate"
    duration = float(row.get("duration_s") or 0)
    if duration > 0 and float(row.get("mp4_bytes") or 0) > 0:
        return float(row["mp4_bytes"]) * 8 / duration / (width * height * fps), "mp4_bytes"
    raise ValueError(f"{row['sample_id']}: no bit rate and no size/duration in the manifest")


def clip_bpp(data: bytes, frames: np.ndarray) -> float:
    """Bits per pixel per frame of an encoded clip."""
    t, h, w = frames.shape[:3]
    return len(data) * 8 / (t * h * w)


def format_match(frames: np.ndarray, target_bpp: float, fps: str) -> tuple[bytes, int, float]:
    """libx264 at the CRF whose bpp is closest to ``target_bpp`` (log distance). bpp falls
    as CRF rises, so a bisection finds the crossing; its two neighbours are compared."""
    cache: dict[int, bytes] = {}

    def enc(crf: int) -> bytes:
        if crf not in cache:
            cache[crf] = encode_clip(frames, FM_CODEC, crf, fps, FM_PRESET)
        return cache[crf]

    lo, hi = CRF_MIN, CRF_MAX
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if clip_bpp(enc(mid), frames) > target_bpp:
            lo = mid
        else:
            hi = mid
    best = min((lo, hi), key=lambda c: abs(math.log(clip_bpp(enc(c), frames) / target_bpp)))
    return enc(best), best, clip_bpp(enc(best), frames)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def build_one(job: dict[str, Any]) -> dict[str, Any]:
    """One pair: format-matched reconstruction, then (train only) the shared re-encode."""
    started = time.perf_counter()
    row, member, split = job["row"], job["member"], job["split"]
    out = Path(job["out_root"]) / member / row["sample_id"]
    real = read_png_clip(Path(job["reals_root"]) / row["clip_dir"])
    recon = read_png_clip(Path(job["recon_root"]) / member / row["sample_id"])
    if real.shape != recon.shape:
        raise ValueError(f"{row['sample_id']}/{member}: real {real.shape} != recon {recon.shape}")
    fps = row.get("fps_fraction") or row["fps"]
    target, basis = source_bpp(row)
    fm, crf, bpp = format_match(recon, target, fps)
    write_bytes(out / "fm.mp4", fm)
    record: dict[str, Any] = {
        "pair_id": f"{row['sample_id']}:{member}", "sample_id": row["sample_id"],
        "video_id": row["video_id"], "member": member, "split": split,
        "real_clip": row["clip_dir"], "frames": int(real.shape[0]), "fps": fps,
        "fm": {"path": f"{member}/{row['sample_id']}/fm.mp4", "sha256": sha256_bytes(fm),
               "crf": crf, "bpp": round(bpp, 6), "source_bpp": round(target, 6),
               "source_bpp_from": basis}}
    draw = pair_draw(row["sample_id"], member)
    record["draw"] = draw
    if split == "train" and draw["degrade"]:
        fm_frames = decode_clip(fm, expect_frames=len(recon))
        real_deg = encode_clip(real, draw["codec"], draw["crf"], fps)
        recon_deg = encode_clip(fm_frames, draw["codec"], draw["crf"], fps)
        for name, data in (("real_deg", real_deg), ("recon_deg", recon_deg)):
            decode_clip(data, expect_frames=len(real))        # the frame count is kept
            write_bytes(out / f"{name}.mp4", data)
            record[name] = {"path": f"{member}/{row['sample_id']}/{name}.mp4",
                            "sha256": sha256_bytes(data)}
    if split == "validation":            # the diagnostic copy, never trained on
        fm_frames = decode_clip(fm, expect_frames=len(recon))
        for name, frames in (("real_diag", real), ("recon_diag", fm_frames)):
            data = encode_clip(frames, draw["codec"], draw["crf"], fps)
            decode_clip(data, expect_frames=len(real))
            write_bytes(out / f"{name}.mp4", data)
            record[name] = {"path": f"{member}/{row['sample_id']}/{name}.mp4",
                            "sha256": sha256_bytes(data)}
    record["degraded"] = "real_deg" in record
    record["seconds"] = round(time.perf_counter() - started, 3)
    return record


def file_ok(root: Path, entry: dict[str, Any]) -> bool:
    path = root / entry["path"]
    return path.is_file() and sha256_bytes(path.read_bytes()) == entry["sha256"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits-table", type=Path, required=True,
                        help="reports/p5_train_recon/p5_train_splits.csv "
                             "(sample_id,video_id,split)")
    parser.add_argument("--members", required=True)
    parser.add_argument("--reals-root", type=Path, required=True)
    parser.add_argument("--recon-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports" / "p5_pairs")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        rows = {r["sample_id"]: r for r in csv.DictReader(handle) if r["status"] == "kept"}
    with args.splits_table.open(encoding="utf-8", newline="") as handle:
        split_of = {r["sample_id"]: r["split"] for r in csv.DictReader(handle)}
    members = args.members.split(",")
    args.reports.mkdir(parents=True, exist_ok=True)
    records_path = args.reports / "p5_pairs.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if records_path.is_file():
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if all(file_ok(args.out_root, rec[k]) for k in
                       ("fm", "real_deg", "recon_deg", "real_diag", "recon_diag") if k in rec):
                    done[rec["pair_id"]] = rec
    jobs, missing = [], []
    for sample_id, split in sorted(split_of.items()):
        if split not in ("train", "validation"):
            continue                              # threshold reals: never reconstructed
        for member in members:
            if f"{sample_id}:{member}" in done:
                continue
            if not (args.recon_root / member / sample_id).is_dir():
                missing.append(f"{sample_id}:{member}")
                continue
            jobs.append({"row": rows[sample_id], "member": member, "split": split,
                         "reals_root": str(args.reals_root), "recon_root": str(args.recon_root),
                         "out_root": str(args.out_root)})
    if missing:
        raise SystemExit(f"{len(missing)} reconstruction(s) missing, e.g. {missing[:5]}: the "
                         f"reconstruction is not complete for {members}")
    print(f"pairs: {len(done)} done, {len(jobs)} to build, {args.workers} workers", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool, records_path.open("a", encoding="utf-8") as out:
        futures = [pool.submit(build_one, job) for job in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            rec = future.result()
            out.write(json.dumps(rec) + "\n")
            out.flush()
            done[rec["pair_id"]] = rec
            if i % 500 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} pairs, {time.time() - started:.0f} s", flush=True)
    recs = list(done.values())
    summary = {"pairs": len(recs), "members": members,
               "by_split": {s: sum(r["split"] == s for r in recs) for s in ("train", "validation")},
               "train_degraded": sum(r["degraded"] for r in recs if r["split"] == "train"),
               "train_degraded_h265": sum(r["degraded"] and r["draw"]["codec"] == "libx265"
                                          for r in recs if r["split"] == "train"),
               "fm_crf": {"min": min(r["fm"]["crf"] for r in recs) if recs else None,
                          "max": max(r["fm"]["crf"] for r in recs) if recs else None},
               "seconds": round(time.time() - started, 1),
               "records_sha256": sha256_bytes(records_path.read_bytes())}
    (args.reports / "p5_pairs_summary.json").write_text(json.dumps(summary, indent=1) + "\n",
                                                        encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
