"""P5: Pexels real clips -- shard plan, download, pre-declared clip protocol, PNG clips.

Source (approved A1, 2026-11-21): ``zengxianyu/open-sora-pexels-subset`` at the pinned
commit below; treated as the **Pexels License: research use, private, no
redistribution** (the Hub's ``apache-2.0`` tag is recorded beside it, it may be
inaccurate). Downloads happen on Kaggle (the pilot pool) or AWS (the training pool),
never on the local machine.

**Pools are disjoint by construction.** The shard names are ordered by
``sha256("2026:<name>")`` -- the plan is written to ``p5_reals_shard_plan.json`` -- and
each pool owns a fixed slice of that order (``--shards A:B``): the pilot pool is
``0:4``, training is ``4:35``. Video ids are also checked across pools from the manifest.

**Per video** (``dbp.data.video``): probe; the pre-declared exclusions; the window of
``frames`` native frames from :func:`window_start`; centre crop; static check; PNG clip.
Then every eligible clip is ranked by ``sha256("2026:<video_id>")`` and the first ``--n``
are kept (one clip per video id); the rest are removed. The manifest records every video,
kept, spare or excluded, with the reason.

Usage (Kaggle pilot):
    python scripts/p5_fetch_reals.py --source hub --pool pilot --shards 0:4 --n 200 \\
        --out-root /kaggle/working/p5_reals --cache /tmp/p5_shards
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import tarfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.video import (  # noqa: E402
    ShortClipError,
    VideoError,
    centre_crop_clip,
    decode_window,
    exclusion,
    is_static,
    probe,
    window_start,
    write_png_clip,
)
from dbp.decoders.registry import load_video_bank  # noqa: E402

#: The source, pinned (HfApi.dataset_info, 2026-11-20; docs/P5_PLAN.md section 1.1).
PEXELS_REPO = "zengxianyu/open-sora-pexels-subset"
PEXELS_REVISION = "c595493c834ec2c47ceee5d18ecca002aa646540"
PEXELS_LICENCE = ("Pexels License (research use, private, no redistribution; user 2026-11-21); "
                  "Hub tag apache-2.0 recorded, may be inaccurate")
SEED = 2026
COLUMNS = ["sample_id", "video_id", "shard", "member", "mp4_sha256", "mp4_bytes", "codec",
           "profile", "pix_fmt", "width", "height", "fps", "fps_fraction", "frames_reported",
           "duration_s", "bit_rate", "rotation", "start", "status", "exclusion", "rank",
           "clip_dir", "clip_digest", "png_bytes", "detail"]


def rank_key(text: str) -> str:
    """The pre-declared order: ``sha256("2026:<text>")``."""
    return hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()


def shard_plan(names: list[str]) -> list[str]:
    """Every shard name, in the pre-declared order."""
    return sorted(names, key=rank_key)


def read_manifest(path: Path) -> list[dict[str, str]]:
    """The source manifest: parquet (pyarrow, preinstalled on Kaggle) or a CSV of the same
    columns (the local dry-run fixture, which has no pyarrow)."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return [{k: str(v) for k, v in row.items()} for row in pq.read_table(path).to_pylist()]
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class Source:
    """The shards and manifest, from the Hub (pinned) or a local directory (dry-run)."""

    def __init__(self, spec: str, cache: Path) -> None:
        self.spec, self.cache = spec, cache
        self.hub = spec == "hub"
        self.sizes: dict[str, int] = {}
        self.lfs: dict[str, str] = {}
        if self.hub:
            from huggingface_hub import HfApi

            info = HfApi().dataset_info(PEXELS_REPO, revision=PEXELS_REVISION,
                                        files_metadata=True)
            for s in info.siblings or []:
                self.sizes[s.rfilename] = int(s.size or 0)
                if s.lfs is not None:
                    self.lfs[s.rfilename] = s.lfs.sha256
        else:
            self.root = Path(spec)
            for p in self.root.rglob("*"):
                if p.is_file():
                    self.sizes[p.relative_to(self.root).as_posix()] = p.stat().st_size

    def shards(self) -> list[str]:
        return sorted(n for n in self.sizes if n.startswith("videos/") and n.endswith(".tar"))

    def manifest_name(self) -> str:
        for name in ("manifest.parquet", "manifest.csv"):
            if name in self.sizes:
                return name
        raise VideoError(f"{self.spec}: no manifest.parquet / manifest.csv")

    def fetch(self, name: str) -> Path:
        """A local copy of ``name``; for a Hub shard, its sha256 checked against the LFS
        record."""
        if not self.hub:
            return self.root / name
        from huggingface_hub import hf_hub_download

        path = Path(hf_hub_download(PEXELS_REPO, name, repo_type="dataset",
                                    revision=PEXELS_REVISION, cache_dir=str(self.cache)))
        if name in self.lfs:
            digest = sha256_file(path)
            if digest != self.lfs[name]:
                raise VideoError(f"{name}: sha256 {digest} != Hub LFS {self.lfs[name]}")
        return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_video(task: tuple[str, str, str, str, str, int, int, int]) -> dict[str, Any]:
    """One video: probe, exclusions, window, crop, static check, PNG clip. Never raises
    for a bad video -- the reason is the row's ``exclusion``."""
    mp4, sample_id, video_id, shard, member, frames, crop, min_side = task
    path, out = Path(mp4), Path(mp4).with_suffix("")
    row: dict[str, Any] = {"sample_id": sample_id, "video_id": video_id, "shard": shard,
                           "member": member, "mp4_sha256": sha256_file(path),
                           "mp4_bytes": path.stat().st_size, "status": "excluded"}
    try:
        info = probe(path)
    except VideoError as exc:
        return {**row, "exclusion": "decode_error", "detail": str(exc)[:200]}
    row.update({k: info.get(k) for k in COLUMNS if k in info})
    reason = exclusion(info, min_side)
    if reason:
        return {**row, "exclusion": reason}
    row["start"] = window_start(video_id, int(info["frames_reported"]), frames, SEED)
    try:
        clip = centre_crop_clip(decode_window(path, row["start"], frames), crop)
    except ShortClipError:
        return {**row, "exclusion": "too_few_frames"}
    except VideoError as exc:
        return {**row, "exclusion": "decode_error", "detail": str(exc)[:200]}
    if is_static(clip):
        return {**row, "exclusion": "static"}
    written = write_png_clip(clip, out)
    return {**row, "status": "eligible", "exclusion": "", "clip_dir": str(out),
            "clip_digest": written["digest"], "png_bytes": written["bytes"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", required=True, help="'hub' or a local directory")
    parser.add_argument("--pool", required=True, choices=["pilot", "train"])
    parser.add_argument("--shards", required=True, help="A:B, a slice of the shard plan")
    parser.add_argument("--n", type=int, required=True, help="clips to keep")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="shard download cache")
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports" / "p5_reals")
    parser.add_argument("--exclude-ids", type=Path, action="append", default=[],
                        help="a manifest CSV whose video ids no clip here may share")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--skip-pins", action="store_true",
                        help="dry-run only: the pin check needs the network")
    args = parser.parse_args()
    started = time.time()
    bank = load_video_bank(THESIS_DBP / "configs" / "video_bank.yaml")
    if not args.skip_pins:
        from p0_verify_pins import require_video_pins

        require_video_pins()   # CLAUDE.md 7a: before any data is written
    first, last = (int(x) for x in args.shards.split(":"))
    source = Source(args.source, args.cache)
    plan = shard_plan(source.shards())
    if not 0 <= first < last <= len(plan):
        raise SystemExit(f"--shards {args.shards} is outside the plan of {len(plan)} shards")
    mine = plan[first:last]
    manifest = read_manifest(source.fetch(source.manifest_name()))
    by_member = {Path(r.get("shard", "")).name + "/" + r["sample_id"]: r for r in manifest}
    excluded_ids = set()
    for other in args.exclude_ids:
        with other.open(encoding="utf-8", newline="") as handle:
            excluded_ids |= {r["video_id"] for r in csv.DictReader(handle)}
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "p5_reals_shard_plan.json").write_text(json.dumps(
        {"repo": PEXELS_REPO, "revision": PEXELS_REVISION, "licence": PEXELS_LICENCE,
         "order": "sha256('2026:<shard name>')", "plan": plan,
         "pools": {"pilot": "0:4", "train": "4:35"}}, indent=2) + "\n", encoding="utf-8")

    pool_dir = args.out_root / args.pool
    pool_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for shard in mine:
        tar_path = source.fetch(shard)
        work = args.cache / "unpacked" / Path(shard).stem
        if work.exists():
            shutil.rmtree(work)
        work.mkdir(parents=True)
        tasks = []
        with tarfile.open(tar_path) as archive:
            for info in archive.getmembers():
                if not info.isfile() or not info.name.endswith(".mp4"):
                    continue
                sample_id = Path(info.name).stem
                meta = by_member.get(f"{Path(shard).name}/{sample_id}")
                if meta is None:
                    raise SystemExit(f"{shard}:{info.name} is not in the manifest")
                if meta["video_id"] in excluded_ids:
                    rows.append({"sample_id": sample_id, "video_id": meta["video_id"],
                                 "shard": shard, "member": info.name, "status": "excluded",
                                 "exclusion": "other_pool"})
                    continue
                target = work / f"{sample_id}.mp4"
                with archive.extractfile(info) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                tasks.append((str(target), sample_id, meta["video_id"], shard, info.name,
                              bank.frames, bank.crop, bank.min_short_side))
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows.extend(pool.map(process_video, tasks))
        for task in tasks:
            Path(task[0]).unlink(missing_ok=True)
        if source.hub and not args.keep_shards:
            Path(tar_path).unlink(missing_ok=True)
        print(f"{shard}: {len(tasks)} videos, "
              f"{sum(r['status'] == 'eligible' for r in rows)} eligible so far", flush=True)

    # one clip per video id, then the pre-declared rank; the first --n are kept
    eligible = sorted((r for r in rows if r["status"] == "eligible"),
                      key=lambda r: rank_key(r["video_id"]))
    seen: set[str] = set()
    kept = 0
    for r in eligible:
        r["rank"] = rank_key(r["video_id"])[:16]
        if r["video_id"] in seen:
            r["status"], r["exclusion"] = "excluded", "duplicate_video_id"
        elif kept < args.n:
            r["status"], kept = "kept", kept + 1
            final = pool_dir / r["sample_id"]
            Path(r["clip_dir"]).replace(final)
            r["clip_dir"] = f"{args.pool}/{r['sample_id']}"
        else:
            r["status"] = "spare"
        seen.add(r["video_id"])
        if r["status"] != "kept" and r.get("clip_dir") and Path(r["clip_dir"]).exists():
            shutil.rmtree(r["clip_dir"])
            r["clip_dir"] = ""
    if (args.cache / "unpacked").exists():
        shutil.rmtree(args.cache / "unpacked")
    manifest_path = args.reports / f"p5_reals_{args.pool}_manifest.csv"
    tmp = manifest_path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["shard"], r["sample_id"])))
    tmp.replace(manifest_path)
    status = Counter(r["status"] for r in rows)
    summary = {
        "pool": args.pool, "shards": mine, "source": PEXELS_REPO, "revision": PEXELS_REVISION,
        "licence": PEXELS_LICENCE, "clip": {"frames": bank.frames, "crop": bank.crop,
                                            "min_short_side": bank.min_short_side},
        "videos": len(rows), "status": dict(status),
        "exclusions": dict(Counter(r["exclusion"] for r in rows if r.get("exclusion"))),
        "kept": kept, "requested": args.n, "complete": kept == args.n,
        "fps_kept": dict(Counter(str(r.get("fps_fraction")) for r in rows
                                 if r["status"] == "kept")),
        "png_bytes_kept": sum(int(r.get("png_bytes") or 0) for r in rows if r["status"] == "kept"),
        "manifest": manifest_path.name, "manifest_sha256": sha256_file(manifest_path),
        "seconds": round(time.time() - started, 1),
    }
    (args.reports / f"p5_reals_{args.pool}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("pool", "videos", "status", "exclusions", "kept",
                                               "complete", "seconds")}))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
