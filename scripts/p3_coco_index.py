"""P3 Phase B step 3a: sha256 + pHash of every COCO 2017 image, for the Open Images dedup.

Open Images and COCO are both drawn from Flickr, so the same photograph can be in both.
The Open Images draw must be disjoint from **all of COCO** (``docs/P3_PLAN.md`` §4a), not
only from the 2,500 T-real crops, so every ``train2017`` and ``val2017`` image is hashed
here: sha256 of the file and pHash of the whole image, with ``phash_file`` -- the
function the evaluation manifests used, so the hashes are comparable.

Also the descriptive count the plan asks for (not a gate): **E3 reals with a pHash ≤ 6
match in COCO train2017 + val2017**.

The index goes to ``--out`` (``/tmp`` on Kaggle; it is an input of the next step, not an
output of the run); the summary to ``reports/p3_coco_index_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

import numpy as np  # noqa: E402

from dbp.data.phash import (  # noqa: E402
    DUPLICATE_THRESHOLD_BITS,
    phash_file,
    phash_from_hex,
    phash_hex,
)
from dbp.eval.panel_b import hamming_to_all  # noqa: E402
from dbp.utils.disk import df_report  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

COLUMNS = ("split", "file", "sha256", "phash")
SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


def hash_one(path: Path) -> tuple[str, str]:
    """(sha256, pHash hex) of one file."""
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return sha, phash_hex(phash_file(path))


def main() -> int:
    """Hash COCO, write the index and the summary. Returns 0 on success."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--coco-root", required=True, type=Path,
                        help="The directory holding train2017/ and val2017/.")
    parser.add_argument("--splits", default="train2017,val2017")
    parser.add_argument("--expect", action="append", default=[], metavar="SPLIT=N",
                        help="Required image count per split (118287 / 5000 on Kaggle).")
    parser.add_argument("--out", required=True, type=Path, help="Index CSV (in /tmp).")
    parser.add_argument("--e3-manifest", type=Path, default=None,
                        help="E3 manifest: count its reals with a COCO pHash match.")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    args = parser.parse_args()
    log = setup_logging("p3_coco_index")
    disk = [df_report("coco index: start", [args.out.parent, args.reports], log)]

    expected = {k: int(v) for k, _, v in (e.partition("=") for e in args.expect)}
    files: list[tuple[str, Path]] = []
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        folder = args.coco_root / split
        found = sorted(p for p in folder.iterdir() if p.suffix.lower() in SUFFIXES) \
            if folder.is_dir() else []
        log.info("%s: %d image(s) in %s", split, len(found), folder)
        if split in expected and len(found) != expected[split]:
            log.error("COCO INDEX FAILED: %s holds %d image(s), expected %d", split,
                      len(found), expected[split])
            return 1
        files += [(split, p) for p in found]
    if not files:
        log.error("COCO INDEX FAILED: no image under %s", args.coco_root)
        return 1

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = list(pool.map(lambda item: hash_one(item[1]), files, chunksize=64))
    seconds = time.perf_counter() - start
    log.info("hashed %d image(s) in %.0f s (%d worker(s); %.1f ms per image)", len(files),
             seconds, args.workers, 1000 * seconds / len(files))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for (split, path), (sha, ph) in zip(files, hashes, strict=True):
            writer.writerow((split, path.name, sha, ph))
    os.replace(tmp, args.out)
    log.info("index -> %s", args.out)

    summary: dict = {
        "coco_root": str(args.coco_root),
        "counts": {s: sum(1 for x, _ in files if x == s) for s in {x for x, _ in files}},
        "index_rows": len(files),
        "seconds": round(seconds, 1),
        "workers": args.workers,
        "ms_per_image_wall": round(1000 * seconds / len(files), 2),
        "phash_function": "dbp.data.phash.phash_file (whole image), as in the eval manifests",
    }
    if args.e3_manifest:
        with args.e3_manifest.open(encoding="utf-8") as handle:
            reals = [r for r in csv.DictReader(handle) if r["label"] == "real"]
        coco = np.array([phash_from_hex(ph) for _, ph in hashes], dtype=np.uint64)
        hit = sum(1 for r in reals
                  if (hamming_to_all(phash_from_hex(r["phash"]), coco)
                      <= DUPLICATE_THRESHOLD_BITS).any())
        summary["e3_reals_with_coco_phash_match"] = {
            "e3_reals": len(reals), "matched": hit, "threshold_bits": DUPLICATE_THRESHOLD_BITS,
            "note": "descriptive only, not a gate (P3_PLAN.md section 4a)"}
        log.info("E3 reals with a pHash <= %d match in COCO: %d of %d (descriptive)",
                 DUPLICATE_THRESHOLD_BITS, hit, len(reals))
    disk.append(df_report("coco index: end", [args.out.parent, args.reports], log))
    summary["disk"] = disk
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / "p3_coco_index_summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("summary -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
