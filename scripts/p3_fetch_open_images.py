"""P3 Phase B step 3b: draw the Open Images reals from the Flickr originals.

The user's rules (2026-10-11, ``docs/P3_PLAN.md`` §4a) and how each is met:

* **(a)** Only ``OriginalURL``, only from an approved host (``--hosts``; default
  ``farm1-8`` and ``c1-c8.staticflickr.com``, approved 2026-10-11 and 2026-10-13),
  redirects only to an approved host or ``live.staticflickr.com``. A row on any other
  host is **not contacted**; it is rejected, counted and its host listed. Every row is
  walked in one seed order whatever its host; the summary reports each host group
  (``farm*`` / ``c*``) separately, descriptively. At most 8 HTTP connections are open at
  once; retries back off exponentially, and repeated 429/503 slow every request down
  (and, if they persist, stop the walk) -- ``dbp.data.open_images.Fetcher`` / ``Throttle``.
* **(b)** Every file is checked against ``OriginalMD5``; a mismatch is a re-encode,
  rejected and counted.
* **(c)** All CC BY 2.0 rows in one seed-2026 order; walk it, keep the first ``--target``
  that pass download, checksum, crop eligibility and dedup. Batches are downloaded in
  parallel but **decided strictly in walk order**, so the result does not depend on
  which download finished first. Dedup is sha256 and pHash ≤ 6, of the whole original
  **and** of the crop, against Panel B, the P2 panel, the training corpus (T-real, T-ctrl,
  the SD1.5 generations, their matched JPEGs and the bank reconstructions), all of COCO
  and every real of E3, E4 and E5. Each rejection names the reference set it matched.
  A seeded split (separate key) then holds out ``--holdout``.
* **(d)** Never resized: ``p1_build_reals.build_rows`` -- the function that made T-real
  -- centre-crops at native resolution to the COCO working size (384) and writes PNG.
  Rule (c)'s "≥ 320" is therefore met as "≥ 384": a 320-383 px image cannot give a 384
  crop without resizing; that band is counted separately. Greyscale originals are
  rejected, as ``build_rows`` does for COCO (D-039). Width/height distributions and the
  count of files with an EXIF camera make or model are logged (counts only).
* **(e)** Originals go to ``--tmp`` and each is deleted once processed; only the crops
  (``<out-root>/reals/t_oi/``) and the reports are output. ``df -h`` is logged at every
  stage, and the run stops **before its first download** if free space is short.
* **(f)** The attribution file (image id, author, original URL, licence) is written to
  ``reports/p3_open_images_attribution.csv``, to be committed under ``docs/``.

Outputs (``--reports``): ``p3_open_images_manifest.csv``, ``p3_open_images_attribution.csv``,
``p3_open_images_walk.csv`` (every walked row and its outcome) and
``p3_open_images_summary.json``. Exit 0 only if the target was met.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from PIL import Image  # noqa: E402

from dbp.data.open_images import (  # noqa: E402
    APPROVED_HOST_PATTERNS,
    MAX_PARALLEL,
    REDIRECT_HOSTS,
    S3_HOST_PATTERNS,
    SOURCE_COPY,
    Fetcher,
    LocalFetcher,
    Throttle,
    assign_split,
    colour_mode_reason,
    host_approved,
    host_group,
    host_of,
    load_metadata,
    order_key,
    s3_url,
)
from dbp.data.phash import phash_file, phash_hex  # noqa: E402
from dbp.eval.panel_b import HashIndex  # noqa: E402
from dbp.utils.disk import df_report, free_gb, require_free  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p1_build_reals import DEFAULT_CROP, MANIFEST_COLUMNS, build_rows  # noqa: E402

SET_ID = "t_oi"
EXIF_MAKE, EXIF_MODEL = 271, 272
EXTRA_COLUMNS = ("oi_image_id", "original_url", "original_md5", "original_size",
                 "downloaded_bytes", "rotation", "exif_camera", "source_phash", "order_key",
                 "split", "fetch_url", "source_copy", "md5_checked")
ATTRIBUTION_COLUMNS = ("image_id", "author", "original_url", "licence")
SHORT_SIDE_BINS = ((0, 320), (320, 384), (384, 1024), (1024, 2048), (2048, 3072),
                   (3072, 10**9))


def read_rows(path: Path) -> list[dict[str, str]]:
    """A CSV as a list of dicts."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_references(args: argparse.Namespace, log: Any) -> tuple[HashIndex, dict[str, int]]:
    """Every reference image the draw must be disjoint from, tagged by its set."""
    index = HashIndex()
    counts: dict[str, int] = {}
    counts["Panel B"] = index.add_rows("Panel B", read_rows(args.panel_b_selection))
    for path in sorted(args.p2_panel_sets.glob("p2_panel_*.csv")):
        view = path.stem.removeprefix("p2_panel_")
        counts[f"P2 panel {view}"] = index.add_rows(f"P2 panel {view}", read_rows(path))
    for name, path in (("T-real", args.t_real), ("T-ctrl", args.t_ctrl)):
        rows = read_rows(path)
        counts[name] = index.add_rows(name, rows)
        index.add_rows(f"{name} source", rows, sha_column="source_sha256", phash_column=None)
    counts["SD1.5 generations"] = index.add_rows("SD1.5 generations",
                                                 read_rows(args.gen_manifest))
    counts["gen_matched"] = index.add_rows("gen_matched", read_rows(args.gen_matched),
                                           sha_column="jpg_sha256", phash_column=None)
    counts["bank reconstructions"] = index.add_rows(
        "bank reconstructions", read_rows(args.pairs_manifest), phash_column=None)
    for item in args.eval_manifest:
        name, _, path = item.partition("=")
        reals = [r for r in read_rows(Path(path)) if r["label"] == "real"]
        counts[f"{name} reals"] = index.add_rows(f"{name} reals", reals)
    coco = read_rows(args.coco_index)
    for split in sorted({r["split"] for r in coco}):
        counts[f"COCO {split}"] = index.add_rows(f"COCO {split}",
                                                 [r for r in coco if r["split"] == split])
    for name, n in counts.items():
        log.info("reference %-34s %7d", name, n)
    return index, counts


def fetch_url(row: dict[str, str], source: str) -> str:
    """Where a row is downloaded from: the Flickr original or the CVDF S3 copy."""
    return s3_url(row["ImageID"]) if source == "s3" else row["OriginalURL"]


def process(row: dict[str, str], fetcher: Fetcher, download_dir: Path, crop_dir: Path,
            crop: int, keep_original: bool = False, source: str = "flickr") -> dict[str, Any]:
    """Download, verify and crop one row. Runs in a worker; decides nothing about dedup.

    The original is deleted once processed, unless ``keep_original`` and the row passed
    every check (the local route keeps candidates' originals in ``kaggle_raw``).
    """
    image_id = row["ImageID"]
    url = fetch_url(row, source)
    suffix = Path(url).suffix.lower() or ".jpg"
    original = download_dir / f"{image_id}{suffix}"
    result = fetcher.fetch(url, original)
    out: dict[str, Any] = {"outcome": result.status, "detail": result.detail,
                           "bytes": result.size, "final_host": result.final_host,
                           "attempts": result.attempts}
    if result.status != "ok":
        original.unlink(missing_ok=True)
        return out
    try:
        # OriginalMD5 describes the Flickr original; the CVDF copy is rescaled, so its own
        # sha256 is recorded instead (user decision 2026-10-15).
        if source == "flickr" and result.md5_b64 != row["OriginalMD5"]:
            out.update(outcome="md5", detail="checksum mismatch (re-encode served)")
            return out
        with Image.open(original) as image:
            out["width"], out["height"] = image.size
            out["mode"] = image.mode
            exif = image.getexif()
            out["exif_camera"] = bool(exif.get(EXIF_MAKE) or exif.get(EXIF_MODEL))
            mode_reason = colour_mode_reason(image.mode, image.info)
        if mode_reason:                         # never converted (user rule 2026-10-14)
            out.update(outcome="skip", detail=f"mode:{mode_reason}")
            return out
        # Decoded exactly as P1 decoded COCO (p1_build_reals -> load_rgb_uint8): EXIF
        # orientation is NOT applied and ICC profiles are NOT applied.
        rows, skip = build_rows(original, download_dir, SET_ID, "real", crop, crop_dir)
        if skip:
            out.update(outcome="skip", detail=skip)
            return out
        out.update(outcome="ok", sha256=result.sha256, crop_row=rows[0],
                   source_phash=phash_hex(phash_file(original)))
        return out
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
        out.update(outcome="undecodable", detail=f"{type(exc).__name__}: {str(exc)[:80]}")
        return out
    finally:
        if not (keep_original and out["outcome"] == "ok"):
            original.unlink(missing_ok=True)


def reason_for(result: dict[str, Any], crop: int) -> str:
    """The logged rejection reason for a row that did not reach the dedup."""
    outcome, detail = result["outcome"], result.get("detail", "")
    if outcome == "host":
        return f"redirect to an unapproved host ({detail}; not followed)"
    if outcome == "gone":
        return f"download: gone ({detail})"
    if outcome == "failed":
        return f"download failed ({detail.split(' after ')[0]})"
    if outcome == "md5":
        return "checksum mismatch (re-encode served)"
    if outcome == "undecodable":
        return "undecodable"
    if detail == "too_small":
        short = min(result.get("width", 0), result.get("height", 0))
        return ("shorter side < 320" if short < 320
                else f"shorter side 320-{crop - 1} (below the {crop} COCO crop)")
    if detail.startswith("mode:"):
        return f"colour mode: {detail.split(':', 1)[1]} (rejected, not converted)"
    if detail.startswith("greyscale:"):
        return f"greyscale ({detail.split(':', 1)[1]}; COCO pipeline rule D-039)"
    return f"skipped ({detail})"


def distribution(values: list[float]) -> dict[str, float]:
    """Min, quartiles, 5th/95th percentiles and max."""
    if not values:
        return {}
    ordered = sorted(values)
    q = statistics.quantiles(ordered, n=20) if len(ordered) > 1 else [ordered[0]] * 19
    return {"n": len(ordered), "min": ordered[0], "p5": q[0], "p25": q[4],
            "median": statistics.median(ordered), "p75": q[14], "p95": q[18],
            "max": ordered[-1]}


def size_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Width/height distributions and the shorter-side histogram (counts only)."""
    shorts = [min(r["width"], r["height"]) for r in records]
    return {
        "width": distribution([r["width"] for r in records]),
        "height": distribution([r["height"] for r in records]),
        "megapixels": distribution([round(r["width"] * r["height"] / 1e6, 2) for r in records]),
        "shorter_side_histogram": {f"{lo}-{hi - 1 if hi < 10**9 else 'max'}":
                                   sum(1 for s in shorts if lo <= s < hi)
                                   for lo, hi in SHORT_SIDE_BINS},
        "exif_camera_make_or_model": sum(1 for r in records if r.get("exif_camera")),
    }


def write_csv(path: Path, columns: tuple[str, ...] | list[str], rows: list[dict]) -> None:
    """Atomic CSV write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def main() -> int:  # noqa: PLR0915 - one linear walk, kept in one place on purpose
    """Draw the Open Images reals. Returns 0 only if the target was met."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--tmp", required=True, type=Path, help="Download dir (/tmp/...).")
    parser.add_argument("--out-root", required=True, type=Path,
                        help="Crops go to <out-root>/reals/t_oi/.")
    parser.add_argument("--reports", type=Path, default=THESIS_DBP / "reports")
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--holdout", type=int, default=400)
    parser.add_argument("--crop", type=int, default=DEFAULT_CROP)
    parser.add_argument("--workers", type=int, default=MAX_PARALLEL)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--hosts", default=None,
                        help="Comma-separated approved host regexes (default: the source's).")
    parser.add_argument("--source", choices=("flickr", "s3"), default="flickr",
                        help="flickr: OriginalURL + OriginalMD5; s3: the CVDF copy "
                             "(rescaled, max 1024 px; sha256 recorded).")
    parser.add_argument("--min-free-tmp-gb", type=float, default=3.0)
    parser.add_argument("--min-free-out-gb", type=float, default=3.0)
    parser.add_argument("--coco-index", required=True, type=Path)
    parser.add_argument("--eval-manifest", action="append", default=[], metavar="NAME=CSV",
                        help="E3/E4/E5 manifests; every real in them is a reference.")
    parser.add_argument("--panel-b-selection", type=Path,
                        default=THESIS_DBP / "reports/kaggle_20261011/p3_panel_b_selection.csv")
    parser.add_argument("--p2-panel-sets", type=Path,
                        default=THESIS_DBP / "reports" / "p2_panel_sets")
    parser.add_argument("--t-real", type=Path,
                        default=THESIS_DBP / "reports" / "p1_reals_manifest_t_real.csv")
    parser.add_argument("--t-ctrl", type=Path,
                        default=THESIS_DBP / "reports" / "p1_reals_manifest_t_ctrl.csv")
    parser.add_argument("--gen-matched", type=Path,
                        default=THESIS_DBP / "reports" / "p1_gen_matched_manifest.csv")
    parser.add_argument("--gen-manifest", required=True, type=Path)
    parser.add_argument("--pairs-manifest", required=True, type=Path)
    parser.add_argument("--local-source", type=Path, default=None,
                        help="DRY-RUN ONLY (DBP_DRYRUN=1): serve files from this directory.")
    args = parser.parse_args()
    log = setup_logging("p3_fetch_open_images")
    started = time.perf_counter()
    crop_dir = args.out_root / "reals" / SET_ID
    summary: dict[str, Any] = {"target": args.target, "holdout": args.holdout,
                               "crop": args.crop, "set_id": SET_ID, "disk": []}
    summary_path = args.reports / "p3_open_images_summary.json"

    def save_summary() -> None:
        summary["seconds"] = round(time.perf_counter() - started, 1)
        args.reports.mkdir(parents=True, exist_ok=True)
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, summary_path)

    if args.workers > MAX_PARALLEL:
        log.error("OPEN IMAGES FAILED: --workers %d exceeds the polite limit of %d",
                  args.workers, MAX_PARALLEL)
        return 1
    if args.local_source and os.environ.get("DBP_DRYRUN") != "1":
        log.error("OPEN IMAGES FAILED: --local-source is for the dry-run only (DBP_DRYRUN=1)")
        return 1
    default_hosts = S3_HOST_PATTERNS if args.source == "s3" else APPROVED_HOST_PATTERNS
    patterns = tuple(h.strip() for h in args.hosts.split(",") if h.strip()) if args.hosts \
        else default_hosts
    summary["source"] = {"mode": args.source, "copy": SOURCE_COPY[args.source]}
    # S3 may redirect only to a regional endpoint of the same bucket (in the patterns).
    redirects = () if args.source == "s3" else REDIRECT_HOSTS
    summary["hosts"] = {"approved_patterns": list(patterns),
                        "redirect_hosts": list(redirects)}
    stage_paths = [args.tmp, args.out_root, args.reports]
    summary["disk"].append(df_report("start", stage_paths, log))
    problems = require_free({args.tmp: args.min_free_tmp_gb,
                             args.out_root: args.min_free_out_gb})
    if problems:
        summary["stopped"] = "insufficient disk before any download: " + "; ".join(problems)
        save_summary()
        log.error("OPEN IMAGES FAILED (nothing downloaded): %s", "; ".join(problems))
        return 1

    rows = load_metadata(args.metadata)
    host_totals = Counter(host_of(fetch_url(r, args.source)) for r in rows)
    summary["metadata"] = {"cc_by_2_rows": len(rows),
                           "rows_on_approved_hosts": sum(
                               n for h, n in host_totals.items() if host_approved(h, patterns)),
                           "rows_by_host": dict(sorted(host_totals.items()))}
    log.info("metadata: %d CC BY 2.0 row(s); %d on approved hosts", len(rows),
             summary["metadata"]["rows_on_approved_hosts"])
    index, ref_counts = build_references(args, log)
    summary["references"] = ref_counts
    summary["disk"].append(df_report("references loaded", stage_paths, log))

    if args.local_source:
        fetcher: Fetcher = LocalFetcher(
            args.local_source, patterns=patterns, max_parallel=args.workers,
            redirect_hosts=redirects, throttle=Throttle(base=0.01, ceiling=0.05,
                                                        log=log.warning))
    else:
        fetcher = Fetcher(patterns=patterns, max_parallel=args.workers,
                          redirect_hosts=redirects, throttle=Throttle(log=log.warning))
    log.info("fetch policy: at most %d connections open at once; %d retries, backoff base "
             "%.1f s; shared slow-down on 403/429/503", fetcher.max_parallel, fetcher.retries,
             fetcher.backoff)
    groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
    group_sizes: defaultdict[str, list[tuple[int, int]]] = defaultdict(list)
    accepted: list[dict[str, Any]] = []
    verified: list[dict[str, Any]] = []
    walk: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    unapproved: Counter[str] = Counter()
    downloaded_bytes = 0
    halted = False
    position = 0
    next_disk_log = 1000
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while len(accepted) < args.target and position < len(rows):
            batch: list[tuple[int, dict[str, str]]] = []
            while len(batch) < args.batch and position < len(rows):
                row = rows[position]
                position += 1
                host = host_of(fetch_url(row, args.source))
                if not host_approved(host, patterns):
                    reason = "host not approved (not contacted)"
                    rejections[reason] += 1
                    unapproved[host] += 1
                    groups[host_group(host)]["rows_walked"] += 1
                    groups[host_group(host)]["not_contacted"] += 1
                    walk.append({"position": position, "oi_image_id": row["ImageID"],
                                 "host": host, "outcome": reason})
                    continue
                batch.append((position, row))
            results = list(pool.map(
                lambda item: process(item[1], fetcher, args.tmp, crop_dir, args.crop,
                                     source=args.source), batch))
            for (row_position, row), result in zip(batch, results, strict=True):
                walked = {"position": row_position, "oi_image_id": row["ImageID"],
                          "host": host_of(fetch_url(row, args.source))}
                group = groups["s3" if args.source == "s3" else host_group(walked["host"])]
                downloaded_bytes += result.get("bytes", 0)
                crop_row = result.get("crop_row")
                if result["outcome"] in ("stopped", "failed") or halted:
                    # A refused or failed download says nothing about the image: the row is
                    # undecided, and in strict walk order nothing after it is decided.
                    halted = True
                    walk.append({**walked, "outcome": "not walked (throttle stopped)"})
                    if crop_row:
                        (args.out_root / crop_row["relative_path"]).unlink(missing_ok=True)
                    continue
                group["rows_walked"] += 1
                group["gone"] += result["outcome"] == "gone"
                group["checksum_mismatch"] += result["outcome"] == "md5"
                group["download_failed"] += result["outcome"] == "failed"
                if result.get("width") and result["outcome"] in ("ok", "skip"):
                    verified.append(result)
                    group_sizes["s3" if args.source == "s3" else host_group(walked["host"])
                                ].append((result["width"], result["height"]))
                if len(accepted) >= args.target:
                    group["not_needed"] += 1
                    walk.append({**walked, "outcome": "not needed (target reached)"})
                    if crop_row:
                        (args.out_root / crop_row["relative_path"]).unlink(missing_ok=True)
                    continue
                if result["outcome"] != "ok":
                    reason = reason_for(result, args.crop)
                else:
                    match = index.match(result["sha256"], result["source_phash"])
                    reason = f"duplicate of a reference ({match})" if match else ""
                    if not reason:
                        match = index.match(crop_row["sha256"], crop_row["phash"])
                        reason = f"crop duplicate of a reference ({match})" if match else ""
                if reason:
                    rejections[reason] += 1
                    walk.append({**walked, "outcome": reason})
                    if crop_row:
                        (args.out_root / crop_row["relative_path"]).unlink(missing_ok=True)
                    continue
                index.add("Open Images (accepted)", result["sha256"], result["source_phash"])
                index.add("Open Images (accepted crop)", crop_row["sha256"], crop_row["phash"])
                accepted.append({"row": row, "result": result})
                group["accepted"] += 1
                walk.append({**walked, "outcome": "accepted"})
                if len(accepted) >= next_disk_log:
                    summary["disk"].append(df_report(f"{len(accepted)} accepted", stage_paths,
                                                     log))
                    next_disk_log += 1000
                    low = [p for p in (args.tmp, args.out_root) if free_gb(p) < 0.5]
                    if low:
                        summary["stopped"] = f"free space below 0.5 GB on {low}"
                        log.error("OPEN IMAGES FAILED: %s", summary["stopped"])
                        position = len(rows)
                        break
            log.info("walked %d / %d row(s): %d accepted, %d rejected", position, len(rows),
                     len(accepted), sum(rejections.values()))
            if fetcher.throttle.tripped:
                summary["stopped"] = ("Flickr kept refusing (403/429/503) for 10 minutes at "
                                      "the longest delay; the walk stopped rather than keep "
                                      "sending requests")
                log.error("OPEN IMAGES FAILED: %s", summary["stopped"])
                break
            if halted:
                summary["stopped"] = ("a download failed after its retries; the walk stopped "
                                      "there to keep its order (the row is undecided)")
                log.error("OPEN IMAGES FAILED: %s", summary["stopped"])
                break

    split = assign_split([a["row"]["ImageID"] for a in accepted], min(args.holdout,
                                                                        len(accepted)))
    manifest, attribution = [], []
    for item in accepted:
        row, result = item["row"], item["result"]
        manifest.append({**result["crop_row"], "oi_image_id": row["ImageID"],
                         "original_url": row["OriginalURL"], "original_md5": row["OriginalMD5"],
                         "original_size": row["OriginalSize"],
                         "downloaded_bytes": result["bytes"], "rotation": row.get("Rotation", ""),
                         "exif_camera": result.get("exif_camera", False),
                         "source_phash": result["source_phash"],
                         "order_key": order_key(row["ImageID"]),
                         "split": split[row["ImageID"]],
                         "fetch_url": fetch_url(row, args.source),
                         "source_copy": SOURCE_COPY[args.source],
                         "md5_checked": args.source == "flickr"})
        attribution.append({"image_id": row["ImageID"], "author": row["Author"],
                            "original_url": row["OriginalURL"], "licence": row["License"]})
    attribution.sort(key=lambda r: r["image_id"])
    write_csv(args.reports / "p3_open_images_manifest.csv",
              (*MANIFEST_COLUMNS, *EXTRA_COLUMNS), manifest)
    write_csv(args.reports / "p3_open_images_attribution.csv", ATTRIBUTION_COLUMNS, attribution)
    write_csv(args.reports / "p3_open_images_walk.csv",
              ("position", "oi_image_id", "host", "outcome"), walk)

    summary["disk"].append(df_report("end", stage_paths, log))
    crop_bytes = sum((args.out_root / m["relative_path"]).stat().st_size for m in manifest)
    summary.update({
        "accepted": len(accepted),
        "shortfall": max(0, args.target - len(accepted)),
        "walked_rows": position,
        "split": dict(Counter(split.values())),
        "rejections": dict(rejections.most_common()),
        "unapproved_hosts_walked": dict(sorted(unapproved.items())),
        "redirect_hosts_seen": dict(Counter(
            a["result"]["final_host"] for a in accepted
            if a["result"]["final_host"] != host_of(fetch_url(a["row"], args.source)))),
        "downloaded_bytes": downloaded_bytes,
        "crop_bytes": crop_bytes,
        "originals_verified": size_stats(verified),
        "originals_accepted": size_stats([a["result"] for a in accepted]),
        "rotation_accepted": dict(Counter(a["row"].get("Rotation", "") for a in accepted)),
        "by_host_group": {
            name: {**{k: int(groups[name][k]) for k in (
                "rows_walked", "accepted", "gone", "checksum_mismatch", "download_failed",
                "not_contacted", "not_needed")},
                "originals_verified": len(group_sizes[name]),
                "median_width": statistics.median(w for w, _ in group_sizes[name])
                if group_sizes[name] else None,
                "median_height": statistics.median(h for _, h in group_sizes[name])
                if group_sizes[name] else None}
            for name in sorted(groups)},
        "fetch_policy": {"max_parallel_connections": fetcher.max_parallel,
                         "retries": fetcher.retries, "backoff_base_s": fetcher.backoff,
                         "timeout_s": fetcher.timeout,
                         "slow_down_on": "403/429/503: shared delay doubles once per wave "
                                         "(or Retry-After), halves after 50 successes; "
                                         "stops after 600 s of refusals at the 60 s ceiling"},
        "throttle": fetcher.throttle.summary(),
        "dry_run_local_source": bool(args.local_source),
    })
    save_summary()
    log.info("Open Images: %d accepted (%s), %d walked, %.2f GB downloaded, crops %.2f GB",
             len(accepted), summary["split"], position, downloaded_bytes / 1e9, crop_bytes / 1e9)
    for reason, n in rejections.most_common():
        log.info("  rejected %6d  %s", n, reason)
    for name, stats in summary["by_host_group"].items():
        log.info("host group %-5s %s", name, stats)
    if unapproved:
        log.info("hosts NOT contacted (approval needed): %s", dict(sorted(unapproved.items())))
    if len(accepted) < args.target:
        log.error("OPEN IMAGES FAILED: %d of %d accepted (shortfall %d)", len(accepted),
                  args.target, args.target - len(accepted))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
