"""P3 Phase B step 3, route (ii), LOCAL stage: fetch Open Images originals on the user's PC.

Flickr's CDN refuses Kaggle's IPs (run 1: 24/24 × 429; probe 2026-10-13: 6/6 × 429 at
30 s spacing), so -- by the user's decision of 2026-10-13, a CLAUDE.md §6a(a) runbook
deviation for this one source, not a data-rule change -- the download moves to the
user's PC. **The data rules are unchanged**: same metadata, same seed-2026 walk order,
same hosts, same ``OriginalMD5`` check, same ≥ 384 / greyscale rules, same crop
(``p1_build_reals.build_rows``: native resolution, PNG), same sha256 + pHash.

What this stage does **not** do is the dedup: the references (all of COCO, Panel B,
P2, the training corpus, every E3/E4/E5 real) are checked on Kaggle by
``p3_oi_dedup_accept.py``, which accepts candidates in the same seed order up to 4,000.

* Walks the seed order and keeps the first ``--target-candidates`` rows that pass
  download, checksum, eligibility and greyscale -- **candidates**. The target exceeds
  4,000 by a margin for the dedup losses; if the Kaggle stage still falls short, rerun
  this script with a larger target: it **resumes** after the last decided row, so the
  order is never broken and nothing is fetched twice.
* Raw originals of candidates are kept in ``--originals``; everything else is deleted.
  Both ``--originals`` and ``--out`` must be outside REPO_ROOT (refused otherwise).
* Politeness as on Kaggle: approved hosts only, at most ``--workers`` (≤ 8; default 4)
  connections, retries with backoff, the shared slow-down on 403/429/503, and a stop
  after 10 minutes of sustained refusals.
* Stops before the first download if the disk holding ``--originals`` has less than
  ``--min-free-gb`` free; logs disk use at every stage.

Outputs in ``--out``: ``reals/t_oi/<image_id>.png`` (the crops to upload),
``p3_oi_candidates_manifest.csv``, ``p3_oi_local_walk.csv``, ``p3_oi_local_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
REPO_ROOT = THESIS_DBP.parent
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.open_images import (  # noqa: E402
    APPROVED_HOST_PATTERNS,
    MAX_PARALLEL,
    Fetcher,
    LocalFetcher,
    Throttle,
    decoder_versions,
    host_approved,
    host_group,
    host_of,
    load_metadata,
    order_key,
)
from dbp.utils.disk import df_report, require_free  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p1_build_reals import DEFAULT_CROP, MANIFEST_COLUMNS  # noqa: E402
from p3_fetch_open_images import SET_ID, process, reason_for, size_stats  # noqa: E402

CANDIDATE_COLUMNS = (*MANIFEST_COLUMNS, "position", "order_key", "oi_image_id", "host",
                     "host_group", "original_url", "original_md5", "original_size",
                     "original_sha256", "original_width", "original_height",
                     "downloaded_bytes", "rotation", "exif_camera", "source_phash", "author",
                     "licence", "decoder_python", "decoder_pillow", "decoder_libjpeg")
WALK_COLUMNS = ("position", "oi_image_id", "host", "outcome")
SESSION_COLUMNS = ("start_utc", "end_utc", "workers", "min_interval_s", "rows_decided_before",
                   "rows_decided_after", "candidates_before", "candidates_after",
                   "requests_started", "refusals", "slow_down_raises", "tripped", "ended",
                   "exit_code")
DEFAULT_METADATA = THESIS_DBP / "reports" / "p3_open_images" / "oi_validation_cc_by_2.csv"
#: Walk outcomes that are not decisions about the image (logged by an earlier version).
UNDECIDED = ("download failed",)
#: Failed attempts per row in one session before it waits for the next session.
MAX_FAILURES_PER_SESSION = 2


def read_csv(path: Path) -> list[dict[str, str]]:
    """A CSV as dicts ([] if it does not exist)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    """Atomic CSV write."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def outside_repo(path: Path) -> bool:
    """Whether ``path`` lies outside REPO_ROOT (raw data must never be inside it)."""
    return not Path(path).resolve().is_relative_to(REPO_ROOT.resolve())


def utc_now() -> str:
    """Now, as an ISO UTC timestamp."""
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def print_status(out: Path) -> int:
    """Rows decided, candidates and the last session, from the saved state; no network."""
    summary_path = out / "p3_oi_local_summary.json"
    if not summary_path.is_file():
        print(f"no state yet in {out}")
        return 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sessions = read_csv(out / "p3_oi_local_sessions.csv")
    print(f"rows decided {summary['rows_decided']} | candidates {summary['candidates']} of "
          f"{summary['target_candidates']} | first undecided position "
          f"{summary['first_undecided_position']} | target met "
          f"{summary['target_met_without_gaps']}")
    if sessions:
        last = sessions[-1]
        print(f"last session {last['start_utc']} -> {last['end_utc']}: pace "
              f"{last['workers']} connection(s), min interval {last['min_interval_s']} s; "
              f"refusals {last['refusals']}; ended: {last['ended']} (exit {last['exit_code']})")
    return 0


def sweep_orphans(out: Path, originals: Path, candidates: list[dict[str, str]], log) -> int:  # noqa: ANN001
    """Delete crops and originals that belong to no recorded candidate (a hard kill can
    leave them); returns how many files were removed."""
    keep_crops = {Path(c["relative_path"]).name for c in candidates}
    keep_ids = {c["oi_image_id"] for c in candidates}
    removed = 0
    for path in (out / "reals" / SET_ID).glob("*.png"):
        if path.name not in keep_crops:
            path.unlink()
            removed += 1
    for path in originals.iterdir():
        if path.is_file() and path.name.split(".")[0] not in keep_ids:
            path.unlink()
            removed += 1
    if removed:
        log.info("removed %d file(s) left by an interrupted batch (not candidates)", removed)
    return removed


def main() -> int:  # noqa: PLR0915 - one linear walk, kept in one place on purpose
    """Run (or resume) the local walk. Exit 0 once the candidate target is met."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--originals", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-candidates", type=int, default=4200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--crop", type=int, default=DEFAULT_CROP)
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Stop after this many walked rows (a size check; default none).")
    parser.add_argument("--local-source", type=Path, default=None,
                        help="DRY-RUN ONLY (DBP_DRYRUN=1): serve files from this directory.")
    parser.add_argument("--min-interval", type=float, default=0.0,
                        help="Minimum seconds between the starts of any two requests.")
    parser.add_argument("--max-minutes", type=float, default=None,
                        help="End the session cleanly after this many minutes (resumable).")
    parser.add_argument("--not-before", default=None,
                        help="Refuse to start before this UTC time, e.g. 2026-10-15T06:00Z.")
    parser.add_argument("--status", action="store_true",
                        help="Print rows decided and candidates from the saved state; no "
                             "network.")
    args = parser.parse_args()
    if args.status:
        return print_status(args.out)
    log = setup_logging("p3_oi_local_fetch")
    if args.not_before:
        earliest = dt.datetime.fromisoformat(args.not_before.replace("Z", "+00:00"))
        if dt.datetime.now(dt.UTC) < earliest:
            log.error("NOT STARTED: --not-before %s is still in the future (now %s)",
                      args.not_before, utc_now())
            return 2
    session_start = utc_now()
    started = time.perf_counter()
    if args.workers > MAX_PARALLEL:
        log.error("LOCAL FETCH FAILED: --workers %d exceeds %d", args.workers, MAX_PARALLEL)
        return 1
    if args.local_source and os.environ.get("DBP_DRYRUN") != "1":
        log.error("LOCAL FETCH FAILED: --local-source is for the dry-run only (DBP_DRYRUN=1)")
        return 1
    if os.environ.get("DBP_DRYRUN") != "1":
        for name, path in (("--originals", args.originals), ("--out", args.out)):
            if not outside_repo(path):
                log.error("LOCAL FETCH FAILED: %s %s is inside REPO_ROOT %s; raw data must "
                          "live under F:\\thesis-2\\kaggle_raw\\", name, path, REPO_ROOT)
                return 1
    crop_dir = args.out / "reals" / SET_ID
    for path in (args.originals, crop_dir):
        path.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "p3_oi_candidates_manifest.csv"
    walk_path = args.out / "p3_oi_local_walk.csv"
    summary_path = args.out / "p3_oi_local_summary.json"
    prior = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() \
        else {}
    # The walk log is the source of truth. A refused or failed download says nothing about
    # the image, so such rows are UNDECIDED: they are dropped here and fetched again, in
    # order (regression, local run 1, 2026-10-14: 15 rows refused with 429 had been logged
    # as rejected).
    walk = [w for w in read_csv(walk_path) if not w["outcome"].startswith(UNDECIDED)]
    dropped = sum(1 for w in read_csv(walk_path) if w["outcome"].startswith(UNDECIDED))
    decided: dict[int, str] = {int(w["position"]): w["outcome"] for w in walk}
    candidates = [c for c in read_csv(manifest_path)
                  if decided.get(int(c["position"])) == "candidate"]
    sweep_orphans(args.out, args.originals, candidates, log)
    decided_before, candidates_before = len(decided), len(candidates)
    stage_paths = [args.originals, args.out]
    disk = list(prior.get("disk", []))
    disk.append(df_report(f"start ({len(decided)} rows decided; {dropped} undecided row(s) "
                          "from earlier sessions to fetch again)", stage_paths, log))
    problems = require_free({args.originals: args.min_free_gb})
    if problems:
        log.error("LOCAL FETCH FAILED (nothing downloaded): %s", "; ".join(problems))
        return 1

    rows = load_metadata(args.metadata)
    patterns = APPROVED_HOST_PATTERNS
    versions = decoder_versions()
    log.info("decoder: %s; EXIF orientation and ICC profiles NOT applied (as P1 for COCO)",
             versions)
    throttle = Throttle(log=log.warning) if not args.local_source else \
        Throttle(base=0.01, ceiling=0.05, log=log.warning)
    fetcher: Fetcher = (LocalFetcher(args.local_source, max_parallel=args.workers,
                                     throttle=throttle, min_interval=args.min_interval)
                        if args.local_source else
                        Fetcher(max_parallel=args.workers, throttle=throttle,
                                min_interval=args.min_interval))
    log.info("pace: at most %d connection(s); at least %.1f s between request starts; "
             "session limit %s min", args.workers, args.min_interval, args.max_minutes)
    log.info("walk: %d CC BY 2.0 row(s); %d decided, %d candidate(s) so far; target %d "
             "candidates; %d connection(s) at most", len(rows), len(decided), len(candidates),
             args.target_candidates, args.workers)
    verified_sizes: list[dict[str, Any]] = []
    downloaded = int(prior.get("downloaded_bytes", 0))
    failures: Counter[int] = Counter()        # this session's failed attempts per position
    transient: Counter[str] = Counter()
    halted = False
    next_disk = (len(candidates) // 500 + 1) * 500

    def prefix() -> tuple[bool, int, int]:
        """(target met with no undecided row before it, candidates so far, first gap)."""
        found = 0
        for pos in range(1, len(rows) + 1):
            if pos not in decided:
                return False, found, pos
            if decided[pos] == "candidate":
                found += 1
                if found >= args.target_candidates:
                    return True, found, pos
        return False, found, len(rows) + 1

    def save() -> None:
        write_csv(manifest_path, CANDIDATE_COLUMNS,
                  sorted(candidates, key=lambda c: int(c["position"])))
        write_csv(walk_path, WALK_COLUMNS, sorted(walk, key=lambda w: int(w["position"])))
        rejections = Counter(w["outcome"] for w in walk if w["outcome"] != "candidate")
        groups: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for w in walk:
            g = groups[host_group(w["host"])]
            g["rows_decided"] += 1
            g["candidates"] += w["outcome"] == "candidate"
            g["gone"] += w["outcome"].startswith("download: gone")
            g["checksum_mismatch"] += w["outcome"].startswith("checksum mismatch")
            g["not_contacted"] += w["outcome"].startswith("host not approved")
        done, found, gap = prefix()
        summary = {
            "stage": "p3_oi_local_fetch", "route": "(ii) local download, user 2026-10-13",
            "target_candidates": args.target_candidates, "candidates": len(candidates),
            "rows_decided": len(decided), "highest_decided_position": max(decided, default=0),
            "first_undecided_position": gap, "target_met_without_gaps": done,
            "rejections": dict(rejections.most_common()),
            "by_host_group": {k: dict(v) for k, v in sorted(groups.items())},
            "undecided_failures_this_session": dict(transient),
            "undecided_positions": sorted(p for p in range(1, max(decided, default=0) + 1)
                                          if p not in decided),
            "downloaded_bytes": downloaded,
            "originals_kept_bytes": sum(int(c["downloaded_bytes"]) for c in candidates),
            "crop_bytes": sum((args.out / c["relative_path"]).stat().st_size
                              for c in candidates),
            "originals_verified_this_session": size_stats(verified_sizes),
            "throttle_this_session": throttle.summary(),
            "max_connections": args.workers, "disk": disk, "decoder": versions,
            "decoding": "Image.open().convert('RGB') via p1_build_reals.build_rows, as P1 "
                        "decoded COCO; EXIF orientation and ICC profiles not applied; "
                        "non-RGB modes rejected, never converted",
            "seconds_this_session": round(time.perf_counter() - started, 1),
            "dry_run_local_source": bool(args.local_source),
        }
        tmp = summary_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, summary_path)

    def discard(row: dict[str, str], result: dict[str, Any] | None) -> None:
        """Remove what an undecided row left on disk."""
        if result and result.get("crop_row"):
            (args.out / result["crop_row"]["relative_path"]).unlink(missing_ok=True)
        for path in args.originals.glob(f"{row['ImageID']}.*"):
            path.unlink()

    pool = ThreadPoolExecutor(max_workers=args.workers)
    interrupted = timed_out = False
    try:
        while not halted:
            done, found, _gap = prefix()
            if done:
                break
            if args.max_minutes is not None and \
                    time.perf_counter() - started >= 60 * args.max_minutes:
                timed_out = True
                log.info("session limit of %.0f min reached; ending cleanly (resumable)",
                         args.max_minutes)
                break
            # Earlier gaps first; new rows only while candidates are still short.
            limit = len(rows) if len(candidates) < args.target_candidates else \
                max(decided, default=0)
            if args.max_rows is not None:
                limit = min(limit, args.max_rows)
            todo = [pos for pos in range(1, limit + 1)
                    if pos not in decided and failures[pos] < MAX_FAILURES_PER_SESSION]
            if not todo:
                break
            batch = [(pos, rows[pos - 1]) for pos in todo[:args.batch]]
            fetchable = [(pos, r) for pos, r in batch
                         if host_approved(host_of(r["OriginalURL"]), patterns)]
            futures = {pos: pool.submit(process, r, fetcher, args.originals, crop_dir,
                                        args.crop, True) for pos, r in fetchable}
            pending = set(futures.values())
            while pending:                      # short waits keep Ctrl+C responsive on Windows
                _finished, pending = wait(pending, timeout=1.0)
            results = {pos: future.result() for pos, future in futures.items()}
            for row_position, row in batch:
                host = host_of(row["OriginalURL"])
                result = results.get(row_position)
                record = {"position": row_position, "oi_image_id": row["ImageID"], "host": host}
                if result is not None and result["outcome"] in ("stopped", "failed"):
                    # Not a property of the image: undecided, fetched again later.
                    if result["outcome"] == "stopped":
                        halted = True
                    else:
                        failures[row_position] += 1
                        transient[result.get("detail", "").split(" after ")[0] or
                                  result["outcome"]] += 1
                    discard(row, result)
                    continue
                if result is None:
                    reason = "host not approved (not contacted)"
                else:
                    downloaded += result.get("bytes", 0)
                    if result.get("width") and result["outcome"] in ("ok", "skip"):
                        verified_sizes.append(result)
                    reason = "" if result["outcome"] == "ok" else reason_for(result, args.crop)
                decided[row_position] = reason or "candidate"
                if reason:
                    walk.append({**record, "outcome": reason})
                    continue
                crop_row = result["crop_row"]
                candidates.append({
                    **crop_row, "position": row_position, "order_key": order_key(row["ImageID"]),
                    "oi_image_id": row["ImageID"], "host": host, "host_group": host_group(host),
                    "original_url": row["OriginalURL"], "original_md5": row["OriginalMD5"],
                    "original_size": row["OriginalSize"], "original_sha256": result["sha256"],
                    "original_width": result["width"], "original_height": result["height"],
                    "downloaded_bytes": result["bytes"], "rotation": row.get("Rotation", ""),
                    "exif_camera": result.get("exif_camera", False),
                    "source_phash": result["source_phash"], "author": row["Author"],
                    "licence": row["License"], **versions})
                walk.append({**record, "outcome": "candidate"})
            if len(candidates) >= next_disk:
                disk.append(df_report(f"{len(candidates)} candidates", stage_paths, log))
                next_disk += 500
            save()
            log.info("decided %d row(s) (highest position %d): %d candidate(s); %d undecided "
                     "attempt(s) this session", len(decided), max(decided, default=0),
                     len(candidates), sum(failures.values()))
    except KeyboardInterrupt:
        interrupted = True
        log.warning("interrupted (Ctrl+C): the batch in progress is not recorded and will be "
                    "fetched again on resume; every earlier batch is saved")
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if interrupted:
        sweep_orphans(args.out, args.originals, candidates, log)

    done, found, gap = prefix()
    if done:
        # The candidate set is the first N passing rows in seed order: drop any beyond.
        last = gap
        for cand in [c for c in candidates if int(c["position"]) > last]:
            candidates.remove(cand)
            walk[:] = [w for w in walk if int(w["position"]) != int(cand["position"])]
            decided.pop(int(cand["position"]), None)
            (args.out / cand["relative_path"]).unlink(missing_ok=True)
            for path in args.originals.glob(f"{cand['oi_image_id']}.*"):
                path.unlink()
    disk.append(df_report("end", stage_paths, log))
    save()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for group_name, counts in summary["by_host_group"].items():
        log.info("host group %-5s %s", group_name, counts)
    for reason, n in summary["rejections"].items():
        log.info("  rejected %6d  %s", n, reason)
    if halted:
        ended, code = "throttle stop (10 min of refusals)", 1
        log.error("LOCAL FETCH STOPPED: Flickr kept refusing; state saved (%d decided, %d "
                  "candidates, first undecided position %d). Rerun only when told to resume.",
                  len(decided), len(candidates), gap)
    elif interrupted:
        ended, code = "interrupted (Ctrl+C)", 130
        log.warning("LOCAL FETCH INTERRUPTED: state saved (%d decided, %d candidates); the "
                    "same command resumes exactly", len(decided), len(candidates))
    elif done:
        ended, code = "target met", 0
        log.info("LOCAL FETCH DONE: %d candidates, the last at position %d; no undecided row "
                 "before it", len(candidates), gap)
    elif timed_out:
        ended, code = "session time limit", 0
        log.info("LOCAL FETCH SESSION ENDED at the time limit: %d decided, %d candidates; run "
                 "the continue command to resume", len(decided), len(candidates))
    else:
        ended, code = "incomplete (undecided rows left)", 1
        log.error("LOCAL FETCH INCOMPLETE: %d of %d candidates; first undecided position %d",
                  found, args.target_candidates, gap)
    record = {"start_utc": session_start, "end_utc": utc_now(), "workers": args.workers,
              "min_interval_s": args.min_interval, "rows_decided_before": decided_before,
              "rows_decided_after": len(decided), "candidates_before": candidates_before,
              "candidates_after": len(candidates), "requests_started": fetcher.requests_started,
              "refusals": throttle.refusals, "slow_down_raises": throttle.raises,
              "tripped": throttle.tripped, "ended": ended, "exit_code": code}
    sessions_path = args.out / "p3_oi_local_sessions.csv"
    write_csv(sessions_path, SESSION_COLUMNS, [*read_csv(sessions_path), record])
    log.info("session: %s", record)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
