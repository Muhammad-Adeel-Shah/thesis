"""P3 step 4a checks: the Open Images inputs, the output-size gate, and the finished pairs.

Three modes, each writing one JSON record under ``reports/`` before it returns:

* ``inputs`` -- every row of the **committed** Open Images manifest has its PNG under the
  attached dataset with the sha256 the manifest records (CLAUDE.md section 12.2: inputs
  are checked against the committed reference, never against what their writer said).
* ``gate`` -- the output-size gate, run on the pre-flight sample before the long job.
  Each member's mean PNG size is **measured** on the sample (``bytes_written / written``
  in the sample build summaries); the projected output is the sum over members of that
  mean times the number of reals. The gate fails if the projection exceeds the free
  space of the output filesystem minus ``--margin-gb``.
* ``final`` -- the merged pairs manifest: exactly one row per (real, distinct member),
  every PNG present with its recorded sha256, crop 384, and each member at the precision
  P1's committed bench decided (``fp16_build_allowed``), so the new pairs are built
  exactly as the training corpus was. Records the measured bytes per member.

Exit 0 when the check passes, 1 when it does not.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))

from dbp.data.manifest import file_sha256, read_manifest  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

GB = 1024**3


def write_record(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def check_inputs(manifest: Path, reals_root: Path, rows_expected: int) -> dict[str, Any]:
    """Every committed manifest row's PNG is present with its recorded sha256."""
    rows = read_manifest(manifest)
    missing, mismatched = [], []
    for row in rows:
        path = reals_root / row["set_id"] / f"{row['image_id']}.png"
        if not path.is_file():
            missing.append(row["image_id"])
        elif file_sha256(path) != row["sha256"]:
            mismatched.append(row["image_id"])
    ok = len(rows) == rows_expected and not missing and not mismatched
    return {"ok": ok, "rows": len(rows), "rows_expected": rows_expected,
            "missing": missing[:20], "n_missing": len(missing),
            "sha256_mismatch": mismatched[:20], "n_sha256_mismatch": len(mismatched),
            "splits": dict(Counter(r.get("split", "") for r in rows))}


def project(summaries: list[Path], members: list[str], n_reals: int, free_bytes: int,
            margin_gb: float) -> dict[str, Any]:
    """Projected output bytes from the measured sample, against free space minus a margin."""
    per_member: dict[str, dict[str, Any]] = {}
    for path in summaries:
        for entry in json.loads(path.read_text(encoding="utf-8"))["members"]:
            per_member[entry["member"]] = entry
    problems = [f"{m}: no sample" for m in members if m not in per_member]
    problems += [f"{m}: sample {per_member[m]['status']}" for m in members
                 if m in per_member and per_member[m]["status"] != "OK"]
    projected, table = 0, {}
    for member in members:
        entry = per_member.get(member)
        if not entry or not entry.get("written"):
            continue
        mean = entry["bytes_written"] / entry["written"]
        table[member] = {"sample_images": entry["written"], "mean_bytes": round(mean),
                         "projected_bytes": round(mean * n_reals)}
        projected += mean * n_reals
    limit = free_bytes - margin_gb * GB
    ok = not problems and projected <= limit
    return {"ok": ok, "problems": problems, "n_reals": n_reals, "members": table,
            "projected_bytes": round(projected), "projected_gb": round(projected / GB, 2),
            "free_gb": round(free_bytes / GB, 2), "margin_gb": margin_gb,
            "limit_gb": round(limit / GB, 2), "basis": "MEASURED mean PNG size per member "
            "on the pre-flight sample x the number of reals"}


def check_final(manifest: Path, pairs_parent: Path, reals_manifest: Path, throughput: Path,
                members: list[str], crop: int) -> dict[str, Any]:
    """The merged pairs manifest is complete, on disk, and at P1's precisions."""
    reals = [r["image_id"] for r in read_manifest(reals_manifest)]
    rows = read_manifest(manifest)
    allowed = {}
    with throughput.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            allowed[row["key"]] = "fp16" if row["fp16_build_allowed"].strip().lower() in (
                "true", "1", "yes") else "fp32"
    problems: list[str] = []
    seen = Counter((r["image_id"], r["member"]) for r in rows)
    expected = {(i, m) for i in reals for m in members}
    if set(seen) != expected:
        problems.append(f"{len(expected - set(seen))} (real, member) pair(s) missing, "
                        f"{len(set(seen) - expected)} unexpected")
    repeated = [k for k, n in seen.items() if n > 1]
    if repeated:
        problems.append(f"{len(repeated)} pair(s) listed more than once")
    by_member: dict[str, dict[str, Any]] = defaultdict(lambda: {"rows": 0, "bytes": 0})
    bad_files, bad_precision, bad_crop = 0, set(), 0
    for row in rows:
        path = pairs_parent / row["relative_path"]
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            bad_files += 1
            continue
        stats = by_member[row["member"]]
        stats["rows"] += 1
        stats["bytes"] += path.stat().st_size
        if row["precision"] != allowed.get(row["member"], "fp32"):
            bad_precision.add(row["member"])
        if int(row["crop"]) != crop:
            bad_crop += 1
    if bad_files:
        problems.append(f"{bad_files} PNG(s) missing or not matching their sha256")
    if bad_precision:
        problems.append(f"precision differs from P1's bench for {sorted(bad_precision)}")
    if bad_crop:
        problems.append(f"{bad_crop} row(s) not at crop {crop}")
    total = sum(s["bytes"] for s in by_member.values())
    return {"ok": not problems, "problems": problems, "rows": len(rows),
            "rows_expected": len(expected), "members": dict(sorted(by_member.items())),
            "precision_rule": allowed, "bytes": total, "gb": round(total / GB, 3)}


def main() -> int:
    """Run one check; write its record; exit 0 on pass, 1 on fail."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("inputs", "gate", "final"))
    parser.add_argument("--manifest", type=Path, required=True,
                        help="inputs/final: the committed reals manifest; gate: unused.")
    parser.add_argument("--reals-root", type=Path, default=None)
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--summary", type=Path, action="append", default=[])
    parser.add_argument("--members", default="", help="Default: every distinct bank member.")
    parser.add_argument("--free-path", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--margin-gb", type=float, default=2.0)
    parser.add_argument("--pairs-manifest", type=Path, default=None)
    parser.add_argument("--pairs-parent", type=Path, default=None,
                        help="The directory pair relative_paths are relative to.")
    parser.add_argument("--throughput-csv", type=Path,
                        default=THESIS_DBP / "reports" / "kaggle_20260926" / "p1_throughput.csv")
    parser.add_argument("--crop", type=int, default=384)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    log = setup_logging("p3_oi_pairs_check")
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    members = [m for m in args.members.split(",") if m] or \
        [s.key for s in bank.distinct_members()]

    if args.mode == "inputs":
        record = check_inputs(args.manifest, args.reals_root, args.rows)
    elif args.mode == "gate":
        n_reals = len(read_manifest(args.manifest))
        free = shutil.disk_usage(args.free_path).free
        record = project(args.summary, members, n_reals, free, args.margin_gb)
    else:
        record = check_final(args.pairs_manifest, args.pairs_parent, args.manifest,
                             args.throughput_csv, members, args.crop)
    record = {"mode": args.mode, **record}
    write_record(args.out, record)
    log.info("%s: %s -> %s", args.mode, "PASSED" if record["ok"] else "FAILED", args.out.name)
    for key in ("problems", "missing", "sha256_mismatch"):
        if record.get(key):
            log.error("%s: %s", key, record[key])
    if args.mode == "gate":
        log.info("projected output %.2f GB; free %.2f GB; margin %.1f GB", record["projected_gb"],
                 record["free_gb"], record["margin_gb"])
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
