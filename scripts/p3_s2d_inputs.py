"""P3 S2-D inputs: the merged training files that add the Open Images train reals, verified.

S2-D and S2-B+reals train with the **unchanged** P2/P3 trainer (user, 2026-10-21: the
trainer stays byte-identical). They reach it through merged copies of the three files
it reads -- the pairs manifest, the split and the reals manifest -- with the Open Images
train rows added. The trainer then pools COCO and Open Images reals and samples them
uniformly by real, exactly as it does today (user decision 2026-10-21, option (a)).

Two modes, each writing one JSON record under ``reports/`` before it returns:

* ``build`` -- verify, stage, then write the merged files:

  1. the committed Open Images reals manifest has its committed sha256, and splits into
     exactly 3,600 ``train`` and 400 ``held_out`` reals;
  2. the attached pairs manifest (the step-4a output) matches the committed digest
     (``p3_oi_pairs_manifest_digest.json``: rows, rows per member, and the content
     digest over ``<pair_id>\\t<sha256>``);
  3. no Open Images id collides with a COCO id;
  4. the merged files are prepared in ``<out-dir>/.pending``: COCO rows unchanged, then
     the Open Images train rows; the held-out ids appear in none of them;
  5. the S2-D run is resolved from the **committed** tuning result
     (``reports/p3_degradation_choice.json``: ``chosen_p``), so no run is picked by hand;
  6. each queued run's real pool is built with the trainer's own ``TrainingData`` on the
     merged files and counted by source -- the pool share the run samples from;
  7. **staging** (user, 2026-10-21): every file the queued runs will read -- their train
     and validation pairs, those pairs' reals, and gen_matched -- is read **once** from
     the corpus root by a parallel pass (``--workers`` threads) and written to
     ``--local-root`` on local disk, from which the trainer then reads. The Open Images
     train reals and reconstructions are hashed against the manifests **in the same read**
     (46,800 files); the held-out ones are never read. Before copying, the pass records the
     files' total size, the host's RAM and the free space, and refuses if the copy does
     not fit. A local copy rather than a page-cache warm-up: about 17.5-20 GB against
     33.7 GB of RAM shared with two training processes could not be shown to stay cached.
     gen_matched is counted but not copied: the session's matcher wrote it to local disk,
     and its manifest holds absolute paths the trainer reads whatever its root;
  8. only when everything passed are the merged files moved into ``--out-dir`` and the
     session queue written (``p3_s2d_queue.json``).

* ``check`` -- the pre-flight: re-read the merged files, refuse if any held-out Open
  Images id is in any of them or any held-out file is in the local copy, and confirm the
  merged files' sha256 and Open Images row counts equal what ``build`` recorded.

Exit 0 when the mode passes, 1 when it does not.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.manifest import file_sha256, read_manifest  # noqa: E402
from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.run_config import load_run_configs  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402

OI_SET = "t_oi"
#: The corpus-root links the runbook stages to the attached Open Images inputs.
OI_REALS_LINK = "oi_reals"
OI_PAIRS_LINK = "oi_pairs"
TRAIN, HELD_OUT = "train", "held_out"
PAIRS_NAME = "p1_pairs_manifest.csv"
SPLITS_NAME = "p2_splits.csv"
MANIFESTS_DIR = "manifests"
T_REAL = "p1_reals_manifest_t_real.csv"
#: Copied unchanged into the merged manifests directory (the trainer reads them there).
COPIED = ("p1_reals_manifest_t_ctrl.csv", "p1_gen_matched_manifest.csv")
MERGED_NAMES = (PAIRS_NAME, SPLITS_NAME, f"{MANIFESTS_DIR}/{T_REAL}",
                *[f"{MANIFESTS_DIR}/{n}" for n in COPIED])
PENDING = ".pending"
QUEUE_NAME = "p3_s2d_queue.json"
#: The S2-D entry of ``configs/p3_runs.yaml`` for a chosen p: s2d_p70, s2d_p35.
S2D_RUN = "s2d_p{:02d}"
GB = 1024**3


def write_record(path: Path, payload: dict[str, Any]) -> None:
    """Atomic JSON write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def write_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """Atomic CSV write, LF line endings, every row restricted to ``columns``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
    tmp.replace(path)


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """A CSV's header and rows."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def content_digest(rows: list[dict[str, str]]) -> str:
    """sha256 of the sorted lines ``<pair_id>\\t<sha256>\\n`` (the committed digest's rule)."""
    lines = sorted(f"{r['pair_id']}\t{r['sha256']}\n" for r in rows)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def oi_split(reals: list[dict[str, str]], expect: dict[str, Any]) -> tuple[set[str], set[str],
                                                                           list[str]]:
    """The train and held-out Open Images ids, and any problem with the split."""
    train = {r["image_id"] for r in reals if r.get("split") == TRAIN}
    held = {r["image_id"] for r in reals if r.get("split") == HELD_OUT}
    problems = []
    if len(train) != expect["train_reals"] or len(held) != expect["held_out_reals"]:
        problems.append(f"Open Images split is {len(train)} train / {len(held)} held out, "
                        f"expected {expect['train_reals']} / {expect['held_out_reals']}")
    other = Counter(r.get("split", "") for r in reals) - Counter({TRAIN: len(train),
                                                                  HELD_OUT: len(held)})
    if +other:
        problems.append(f"Open Images rows with another split: {dict(+other)}")
    if any(r.get("excluded_reason") for r in reals):
        problems.append("an Open Images real carries an excluded_reason")
    return train, held, problems


def memory_gb() -> dict[str, float] | None:
    """MemTotal, MemAvailable and Cached from ``/proc/meminfo`` (None off Linux)."""
    info = Path("/proc/meminfo")
    if not info.is_file():
        return None
    values = {}
    for line in info.read_text(encoding="utf-8").splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable", "Cached"):
            values[key] = round(int(rest.split()[0]) / 1024**2, 2)
    return values


def source_of(relative: str) -> str:
    """Which corpus a staged file belongs to, for the size report."""
    if relative.startswith((f"{OI_REALS_LINK}/", f"{OI_PAIRS_LINK}/")):
        return "open_images"
    if Path(relative).is_absolute() or relative.startswith("gen_matched/"):
        return "gen_matched"
    return "coco"


def stage(relatives: list[str], expected: dict[str, str], root: Path, local_root: Path,
          workers: int, margin_gb: float, log: Any) -> dict[str, Any]:
    """Read every file once from ``root``, hash-check those in ``expected``, copy to local disk.

    A stat pass first measures the total size; the copy is refused when it would not fit
    in the free space of ``local_root`` minus ``margin_gb``.
    """
    workers = max(1, workers)
    started = time.perf_counter()

    def size(relative: str) -> int | None:
        path = root / relative
        return path.stat().st_size if path.is_file() else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        sizes = dict(zip(relatives, pool.map(size, relatives), strict=True))
    stat_seconds = time.perf_counter() - started
    missing = sorted(r for r, s in sizes.items() if s is None)
    by_source: dict[str, dict[str, int]] = {}
    for relative, value in sizes.items():
        entry = by_source.setdefault(source_of(relative), {"files": 0, "bytes": 0})
        entry["files"] += 1
        entry["bytes"] += value or 0
    total = sum(v for v in sizes.values() if v)
    local_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(local_root).free
    record: dict[str, Any] = {
        "local_root": str(local_root), "files": len(relatives), "bytes": total,
        "gb": round(total / GB, 3), "by_source": by_source, "workers": workers,
        "free_gb_before": round(free / GB, 2), "margin_gb": margin_gb,
        "memory_gb_before": memory_gb(), "stat_seconds": round(stat_seconds, 1),
        "n_missing": len(missing), "missing": missing[:20],
    }
    log.info("staging %d files, %.2f GB (%s); free %.1f GB at %s; memory %s",
             len(relatives), total / GB, by_source, free / GB, local_root,
             record["memory_gb_before"])
    if missing:
        return {**record, "ok": False, "reason": f"{len(missing)} file(s) missing"}
    if total > free - margin_gb * GB:
        return {**record, "ok": False,
                "reason": f"{total / GB:.2f} GB does not fit in {free / GB:.2f} GB free "
                          f"minus a {margin_gb} GB margin"}

    for parent in sorted({(local_root / r).parent for r in relatives
                          if not Path(r).is_absolute()}):
        parent.mkdir(parents=True, exist_ok=True)

    def copy(relative: str) -> str:
        if Path(relative).is_absolute():
            # gen_matched: the session's matcher wrote these to local disk and the manifest
            # holds their absolute paths, which the trainer reads whatever its root.
            return "local"
        data = (root / relative).read_bytes()
        want = expected.get(relative)
        if want is not None and hashlib.sha256(data).hexdigest() != want:
            return "mismatch"
        target = local_root / relative
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        return "verified" if want is not None else "copied"

    copy_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        states = dict(zip(relatives, pool.map(copy, relatives), strict=True))
    seconds = time.perf_counter() - copy_started
    mismatched = sorted(r for r, s in states.items() if s == "mismatch")
    record.update({
        "seconds": round(seconds, 1),
        "files_per_s": round(len(relatives) / seconds, 1) if seconds else None,
        "mb_per_s": round(total / 1024**2 / seconds, 1) if seconds else None,
        "verified": sum(1 for s in states.values() if s == "verified"),
        "already_local": sum(1 for s in states.values() if s == "local"),
        "n_sha256_mismatch": len(mismatched), "sha256_mismatch": mismatched[:20],
        "memory_gb_after": memory_gb(),
        "free_gb_after": round(shutil.disk_usage(local_root).free / GB, 2),
    })
    log.info("staged in %.0f s (%.1f files/s, %.1f MB/s); %d hash-verified, %d mismatched",
             seconds, record["files_per_s"] or 0, record["mb_per_s"] or 0, record["verified"],
             len(mismatched))
    ok = not mismatched
    return {**record, "ok": ok, "reason": "" if ok else f"{len(mismatched)} sha256 mismatch(es)"}


def resolve_queue(choice: dict[str, Any], expect: dict[str, Any],
                  runs: Any) -> tuple[list[str], list[str]]:
    """The session queue with ``s2d`` replaced by the entry of the chosen p, and problems."""
    problems: list[str] = []
    chosen_p = choice.get("chosen_p")
    if not isinstance(chosen_p, (int, float)):
        return [], [f"the tuning result has no numeric chosen_p: {chosen_p!r}"]
    s2d = S2D_RUN.format(round(100 * float(chosen_p)))
    try:
        config = runs.by_id(s2d)
    except KeyError:
        return [], [f"configs/p3_runs.yaml has no run {s2d!r} for chosen_p {chosen_p}"]
    if abs(config.degrade_p - float(chosen_p)) > 1e-9:
        problems.append(f"{s2d} has degrade_p {config.degrade_p}, the tuning chose {chosen_p}")
    queue = []
    for spec in expect["queue_template"]:
        protocol, _, seed = spec.partition(":")
        queue.append(f"{s2d if protocol == 's2d' else protocol}:{seed}")
    return queue, problems


def training_data(root: Path, merged: Path, run: Any, bank: Any, log: Any) -> tuple[Any, Any]:
    """The trainer's own ``TrainingData`` and reals index on the merged files."""
    from p2_train_s2 import TrainingData, load_reals_index

    manifests = merged / MANIFESTS_DIR
    reals_index = load_reals_index(manifests)
    data = TrainingData(root, merged / SPLITS_NAME, merged / PAIRS_NAME,
                        manifests / "p1_gen_matched_manifest.csv",
                        set(run.active_members(bank)), set(reals_index), log)
    return data, reals_index


def pool_by_source(data: Any, oi_ids: set[str]) -> dict[str, Any]:
    """The run's training real pool, counted by source."""
    oi = sum(1 for real in data.train_reals if real in oi_ids)
    total = len(data.train_reals)
    oi_pairs = sum(len(data.by_real[r]) for r in data.train_reals if r in oi_ids)
    return {"reals": total, "coco_reals": total - oi, "oi_reals": oi,
            "oi_share_of_reals": round(oi / total, 6) if total else None,
            "train_pair_rows": len(data.pairs["train"]), "oi_pair_rows": oi_pairs,
            "basis": "the trainer's TrainingData on the merged files; sampling is uniform "
                     "by real, so this is the expected share of sampled reals"}


def files_read(data: Any, reals_index: dict[str, str]) -> set[str]:
    """Every corpus-relative path the trainer can read for this run's data."""
    paths = set()
    for side in ("train", "val"):
        for row in data.pairs[side]:
            paths.add(row["relative_path"])
            paths.add(reals_index[row["image_id"]])
        for gen in data.gen[side]:
            paths.add(gen["jpg_path"])
    return paths


def build(args: argparse.Namespace, log: Any) -> dict[str, Any]:
    """Verify every input, stage the run's files, then write the merged files and the queue."""
    expect = json.loads(args.expect.read_text(encoding="utf-8"))["oi"]
    problems: list[str] = []
    record: dict[str, Any] = {"mode": "build"}

    # 1. the committed reals manifest and its split
    reals_sha = file_sha256(args.oi_manifest)
    record["oi_reals_manifest"] = {"path": str(args.oi_manifest), "sha256": reals_sha}
    if reals_sha != expect["reals_manifest_sha256"]:
        problems.append("the Open Images reals manifest is not the committed one")
    oi_reals = read_manifest(args.oi_manifest)
    train, held, split_problems = oi_split(oi_reals, expect)
    problems += split_problems

    # 2. the attached pairs manifest against the committed digest
    digest = json.loads(args.oi_pairs_digest.read_text(encoding="utf-8"))["manifest"]
    oi_pairs = read_manifest(args.oi_pairs_manifest)
    per_member = dict(sorted(Counter(r["member"] for r in oi_pairs).items()))
    content = content_digest(oi_pairs)
    record["oi_pairs_manifest"] = {"path": str(args.oi_pairs_manifest), "rows": len(oi_pairs),
                                   "content_sha256": content, "rows_per_member": per_member}
    if len(oi_pairs) != digest["rows"] or per_member != digest["rows_per_member"]:
        problems.append(f"the pairs manifest has {len(oi_pairs)} rows / {per_member}, the "
                        f"committed digest {digest['rows']} / {digest['rows_per_member']}")
    if content != digest["content_sha256"]:
        problems.append("the pairs manifest's content digest differs from the committed one")
    unknown = {r["image_id"] for r in oi_pairs} - train - held
    if unknown:
        problems.append(f"{len(unknown)} pair row(s) of reals not in the reals manifest")
    train_pairs = [r for r in oi_pairs if r["image_id"] in train]
    members = sorted(per_member)
    if len(train_pairs) != len(train) * len(members):
        problems.append(f"{len(train_pairs)} train pair rows, expected "
                        f"{len(train)} x {len(members)}")
    if problems:
        return {**record, "ok": False, "problems": problems}

    # 3. no collision with the COCO corpus
    pair_cols, coco_pairs = read_rows(args.pairs_manifest)
    split_cols, coco_splits = read_rows(args.splits)
    real_cols, coco_reals = read_rows(args.manifests / T_REAL)
    coco_ids = {r["image_id"] for r in coco_reals}
    for name in COPIED[:1]:
        coco_ids |= {r["image_id"] for r in read_manifest(args.manifests / name)}
    row_ids = {r["row_id"] for r in coco_splits}
    clash = (train | held) & (coco_ids | row_ids)
    clash |= {r["pair_id"] for r in oi_pairs} & (row_ids | {r["pair_id"] for r in coco_pairs})
    if clash:
        problems.append(f"{len(clash)} Open Images id(s) collide with COCO ids")
    if problems:
        return {**record, "ok": False, "problems": problems}

    # 4. the merged files, prepared aside: COCO rows unchanged, then the Open Images train rows
    out, pending = args.out_dir, args.out_dir / PENDING
    by_id = {r["image_id"]: r for r in oi_reals}
    oi_real_rows = [{**by_id[i], "relative_path": f"{OI_REALS_LINK}/{OI_SET}/{i}.png"}
                    for i in sorted(train)]
    oi_split_rows = [{"row_id": i, "set_id": OI_SET, "kind": "real",
                      "content_id": by_id[i].get("oi_image_id", i), "group_id": i,
                      "stratum": OI_SET, "split": TRAIN} for i in sorted(train)]
    oi_split_rows += [{"row_id": r["pair_id"], "set_id": f"{OI_SET}_pair_{r['member']}",
                       "kind": "fake", "content_id": by_id[r["image_id"]].get("oi_image_id",
                                                                            r["image_id"]),
                       "group_id": r["image_id"], "stratum": OI_SET, "split": TRAIN}
                      for r in train_pairs]
    write_rows(pending / PAIRS_NAME, pair_cols, coco_pairs + train_pairs)
    write_rows(pending / SPLITS_NAME, split_cols, coco_splits + oi_split_rows)
    write_rows(pending / MANIFESTS_DIR / T_REAL, real_cols, coco_reals + oi_real_rows)
    for name in COPIED:
        shutil.copyfile(args.manifests / name, pending / MANIFESTS_DIR / name)
    merged = {name: file_sha256(pending / name) for name in MERGED_NAMES}
    record["merged"] = {"dir": str(out), "sha256": merged,
                        "added": {"oi_reals": len(oi_real_rows), "oi_pair_rows": len(train_pairs),
                                  "oi_split_rows": len(oi_split_rows)},
                        "held_out_excluded": {"oi_reals": len(held),
                                              "oi_pair_rows": len(oi_pairs) - len(train_pairs)}}

    # 5. the queue, from the committed tuning result
    choice_bytes = args.choice.read_bytes()
    choice = json.loads(choice_bytes)
    bank = load_bank_config(THESIS_DBP / "configs" / "decoder_bank.yaml")
    runs = load_run_configs(args.runs_config, bank)
    queue, queue_problems = resolve_queue(choice, json.loads(args.expect.read_text("utf-8")),
                                          runs)
    problems += queue_problems
    record["tuning_result"] = {"path": str(args.choice),
                               "sha256": hashlib.sha256(choice_bytes).hexdigest(),
                               "chosen": choice.get("chosen"), "chosen_p": choice.get("chosen_p")}
    record["queue"] = queue
    if problems:
        return {**record, "ok": False, "problems": problems}

    # 6. each queued run's pool, and every file it reads
    pools, relatives = {}, set()
    for protocol in dict.fromkeys(spec.partition(":")[0] for spec in queue):
        data, reals_index = training_data(args.root, pending, runs.by_id(protocol), bank, log)
        pools[protocol] = pool_by_source(data, train)
        relatives |= files_read(data, reals_index)
        if pools[protocol]["oi_reals"] != len(train):
            problems.append(f"{protocol}: {pools[protocol]['oi_reals']} Open Images reals in "
                            f"the pool, expected {len(train)}")
    record["pools"] = pools

    # 7. stage: each file read once; the Open Images train files hash-checked in that read
    expected = {f"{OI_REALS_LINK}/{OI_SET}/{i}.png": by_id[i]["sha256"] for i in sorted(train)}
    expected.update({r["relative_path"]: r["sha256"] for r in train_pairs})
    unread = sorted(set(expected) - relatives)
    if unread:
        problems.append(f"{len(unread)} Open Images train file(s) the runs would not read, "
                        f"e.g. {unread[:3]}")
    held_paths = [r for r in relatives
                  if r.startswith((f"{OI_REALS_LINK}/", f"{OI_PAIRS_LINK}/"))
                  and Path(r).stem in held]
    if held_paths:
        problems.append(f"{len(held_paths)} held-out Open Images file(s) among the runs' reads")
    if problems:
        return {**record, "ok": False, "problems": problems}
    staged = stage(sorted(relatives), expected, args.root, args.local_root, args.workers,
                   args.margin_gb, log)
    record["staged"] = staged
    record["files_verified"] = {"files": staged.get("verified", 0),
                                "expected": len(expected),
                                "n_sha256_mismatch": staged.get("n_sha256_mismatch", 0),
                                "sha256_mismatch": staged.get("sha256_mismatch", []),
                                "n_missing": staged["n_missing"]}
    if not staged["ok"]:
        problems.append(f"staging FAILED: {staged['reason']}")
    elif staged["verified"] != len(expected):
        problems.append(f"{staged['verified']} Open Images files hash-verified, "
                        f"expected {len(expected)}")
    if problems:
        return {**record, "ok": False, "problems": problems}

    # 8. everything passed: the merged files into place, and the queue
    for name in MERGED_NAMES:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        os.replace(pending / name, out / name)
    write_record(args.queue_out, {"queue": queue, "from": record["tuning_result"]})
    return {**record, "ok": True, "problems": []}


def check(args: argparse.Namespace, log: Any) -> dict[str, Any]:
    """Pre-flight: no held-out id anywhere, and the merged files are those ``build`` wrote."""
    built = json.loads(args.built.read_text(encoding="utf-8"))
    problems: list[str] = []
    if not built.get("ok"):
        return {"mode": "check", "ok": False, "problems": ["the build record is not ok"]}
    oi_reals = read_manifest(args.oi_manifest)
    held = {r["image_id"] for r in oi_reals if r.get("split") == HELD_OUT}
    train = {r["image_id"] for r in oi_reals if r.get("split") == TRAIN}
    out = args.out_dir
    for name, digest in built["merged"]["sha256"].items():
        if file_sha256(out / name) != digest:
            problems.append(f"{name} differs from the file build wrote")
    _, pairs = read_rows(out / PAIRS_NAME)
    _, splits = read_rows(out / SPLITS_NAME)
    _, reals = read_rows(out / MANIFESTS_DIR / T_REAL)
    found = {
        PAIRS_NAME: sum(1 for r in pairs if r["image_id"] in held),
        SPLITS_NAME: sum(1 for r in splits if r["row_id"] in held
                         or r["row_id"].split("__")[0] in held or r["group_id"] in held),
        f"{MANIFESTS_DIR}/{T_REAL}": sum(1 for r in reals if r["image_id"] in held),
    }
    if args.local_root is not None:
        local = [p for link in (OI_REALS_LINK, OI_PAIRS_LINK)
                 for p in (args.local_root / link).rglob("*.png") if p.stem in held]
        found["local copy"] = len(local)
    for name, count in found.items():
        if count:
            problems.append(f"{count} held-out Open Images row(s) or file(s) in {name}")
    counts = {"oi_reals": sum(1 for r in reals if r["image_id"] in train),
              "oi_pair_rows": sum(1 for r in pairs if r["image_id"] in train)}
    expected = {k: built["merged"]["added"][k] for k in counts}
    if counts != expected:
        problems.append(f"Open Images rows {counts}, build wrote {expected}")
    log.info("held-out rows or files found: %s; Open Images train rows: %s", found, counts)
    return {"mode": "check", "ok": not problems, "problems": problems,
            "held_out_ids": len(held), "held_out_rows_found": found, "oi_train_rows": counts}


def main() -> int:
    """Run one mode; write its record; exit 0 when it passes."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=("build", "check"))
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--oi-manifest", type=Path, required=True,
                        help="The committed Open Images reals manifest.")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where the merged files are (build: are written).")
    parser.add_argument("--oi-pairs-manifest", type=Path)
    parser.add_argument("--oi-pairs-digest", type=Path,
                        default=THESIS_DBP / "reports" / "kaggle_20261018" / "p3_oi_pairs"
                        / "p3_oi_pairs_manifest_digest.json")
    parser.add_argument("--pairs-manifest", type=Path, help="The COCO pairs manifest.")
    parser.add_argument("--splits", type=Path, help="The COCO split.")
    parser.add_argument("--manifests", type=Path, default=THESIS_DBP / "reports",
                        help="The committed reals and gen_matched manifests.")
    parser.add_argument("--choice", type=Path,
                        default=THESIS_DBP / "reports" / "p3_degradation_choice.json")
    parser.add_argument("--runs-config", type=Path,
                        default=THESIS_DBP / "configs" / "p3_runs.yaml")
    parser.add_argument("--root", type=Path,
                        help="The staged corpus root (links to the attached inputs, incl. "
                             "oi_reals/ and oi_pairs/): where every file is read from.")
    parser.add_argument("--local-root", type=Path,
                        help="Local-disk copy of every file the runs read; the trainer's root.")
    parser.add_argument("--margin-gb", type=float, default=2.0)
    parser.add_argument("--queue-out", type=Path,
                        default=THESIS_DBP / "reports" / QUEUE_NAME)
    parser.add_argument("--built", type=Path,
                        default=THESIS_DBP / "reports" / "p3_s2d_inputs.json")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True, help="The JSON record.")
    args = parser.parse_args()
    log = setup_logging("p3_s2d_inputs")
    if args.mode == "build":
        needed = ("oi_pairs_manifest", "pairs_manifest", "splits", "root", "local_root")
        missing = [f"--{n.replace('_', '-')}" for n in needed if getattr(args, n) is None]
        if missing:
            log.error("build needs %s", ", ".join(missing))
            return 2
        record = build(args, log)
    else:
        record = check(args, log)
    write_record(args.out, record)
    for problem in record["problems"]:
        log.error("%s", problem)
    log.info("%s: %s", args.mode, "PASSED" if record["ok"] else "FAILED")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
