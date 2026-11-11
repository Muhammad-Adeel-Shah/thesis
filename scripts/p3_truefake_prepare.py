"""P3 TrueFake (secondary, descriptive), step 1: verify and stage the sample, check it against
our training reals, and plan the scoring (CPU; the rule of ``P3_PLAN.md``, fixed 2026-10-20).

1. **The sample is the fetched one:** the frozen sample and the committed fetch manifest
   (``reports/truefake/``) have their committed sha256 and list the same 16,000 members.
2. **The 4 models** (``S2-B`` s0, ``S2-B+deg``, ``S2-D`` s0, P2 ``full_bank``) are present and
   equal to their records in ``env/p3_final_expect.json`` (``p3_final_preflight.check_model``).
3. **Every member is copied once to local disk**, its sha256 checked against the manifest in
   the same read (``p3_s2d_inputs.stage``). Its size is read from the header; a member
   smaller than the 320 px evaluation crop cannot be scored by the protocol and is marked
   ``too_small`` (counted in the plan).
4. **Disjointness, before scoring** (the rule): every version of every sampled real is
   checked against our **training reals** -- COCO T-real (the 2,417 not excluded) and Open
   Images train (3,600) -- by sha256 (crops and originals) and by pHash
   (``dbp.data.phash.phash_file``, Hamming <= 6) against the crops' pHash, the Open Images
   originals' committed pHash, and **the COCO originals' pHash, computed here** from the
   attached COCO 2017, each original first checked against its committed sha256 (user
   decision 2026-10-28). A matching real is **dropped in all four versions, no backfill**,
   and counted per pool.
5. **Once only:** a model whose summary exists here or in an attached previous version is
   carried forward and not scored again.

Writes ``reports/p3_truefake_plan.json`` and the scoring list
``reports/p3_truefake_list.csv``. Images stay private: only names, sizes, hashes and counts
are written. Exit 0 when the plan is written, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.data.manifest import file_sha256, read_manifest  # noqa: E402
from dbp.data.phash import phash_file, phash_from_hex  # noqa: E402
from dbp.utils.logging import setup_logging  # noqa: E402
from p3_final_preflight import check_model  # noqa: E402
from p3_s2d_inputs import stage, write_record, write_rows  # noqa: E402

PLAN = Path("reports") / "p3_truefake_plan.json"
LIST = Path("reports") / "p3_truefake_list.csv"
#: Per-image scores: git-ignored, returned in the reports zip, kept in kaggle_raw only.
SCORES = Path("reports") / "p3_truefake_scores"
SUMMARIES = Path("reports") / "p3_truefake"
LIST_COLUMNS = ["image_id", "relative_path", "id", "share", "stratum", "label", "group",
                "sha256", "width", "height", "dropped", "too_small", "unreadable"]
PREFIX = "TrueFake/"
PHASH_BITS = 6
CROP = 320
_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def summary_path(root: Path, model: str) -> Path:
    """Where a model's TrueFake summary lives under a thesis_dbp root."""
    return root / SUMMARIES / f"{model}.json"


def is_complete(root: Path, model: str) -> bool:
    """A full (not a smoke) summary exists."""
    path = summary_path(root, model)
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("limit") is None


def scorable(row: dict[str, Any]) -> bool:
    """A list row the protocol scores: not dropped, not too small, decodable.

    The one rule the pre-flight, the scorer and the analysis share (values are ints when
    the pre-flight builds the row and strings when it is read back from the CSV).
    """
    return all(str(row[flag]) == "0" for flag in ("dropped", "too_small", "unreadable"))


def relative(member: str) -> str:
    """A member's path inside the published sample (the fetch wrote it without ``TrueFake/``)."""
    return member.removeprefix(PREFIX)


def hamming_min(values: np.ndarray, refs: np.ndarray, chunk: int = 256) -> np.ndarray:
    """For each 64-bit hash in ``values``, the smallest Hamming distance to any in ``refs``."""
    out = np.full(len(values), 64, dtype=np.int64)
    if len(refs) == 0:
        return out
    refs = refs.astype(np.uint64)
    for start in range(0, len(values), chunk):
        block = values[start: start + chunk].astype(np.uint64)
        xor = np.bitwise_xor(block[:, None], refs[None, :])
        bits = _POPCOUNT[xor.view(np.uint8)].reshape(len(block), len(refs), 8).sum(axis=2)
        out[start: start + len(block)] = bits.min(axis=1)
    return out


def references(code: Path, expect: dict[str, Any], coco_root: Path | None, workers: int,
               log: Any) -> tuple[set[str], np.ndarray, dict[str, Any]]:
    """Our training reals: sha256 of crops and originals; pHash of crops and originals."""
    t_real = [r for r in read_manifest(code / expect["t_real_manifest"])
              if not r.get("excluded_reason")]
    oi = [r for r in read_manifest(code / expect["oi_manifest"]) if r.get("split") == "train"]
    shas = {r["sha256"] for r in t_real} | {r["source_sha256"] for r in t_real}
    shas |= {r["sha256"] for r in oi} | {r["source_sha256"] for r in oi}
    hashes = [phash_from_hex(r["phash"]) for r in t_real]
    hashes += [phash_from_hex(r["phash"]) for r in oi]
    hashes += [phash_from_hex(r["source_phash"]) for r in oi]
    coco: dict[str, Any] = {"originals": len(t_real), "hashed": 0, "missing": [],
                            "sha256_mismatch": []}
    if coco_root is None:
        coco["skipped"] = "no --coco-root (tests only)"
    else:
        def one(row: dict[str, str]) -> tuple[str, int | None]:
            for split in expect["coco_splits"]:
                path = coco_root / split / row["source_relative_path"]
                if path.is_file():
                    if file_sha256(path) != row["source_sha256"]:
                        return "sha256_mismatch", None
                    return "ok", phash_file(path)
            return "missing", None

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(one, t_real))
        for row, (state, value) in zip(t_real, results, strict=True):
            if state == "ok":
                coco["hashed"] += 1
                hashes.append(value)
            else:
                coco[state].append(row["source_relative_path"])
        log.info("COCO originals: %d hashed, %d missing, %d sha256 mismatches",
                 coco["hashed"], len(coco["missing"]), len(coco["sha256_mismatch"]))
    record = {"t_real": len(t_real), "open_images_train": len(oi), "sha256": len(shas),
              "phash": len(hashes), "coco_originals": {
                  **coco, "missing": coco["missing"][:20],
                  "n_missing": len(coco["missing"]),
                  "sha256_mismatch": coco["sha256_mismatch"][:20],
                  "n_sha256_mismatch": len(coco["sha256_mismatch"])}}
    return shas, np.asarray(hashes, dtype=np.uint64), record


def decode_check(path: Path) -> tuple[int, int, str]:
    """(width, height, error): the header size, and why the full decode failed ("" if not).

    Every member is decoded completely, because a file can carry a valid header and still
    be cut short: TrueFake holds at least one truncated JPEG (Kaggle, 2026-10-31), byte-
    identical to the archive (CRC-32 and sha256 checked). Such a file is never repaired --
    ``LOAD_TRUNCATED_IMAGES`` would invent the missing pixels -- it is recorded instead.
    """
    from PIL import Image

    with Image.open(path) as handle:
        width, height = handle.size
        try:
            handle.convert("RGB")
        except (OSError, SyntaxError, ValueError) as error:
            return width, height, f"{type(error).__name__}: {error}"
    return width, height, ""


def disjointness(reals: list[dict[str, str]], local_root: Path, shas: set[str],
                 refs: np.ndarray, workers: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Which sampled real ids match a training real (any version); counts per pool."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        hashes = np.asarray(list(pool.map(
            lambda r: phash_file(local_root / relative(r["member"])), reals)), dtype=np.uint64)
    distance = hamming_min(hashes, refs)
    matched: dict[str, dict[str, Any]] = {}
    for row, d in zip(reals, distance, strict=True):
        by_sha = row["sha256"] in shas
        if by_sha or d <= PHASH_BITS:
            hit = matched.setdefault(row["id"], {"stratum": row["stratum"], "members": []})
            hit["members"].append({"member": row["member"], "sha256_match": by_sha,
                                   "phash_distance": int(d)})
    pools = {}
    for stratum in sorted({r["stratum"] for r in reals}):
        ids = {r["id"] for r in reals if r["stratum"] == stratum}
        pools[stratum] = {"ids": len(ids), "dropped_ids": sum(1 for i in ids if i in matched)}
    record = {"phash_bits": PHASH_BITS, "pools": pools, "matched": matched,
              "min_distance": int(distance.min()) if len(distance) else None,
              "versions_checked": len(reals)}
    return matched, record


def main() -> int:
    """Verify, stage, check disjointness; write the plan. Exit 0 when the plan is written."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--expect", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True,
                        help="The attached sample: the directory holding PreSocial/ etc.")
    parser.add_argument("--coco-root", type=Path, default=None,
                        help="COCO 2017 (holding train2017/); omitted only in tests.")
    parser.add_argument("--source", action="append", default=[], metavar="NAME=PATH",
                        help="train, tune, s2d (thesis_dbp roots), p2ref (dataset root).")
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--previous", action="append", type=Path, default=[],
                        help="thesis_dbp roots of this notebook's previous versions.")
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--out", type=Path, default=THESIS_DBP / PLAN)
    args = parser.parse_args()
    log = setup_logging("p3_truefake_prepare")
    expect = json.loads(args.expect.read_text(encoding="utf-8"))
    final = json.loads((THESIS_DBP / expect["final_expect"]).read_text(encoding="utf-8"))
    problems: list[str] = []
    record: dict[str, Any] = {"what": "P3 TrueFake scoring plan (secondary, descriptive)"}

    # 1. the sample is the fetched one
    for key in ("sample", "manifest"):
        digest = file_sha256(THESIS_DBP / expect[key])
        record[key] = {"path": expect[key], "sha256": digest}
        if digest != expect[f"{key}_sha256"]:
            problems.append(f"{expect[key]} is not the committed file ({digest})")
    if problems:
        return finish(args.out, {**record, "ok": False, "problems": problems}, log)
    manifest = read_manifest(THESIS_DBP / expect["manifest"])
    sample = {r["member"]: r for r in read_manifest(THESIS_DBP / expect["sample"])}
    if {r["member"] for r in manifest} != set(sample) or len(manifest) != expect["members"] \
            or len(sample) != expect["members"]:
        problems.append(f"the fetch manifest ({len(manifest)}) does not list exactly the "
                        f"sample's {len(sample)} members (expected {expect['members']})")
    if any(r["crc_ok"] != "True" for r in manifest):
        problems.append("a fetched member failed its CRC-32 check")

    # 2. the 4 models
    sources = {}
    for item in args.source:
        name, _, path = item.partition("=")
        sources[name] = Path(path)
    specs = {m["id"]: m for m in final["models"]}
    record["models"] = {}
    for model in expect["models"]:
        found, issues = check_model(specs[model], sources, final["section5"])
        record["models"][model] = found
        problems += issues
    if problems:
        return finish(args.out, {**record, "ok": False, "problems": problems}, log)

    # 3. every member staged once, its sha256 checked in the same read; its size
    expected = {relative(r["member"]): r["sha256"] for r in manifest}
    staged = stage(sorted(expected), expected, args.sample_root, args.local_root,
                   args.workers, expect.get("margin_gb", 2.0), log)
    record["staged"] = {**staged, "sample_root": str(args.sample_root)}
    if not staged["ok"] or staged["verified"] != len(expected):
        return finish(args.out, {**record, "ok": False, "problems": [
            f"staging FAILED: {staged.get('reason') or 'not every member verified'}"]}, log)
    members = [r["member"] for r in manifest]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        decoded = dict(zip(members, pool.map(
            lambda m: decode_check(args.local_root / relative(m)), members), strict=True))
    unreadable = {m: e for m, (_, _, e) in decoded.items() if e}
    log.info("decoded %d members; %d cannot be decoded: %s", len(members), len(unreadable),
             unreadable)

    # 4. disjointness of the sampled reals against our training reals (readable versions;
    #    an id with an unreadable version is excluded from every version anyway)
    shas, refs, ref_record = references(THESIS_DBP, expect, args.coco_root, args.workers, log)
    reals = [r for r in manifest if r["stratum"].startswith("Real/")
             and r["member"] not in unreadable]
    matched, record["disjointness"] = disjointness(reals, args.local_root, shas, refs,
                                                   args.workers)
    record["disjointness"]["references"] = ref_record
    coco = ref_record["coco_originals"]
    if coco.get("n_missing") or coco.get("n_sha256_mismatch"):
        problems.append(f"COCO originals not all verified: {coco['n_missing']} missing, "
                        f"{coco['n_sha256_mismatch']} sha256 mismatches")

    # the scoring list
    rows = []
    for r in manifest:
        width, height, _ = decoded[r["member"]]
        rows.append({"image_id": r["member"], "relative_path": relative(r["member"]),
                     "id": r["id"], "share": r["share"], "stratum": r["stratum"],
                     "label": "real" if r["stratum"].startswith("Real/") else "fake",
                     "group": r["stratum"].split("/")[1], "sha256": r["sha256"],
                     "width": width, "height": height, "dropped": int(r["id"] in matched),
                     "too_small": int(min(width, height) < CROP),
                     "unreadable": int(r["member"] in unreadable)})
    write_rows(THESIS_DBP / LIST, LIST_COLUMNS, rows)
    record["list"] = {"path": str(THESIS_DBP / LIST), "rows": len(rows),
                      "sha256": file_sha256(THESIS_DBP / LIST),
                      "dropped_members": sum(r["dropped"] for r in rows),
                      "too_small_members": sum(r["too_small"] for r in rows),
                      "too_small": [r["image_id"] for r in rows if r["too_small"]][:20],
                      "unreadable_members": len(unreadable),
                      "unreadable": [{"member": m, "error": e}
                                     for m, e in sorted(unreadable.items())],
                      "scored_members": sum(1 for r in rows if scorable(r))}
    if problems:
        return finish(args.out, {**record, "ok": False, "problems": problems}, log)

    # 5. once only
    carried = []
    for model in expect["models"]:
        if is_complete(THESIS_DBP, model):
            continue
        for root in args.previous:
            if is_complete(root, model):
                shutil.copytree(root / SCORES / model, THESIS_DBP / SCORES / model,
                                dirs_exist_ok=True)
                summary_path(THESIS_DBP, model).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(summary_path(root, model), summary_path(THESIS_DBP, model))
                carried.append(model)
                log.info("carried %s forward from %s", model, root)
                break
    due = [m for m in expect["models"] if not is_complete(THESIS_DBP, m)]
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    shares = {d: due[i::len(devices)] for i, d in enumerate(devices)}
    record.update({"stage": "truefake" if due else "complete", "due": due, "carried": carried,
                   "jobs": [f"{d}:{','.join(s)}" for d, s in shares.items() if s],
                   "local_root": str(args.local_root)})
    log.info("TrueFake: %d model(s) due %s; dropped real ids %s; too small %d; scored %d",
             len(due), due, {k: v["dropped_ids"] for k, v in
                             record["disjointness"]["pools"].items()},
             record["list"]["too_small_members"], record["list"]["scored_members"])
    return finish(args.out, {**record, "ok": True, "problems": []}, log)


def finish(out: Path, record: dict[str, Any], log: Any) -> int:
    """Write the plan record; 0 when it is ok."""
    write_record(out, record)
    for problem in record["problems"]:
        log.error("%s", problem)
    log.info("TrueFake pre-flight: %s", "PASSED" if record["ok"] else "FAILED")
    return 0 if record["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
