"""P3 section 4b: the scoring lists for degradation tuning -- exactly the rows the rule reads.

User decision 2026-10-21: tuning scores only the rows the committed rule
(``p3_tune_degradation.py``, unchanged) uses, instead of the whole dev panel and the whole
validation view. The rows are selected **with the rule's own definitions** -- its
``panel_rows``, ``FPR_POOLS`` and ``unseen_slices`` and P2's ``slice_key`` -- never
re-derived here:

* ``p3_tune_panel_rows.csv`` -- from the dev panel list, the real pool of every view the
  rule reads (E4 and E5 for the FPR, the views of the four unseen slices for the AUC
  guard) and the fakes of the four unseen slices. 6,717 of 11,775 rows.
* ``p3_tune_val_reals.csv`` -- from the S1 list, the ``val`` reals: the 491 COCO
  validation reals the k = 0 threshold is fitted on.

Rows are written verbatim (same columns, same order), so a row resolves to the same file.
The rule receives ``p3_tune_panel_rows.csv`` as its ``--panel-list`` and computes every
number on the same rows as before; its ``--val-list`` stays the full S1 list.

With ``--expect``, each file's sha256 and row count must equal the committed values
(``derived_lists`` in the expect file): a regenerated input is checked against the
committed reference. Exit 0 when the lists are written and match, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

THESIS_DBP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(THESIS_DBP / "src"))
sys.path.insert(0, str(THESIS_DBP / "scripts"))

from dbp.utils.logging import setup_logging  # noqa: E402

PANEL_ROWS = "p3_tune_panel_rows.csv"
VAL_REALS = "p3_tune_val_reals.csv"


def rule_rows(panel_list: Path, e3_manifest: Path) -> tuple[list[str], list[dict[str, str]]]:
    """The dev panel rows the committed rule reads, verbatim, with the list's header."""
    from p2_analyze import slice_key
    from p3_tune_degradation import FPR_POOLS, panel_rows, unseen_slices

    slices = set(unseen_slices())
    real_views = set(FPR_POOLS) | {name.split("/", 1)[0] for name in slices}
    with panel_list.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        source = list(reader)
        columns = list(reader.fieldnames or [])
    keep = []
    for raw, row in zip(source, panel_rows(panel_list, e3_manifest), strict=True):
        # A real counts for every view the rule reads (FPR pool or a slice's real pool); a
        # fake only for its own unseen slice.
        if (row["label"] == "real" and row["view"] in real_views) or \
                (row["label"] == "fake" and slice_key(row) in slices):
            keep.append(raw)
    return columns, keep


def val_reals(val_list: Path) -> tuple[list[str], list[dict[str, str]]]:
    """The ``val`` reals, as the rule defines them (view ``val``, label ``real``)."""
    with val_list.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [r for r in reader if r.get("view") == "val" and r["label"] == "real"]
        return list(reader.fieldnames or []), rows


def write_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> str:
    """Atomic CSV write with LF line endings; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    """Write both lists; with ``--expect``, check them against the committed values."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--panel-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--e3-manifest", type=Path, default=THESIS_DBP / "reports"
                        / "p2_panel_sets" / "p2_panel_e3_matched.csv")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expect", type=Path, default=None,
                        help="An expect file whose derived_lists holds sha256 and rows.")
    args = parser.parse_args()
    log = setup_logging("p3_tune_lists")
    written: dict[str, dict[str, Any]] = {}
    for name, (columns, rows) in ((PANEL_ROWS, rule_rows(args.panel_list, args.e3_manifest)),
                                  (VAL_REALS, val_reals(args.val_list))):
        digest = write_rows(args.out_dir / name, columns, rows)
        written[name] = {"rows": len(rows), "sha256": digest}
        log.info("%s: %d rows, sha256 %s", name, len(rows), digest)
    problems = []
    if args.expect is not None:
        expected = json.loads(args.expect.read_text(encoding="utf-8"))["derived_lists"]
        for name, spec in expected.items():
            got = written.get(name, {})
            if got.get("rows") != spec["rows"] or got.get("sha256") != spec["sha256"]:
                problems.append(f"{name}: {got} differs from the committed {spec}")
    for problem in problems:
        log.error("%s", problem)
    log.info("tuning lists: %s", "FAILED" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
