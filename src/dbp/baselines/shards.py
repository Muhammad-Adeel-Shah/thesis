"""Splitting a row across GPUs, and resumable checkpoints (P4 option A, user 2026-11-09).

Execution only: the scores are exactly those the unsplit row would give, because every
baseline here is deterministic per image.

* **Split by pixels.** AEROBLADE's time is linear in the capped pixel count (MEASURED
  2026-11-09: 17.4-18.9 us per pixel on a T4), so images are dealt to the shards by
  longest-processing-time on that count, and both GPUs finish together. The deal depends
  only on the lists and the image sizes, so it is the same in every session.
* **Checkpoints are append-only.** Every ``checkpoint_every`` images a new part file is
  written (temp file, then ``os.replace`` onto a name that must not exist) and one line --
  its rows, sha256 and first and last ids -- is appended to the ledger. A part is never
  rewritten; a part that no longer matches its ledger line stops the run.
* **Every resume re-scores saved rows** (``resume_rescore`` of them, spread over what is
  saved) and stops if any value differs beyond fp32 rounding (:func:`within_fp32`).
  Values are stored with full float precision (``repr``), so that test is meaningful.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dbp.baselines.common import BaselineError
from dbp.data.manifest import file_sha256

#: Relative tolerance of the resume check: about 8 fp32 ulps, plus an absolute floor.
REL_TOL, ABS_TOL = 1e-6, 1e-9
LEDGER = "ledger.jsonl"


def capped_pixels(width: int, height: int, cap: int = 768, multiple: int = 8) -> int:
    """Pixels AEROBLADE processes: each side cut to ``cap``, then to a multiple of 8."""
    return (min(width, cap) // multiple * multiple) * (min(height, cap) // multiple * multiple)


def assign(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    """Deal ``items`` (each with ``pixels`` and ``order``) to ``n`` shards, balanced by pixels.

    Longest processing time first; ties go to the lower shard. Within a shard the items
    keep their original ``order``.
    """
    if n < 1:
        raise ValueError(f"need at least one shard, got {n}")
    loads = [0] * n
    shards: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for item in sorted(items, key=lambda i: (-int(i["pixels"]), int(i["order"]))):
        target = min(range(n), key=lambda k: (loads[k], k))
        shards[target].append(item)
        loads[target] += int(item["pixels"])
    return [sorted(s, key=lambda i: int(i["order"])) for s in shards]


def within_fp32(a: float, b: float) -> bool:
    """Equal up to fp32 rounding: ``|a - b| <= max(REL_TOL * max(|a|, |b|), ABS_TOL)``."""
    return math.isfinite(a) and math.isfinite(b) and \
        abs(a - b) <= max(REL_TOL * max(abs(a), abs(b)), ABS_TOL)


def resume_sample(count: int, k: int) -> list[int]:
    """``min(k, count)`` indices spread evenly over ``range(count)`` (first and last kept)."""
    if count <= 0 or k <= 0:
        return []
    if k >= count:
        return list(range(count))
    return sorted({round(i * (count - 1) / (k - 1)) for i in range(k)}) if k > 1 else [0]


@dataclass
class ShardStore:
    """The append-only checkpoint directory of one shard."""

    root: Path
    columns: list[str]

    @property
    def ledger(self) -> Path:
        """The ledger file."""
        return self.root / LEDGER

    def entries(self) -> list[dict[str, Any]]:
        """The ledger lines, in order."""
        if not self.ledger.is_file():
            return []
        return [json.loads(line) for line in self.ledger.read_text("utf-8").splitlines()
                if line.strip()]

    def load(self) -> list[dict[str, str]]:
        """Every saved row, after checking each part against its ledger line.

        Raises:
            BaselineError: On a missing, extra or changed part, or a column mismatch.
        """
        entries = self.entries()
        named = {e["part"] for e in entries}
        extra = sorted(p.name for p in self.root.glob("part_*.csv") if p.name not in named)
        if extra:
            raise BaselineError(f"{self.root}: part file(s) not in the ledger: {extra}")
        rows: list[dict[str, str]] = []
        for number, entry in enumerate(entries):
            path = self.root / entry["part"]
            if entry["part"] != f"part_{number:05d}.csv":
                raise BaselineError(f"{self.root}: ledger line {number} names {entry['part']}")
            if not path.is_file() or file_sha256(path) != entry["sha256"]:
                raise BaselineError(f"{path} is missing or no longer matches the ledger")
            with path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if list(reader.fieldnames or []) != self.columns:
                    raise BaselineError(f"{path}: columns {reader.fieldnames} != {self.columns}")
                part = list(reader)
            if len(part) != entry["rows"] or (part and (part[0]["image_id"] != entry["first"]
                                                        or part[-1]["image_id"] != entry["last"])):
                raise BaselineError(f"{path}: rows do not match the ledger line")
            rows += part
        return rows

    def append(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Write the next part and append its ledger line; never overwrites anything."""
        if not rows:
            raise BaselineError("an empty checkpoint part")
        self.root.mkdir(parents=True, exist_ok=True)
        number = len(self.entries())
        name = f"part_{number:05d}.csv"
        path = self.root / name
        if path.exists():
            raise BaselineError(f"{path} exists; checkpoints are never overwritten")
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=self.columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: repr(row[c]) if isinstance(row[c], float) else row[c]
                             for c in self.columns})
        tmp = path.with_name(name + ".tmp")
        tmp.write_text(buffer.getvalue(), encoding="utf-8", newline="\n")
        os.replace(tmp, path)
        entry = {"part": name, "rows": len(rows), "sha256": file_sha256(path),
                 "first": rows[0]["image_id"], "last": rows[-1]["image_id"],
                 "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with self.ledger.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")
        return entry


def merge(saved: list[dict[str, str]], lists: dict[str, list[dict[str, str]]]
          ) -> dict[str, list[dict[str, str]]]:
    """Every shard's saved rows, back in each list's committed order.

    Raises:
        BaselineError: Unless every (list, image id) is present exactly once.
    """
    seen: dict[tuple[str, str], dict[str, str]] = {}
    for row in saved:
        key = (row["list"], row["image_id"])
        if key in seen:
            raise BaselineError(f"merge: {key} saved twice")
        seen[key] = row
    out: dict[str, list[dict[str, str]]] = {}
    for name, rows in lists.items():
        missing = [r["image_id"] for r in rows if (name, r["image_id"]) not in seen]
        if missing:
            raise BaselineError(f"merge: {len(missing)} {name} row(s) not scored, e.g. "
                                f"{missing[:3]}")
        out[name] = [seen.pop((name, r["image_id"])) for r in rows]
    if seen:
        raise BaselineError(f"merge: {len(seen)} saved row(s) in no list, e.g. "
                            f"{sorted(seen)[:3]}")
    return out
