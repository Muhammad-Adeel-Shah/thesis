"""Panel B: P3's once-only test panel (``docs/P3_PLAN.md`` section 1, P3 pre-registration §2).

Panel B is drawn from the evaluation images the P2 panel did **not** use, and every rule
here is a hard rule. A row enters Panel B only if:

* its image id and its **source image** are not in the P2 panel. The source image is the
  COCO id for E7, whose six VAE subsets and reals are all built from the same COCO val2017
  images, and the distributed file path everywhere else;
* its shorter side is at least the evaluation crop (I-047), label-blind, no backfill;
* it is not byte-identical (sha256) and not a pHash near-duplicate (≤ 6 bits) of any
  reference image: the P2 panel, the training corpus (T-real, T-ctrl, the SD1.5
  generations and their matched JPEGs, the bank reconstructions by sha256);
* it is not a near-duplicate of a row already accepted into Panel B;
* for E7, its COCO id is not already used by another Panel B row.

Rows are visited in a seeded order (``blake2b`` of seed, view and image id), and each is
assigned to the first slot of its group that still has room, so the draw is reproducible
from the inputs alone. A slot that cannot be filled is reported as a shortfall, never
backfilled from another rule.

Every rejection is counted by reason, so the summary says exactly why the pool shrank.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from dbp.data.phash import DUPLICATE_THRESHOLD_BITS, phash_from_hex
from dbp.eval.panel import EVAL_CROP, shorter_side

#: The P3 seed (the project seed, as for every P2 draw).
PANEL_B_SEED = 2026
PANEL_B_SALT = "panel_b"

#: View -> the rebuilt P1 manifest it is drawn from, and its slots.
#: A slot is (name, label, column, value, capacity, role); capacity ``None`` = every
#: eligible row. Reals list their calibration slot first, so the first rows of the seeded
#: order become the calibration pool (pre-registration §2; E5's pool is 500).
PANEL_B_VIEWS: dict[str, dict[str, Any]] = {
    "e3_matched": {
        "manifest": "p1_eval_e3_raw_jpeg_matched_manifest.csv",
        "slots": [
            ("GPT-Image2-fake", "fake", "source_label", "GPT-Image2-fake", None, "test"),
            ("Nano-Banana2-fake", "fake", "source_label", "Nano-Banana2-fake", None, "test"),
            ("real-calibration", "real", None, None, 50, "calibration"),
            ("real-test", "real", None, None, None, "test"),
        ],
    },
    "e4_wildrf_matched": {
        "manifest": "p1_eval_e4_wildrf_test_jpeg_matched_manifest.csv",
        "slots": [
            ("fake", "fake", None, None, None, "test"),
            ("real-calibration", "real", None, None, 100, "calibration"),
            ("real-test", "real", None, None, None, "test"),
        ],
    },
    "e5_chameleon_pipeline": {
        "manifest": "p1_eval_e5_chameleon_pipeline_matched_manifest.csv",
        "slots": [
            ("fake", "fake", None, None, 2000, "test"),
            ("real-calibration", "real", None, None, 500, "calibration"),
            ("real-test", "real", None, None, 2000, "test"),
        ],
    },
    "e7_dda_coco": {
        "manifest": "p1_eval_e7_dda_coco_manifest.csv",
        "one_source_once": True,
        "slots": [
            *[(vae, "fake", "source_vae", vae, 300, "test") for vae in (
                "FLUX.1", "sd-vae-ft-ema", "sd-vae-ft-mse", "sdxl-vae",
                "stable-diffusion-2-1", "stable-diffusion-3.5-large")],
            ("real-calibration", "real", None, None, 100, "calibration"),
            ("real-test", "real", None, None, 300, "test"),
        ],
    },
    "e8_evalgen": {
        "manifest": "p1_eval_e8_evalgen_manifest.csv",
        "slots": [(g, "fake", "generator", g, 250, "test")
                  for g in ("Flux", "GoT", "Infinity", "NOVA", "OmniGen")],
    },
}

#: E7 FLUX.1 carries the 4:2:0 shortcut (§6d), so its test fakes and E7's test reals get a
#: luma companion, as in P2.
PANEL_B_LUMA = {"e7_dda_coco": ("FLUX.1", "real")}


class PanelBError(ValueError):
    """A Panel B input or rule is violated."""


def coco_id(row: Mapping[str, Any]) -> str:
    """E7's source image: the COCO file stem, e.g. ``000000488075``."""
    return PurePosixPath(str(row["relative_path"]).replace("\\", "/")).stem


def e3_number(row: Mapping[str, Any]) -> tuple[str, str]:
    """E3's (content folder, file number), e.g. ``("Hybrid Images", "181")``.

    Used only for the descriptive ``e3_number_in_p2`` flag (user decision 2026-10-11).

    Raises:
        PanelBError: If the path is not ``.../<folder>/<label folder>/<number>.<ext>``.
    """
    parts = PurePosixPath(str(row["relative_path"]).replace("\\", "/")).parts
    if len(parts) < 3:
        raise PanelBError(f"E3 path {row['relative_path']!r} has no content folder")
    return parts[-3], PurePosixPath(parts[-1]).stem


def source_id(view: str, row: Mapping[str, Any]) -> str:
    """The source-image identity Panel B keeps disjoint from the P2 panel."""
    if view == "e7_dda_coco":
        return f"coco:{coco_id(row)}"
    return f"path:{str(row['relative_path']).replace(chr(92), '/')}"


def order_key(view: str, image_id: str, seed: int = PANEL_B_SEED) -> int:
    """Seeded, platform-independent sort key."""
    text = f"{PANEL_B_SALT}:{seed}:{view}:{image_id}".encode()
    return int.from_bytes(hashlib.blake2b(text, digest_size=8).digest(), "big")


_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_to_all(value: int, hashes: np.ndarray) -> np.ndarray:
    """Hamming distance from one 64-bit hash to every hash in ``hashes`` (uint64)."""
    if hashes.size == 0:
        return np.zeros(0, dtype=np.uint8)
    xor = np.bitwise_xor(hashes, np.uint64(value))
    return _POPCOUNT[xor.view(np.uint8).reshape(-1, 8)].sum(axis=1)


@dataclass
class HashIndex:
    """sha256 and pHash references, each tagged with where it came from."""

    sha: dict[str, str] = field(default_factory=dict)
    phash: list[int] = field(default_factory=list)
    phash_source: list[str] = field(default_factory=list)
    _array: np.ndarray | None = None

    def add(self, source: str, sha256: str | None, phash_hex: str | None) -> None:
        """Register one reference image."""
        if sha256:
            self.sha.setdefault(sha256.strip().lower(), source)
        if phash_hex:
            self.phash.append(phash_from_hex(phash_hex.strip()))
            self.phash_source.append(source)
            self._array = None

    def add_rows(self, source: str, rows: Iterable[Mapping[str, Any]],
                 sha_column: str = "sha256", phash_column: str | None = "phash") -> int:
        """Register a whole manifest; returns how many rows were added."""
        count = 0
        for row in rows:
            self.add(source, row.get(sha_column), row.get(phash_column) if phash_column else None)
            count += 1
        return count

    def match(self, sha256: str, phash_hex: str | None,
              threshold: int = DUPLICATE_THRESHOLD_BITS) -> str | None:
        """``"sha:<source>"``, ``"phash:<source>"`` or ``None``."""
        hit = self.sha.get(str(sha256).strip().lower())
        if hit:
            return f"sha:{hit}"
        if not phash_hex or not self.phash:
            return None
        if self._array is None:
            self._array = np.array(self.phash, dtype=np.uint64)
        distances = hamming_to_all(phash_from_hex(phash_hex), self._array)
        close = np.flatnonzero(distances <= threshold)
        return f"phash:{self.phash_source[int(close[0])]}" if close.size else None


@dataclass
class Slot:
    """One quota of the draw."""

    name: str
    label: str
    column: str | None
    value: str | None
    capacity: int | None
    role: str
    rows: list[dict[str, Any]] = field(default_factory=list)

    def matches(self, row: Mapping[str, Any]) -> bool:
        """Whether a row belongs to this slot's group (label and stratum)."""
        if str(row.get("label")) != self.label:
            return False
        return self.column is None or str(row.get(self.column)) == self.value

    def full(self) -> bool:
        """Whether the slot has reached its capacity."""
        return self.capacity is not None and len(self.rows) >= self.capacity


def draw_view(
    view: str,
    rows: Sequence[Mapping[str, Any]],
    p2_rows: Sequence[Mapping[str, Any]],
    references: HashIndex,
    accepted: HashIndex,
    slots_spec: Sequence[tuple] | None = None,
    min_side: int = EVAL_CROP,
    seed: int = PANEL_B_SEED,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw one view of Panel B.

    Args:
        view: A key of :data:`PANEL_B_VIEWS`.
        rows: The view's rebuilt P1 manifest rows.
        p2_rows: The P2 panel's rows of the same view (all 12,150 drawn rows, including
            those I-047 later removed): their ids and source images are excluded.
        references: Training-corpus and P2-panel hashes.
        accepted: Panel B rows accepted so far, in every view (updated in place).
        slots_spec: Override of the view's slots (the dry-run fixture only).
        min_side: The evaluation crop.
        seed: The draw seed.

    Returns:
        ``(panel rows, summary)``.
    """
    spec = PANEL_B_VIEWS[view]
    slots = [Slot(*s) for s in (slots_spec or spec["slots"])]
    one_source_once = bool(spec.get("one_source_once"))
    p2_ids = {str(r["image_id"]) for r in p2_rows}
    p2_sources = {source_id(view, r) for r in p2_rows}
    p2_e3_numbers = {e3_number(r) for r in p2_rows} if view == "e3_matched" else set()
    used_sources: set[str] = set()
    reasons: Counter = Counter()

    ids = [str(r["image_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise PanelBError(f"{view}: duplicate image ids in the manifest")
    ordered = sorted(rows, key=lambda r: order_key(view, str(r["image_id"]), seed))
    for row in ordered:
        group = [s for s in slots if s.matches(row)]
        if not group:
            reasons["no slot for its label/stratum"] += 1
            continue
        open_slots = [s for s in group if not s.full()]
        if not open_slots:
            reasons["slot full"] += 1
            continue
        if str(row["image_id"]) in p2_ids:
            reasons["in the P2 panel (id)"] += 1
            continue
        source = source_id(view, row)
        if source in p2_sources:
            reasons["source image in the P2 panel"] += 1
            continue
        if one_source_once and source in used_sources:
            reasons["source image already in Panel B"] += 1
            continue
        side = shorter_side(row)
        if side is None or side < min_side:
            reasons[f"shorter side < {min_side} or unmeasured"] += 1
            continue
        sha, phash = str(row.get("sha256") or ""), str(row.get("phash") or "")
        if len(sha) != 64:
            raise PanelBError(f"{view}/{row['image_id']}: no sha256 in the manifest")
        hit = references.match(sha, phash or None)
        if hit:
            kind, origin = hit.split(":", 1)
            reasons[f"duplicate of a reference ({kind}: {origin})"] += 1
            continue
        if accepted.match(sha, phash or None):
            reasons["duplicate of a Panel B row"] += 1
            continue
        slot = open_slots[0]
        out = dict(row)
        out.update({
            "panel_b_view": view, "panel_b_slot": slot.name, "role": slot.role,
            "order_key": f"{order_key(view, str(row['image_id']), seed):016x}",
        })
        if view == "e3_matched":
            out["e3_number_in_p2"] = "true" if e3_number(row) in p2_e3_numbers else "false"
        slot.rows.append(out)
        accepted.add(f"panel_b:{view}", sha, phash or None)
        used_sources.add(source)
    panel = [r for s in slots for r in s.rows]
    summary = {
        "view": view,
        "manifest_rows": len(rows),
        "p2_rows_excluded_from": len(p2_rows),
        "slots": {s.name: {"label": s.label, "role": s.role, "capacity": s.capacity,
                           "filled": len(s.rows),
                           "shortfall": (s.capacity - len(s.rows)) if s.capacity else 0}
                  for s in slots},
        "rejections": dict(sorted(reasons.items())),
        "rows": len(panel),
    }
    if view == "e3_matched":
        flags = Counter((r["label"], r["source_label"], r["e3_number_in_p2"]) for r in panel)
        summary["e3_number_in_p2"] = {f"{lab}|{src}|{flag}": n
                                      for (lab, src, flag), n in sorted(flags.items())}
    return panel, summary
