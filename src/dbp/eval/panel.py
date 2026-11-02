"""The evaluation panel: one fixed set of images that every stream is scored on.

Complementarity is a statement about *which images* each stream gets wrong, so it can
only be computed where the streams saw the same images. Scoring S1 on all 34,969 E7
rows and S3 on a cheaper subset would make every double-fault rate in the report a
comparison of two different sets.

So the panel is drawn once, seeded, stratified, and shared. S1 and S2 are additionally
scored on the **full** views because they are cheap; those numbers are reported
separately and labelled, and they never enter a complementarity measure.

Two capping modes, because the P2 plan uses both:

* :func:`sample_per_stratum` -- "at most N per subset", used where each subset is a
  slice in its own right (E7's source VAEs, E8's generators, E5's two labels);
* :func:`sample_total` -- "at most N overall, stratified by X", used where the cap is
  on the slice as a whole (E3, E4). The allocation is proportional with largest
  remainders, so a rare content type is not rounded out of existence.

Sampling is by a stable hash of the image id, never by position: a view rebuilt in a
different file order yields the same panel.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

#: Panel caps approved for P2 (B1), at GPU_BUDGET_HOURS = 32.
PANEL_CAPS: dict[str, dict[str, Any]] = {
    "e7_dda_coco": {"mode": "per_stratum", "cap": 500, "by": ("source_vae",)},
    "e8_evalgen": {"mode": "per_stratum", "cap": 250, "by": ("generator",)},
    # E3's manifest names these `source_label` (GPT-Image2-fake / Nano-Banana2-fake /
    # real) and `subset` (Texture / Structure / Hybrid Images). They are not called
    # `generator` and `content_type`; those were a guess and the panel refused them,
    # which is what `stratum_of` raising rather than defaulting is for.
    "e3_matched": {"mode": "total", "cap": 1500, "by": ("source_label", "subset")},
    "e4_wildrf_matched": {"mode": "total", "cap": 1500, "by": ("platform", "label")},
    "e5_chameleon_pipeline": {"mode": "per_stratum", "cap": 2000, "by": ("label",)},
    "t_ctrl_diagnostic": {"mode": "per_stratum", "cap": 200, "by": ("label",)},
}

#: Sets that exist only to be measured, never to be trained on (P2 prompt B1).
#:
#: ``t_ctrl`` is the near-lossless control: its reconstructions are the JPEG-reliance
#: diagnostic. A stream trained on them cannot be read as evidence about them, so every
#: training list filters on this and the 400-row diagnostic panel is drawn from rows no
#: stream has seen. The panel itself is unaffected -- the rows are the same rows; what
#: changes is that they no longer also appear in the training lists.
DIAGNOSTIC_ONLY_SETS = frozenset({"t_ctrl"})

#: Derived luma-only views (pre-registration §8).
#:
#: **Deliberately not a member of PANEL_CAPS.** ``PANEL_CAPS`` is what
#: ``p2_materialise_panel.py`` iterates and what the 12,150-row assertion and the panel
#: digest are defined over; adding a seventh key would demand a
#: ``p2_panel_e7_dda_coco_luma.csv`` that no sampler writes and would change the frozen
#: digest. The luma companion is a *transform of rows already in the panel*, so it is
#: declared here instead, at the same level and with the same explicitness.
#:
#: ``replaces_in_gate`` is the §8 rule in machine-readable form: the standard FLUX.1
#: slice is reported but does **not** enter M1 or M2, because on it a rule reading only
#: chroma subsampling is a perfect classifier (§6d). The luma-only slice takes its place.
LUMA_VIEWS: dict[str, dict[str, Any]] = {
    "e7_dda_coco_luma": {
        "parent": "e7_dda_coco",
        "subsets": ("FLUX.1", "real"),
        "replaces_in_gate": ("e7_dda_coco/FLUX.1",),
    },
}

#: Appended to an image id to name its luma companion.
#:
#: The companion shares its parent's pixels-before-transform and therefore its id, but
#: the two must be distinguishable: ``p2_probe_features.py`` writes one ``.npz`` per
#: bank member keyed by image id, so a collision would silently overwrite the standard
#: row's twelve reconstructions with the greyscale ones.
LUMA_ID_SUFFIX = "__luma"


#: The evaluation crop every stream reads at (D-045). A panel row whose shorter side is
#: below this cannot be centre-cropped to it, so it cannot be scored at all.
EVAL_CROP = 320

#: Columns a manifest records its pixel dimensions in.
SIZE_COLUMNS = ("width", "height")


class PanelError(ValueError):
    """Raised when a panel cannot be drawn as specified."""


def shorter_side(row: Mapping[str, Any]) -> int | None:
    """Length of the row's shorter side, or ``None`` when it is not recorded.

    ``None`` is not zero: a manifest without ``width``/``height`` says nothing about the
    image, and treating that as "too small" would silently empty a view. The caller
    decides, and :func:`filter_by_min_side` keeps such rows and counts them.
    """
    try:
        sides = [int(float(row[column])) for column in SIZE_COLUMNS]
    except (KeyError, TypeError, ValueError):
        return None
    return min(sides) if all(side > 0 for side in sides) else None


def filter_by_min_side(
    rows: Sequence[Mapping[str, Any]], min_side: int
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], int]:
    """Split rows into those that survive a ``min_side`` centre crop and those that do not.

    The test is on pixels alone -- never on the label -- so the filter cannot become a
    way of removing inconvenient rows from one class. Where it still falls unevenly
    across classes, that is a property of the source set and is reported per view and
    per label rather than corrected: a backfill would be a second, unseeded draw.

    Args:
        rows: Manifest rows.
        min_side: Minimum shorter side, in pixels. ``0`` disables the filter.

    Returns:
        ``(kept, dropped, unmeasured)`` -- ``unmeasured`` counts rows that record no
        size and are therefore kept.
    """
    if min_side <= 0:
        return list(rows), [], 0
    kept: list[Mapping[str, Any]] = []
    dropped: list[Mapping[str, Any]] = []
    unmeasured = 0
    for row in rows:
        side = shorter_side(row)
        if side is None:
            unmeasured += 1
            kept.append(row)
        elif side >= min_side:
            kept.append(row)
        else:
            dropped.append(row)
    return kept, dropped, unmeasured


def luma_image_id(image_id: str) -> str:
    """Id of ``image_id``'s luma companion.

    Raises:
        PanelError: If ``image_id`` is already a luma id. Double-suffixing would make a
            row that no parent lookup can resolve.
    """
    if is_luma_id(image_id):
        raise PanelError(f"{image_id!r} is already a luma id")
    return f"{image_id}{LUMA_ID_SUFFIX}"


def is_luma_id(image_id: str) -> bool:
    """Whether this id names a luma companion."""
    return image_id.endswith(LUMA_ID_SUFFIX)


def parent_image_id(image_id: str) -> str:
    """The panel row a luma companion was derived from.

    Raises:
        PanelError: If ``image_id`` is not a luma id, because returning it unchanged
            would let a standard row be joined against luma metadata unnoticed.
    """
    if not is_luma_id(image_id):
        raise PanelError(f"{image_id!r} is not a luma id")
    return image_id[: -len(LUMA_ID_SUFFIX)]


def gate_slices_replaced_by_luma() -> frozenset[str]:
    """Slice names that a luma view displaces in M1 and M2 (pre-registration §8)."""
    return frozenset(
        name for spec in LUMA_VIEWS.values() for name in spec["replaces_in_gate"]
    )


def _key(image_id: str, salt: str) -> int:
    """Stable 64-bit sort key for an image, independent of file order or platform."""
    digest = hashlib.blake2b(
        f"{salt}:{image_id}".encode(), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big")


def stratum_of(row: Mapping[str, Any], by: Sequence[str]) -> str:
    """Stratum label of a row, as ``"a|b"`` over the named columns.

    Raises:
        PanelError: If a column is missing. Falling back to ``"unknown"`` would pool
            strata that the plan says to keep apart.
    """
    missing = [column for column in by if column not in row]
    if missing:
        raise PanelError(f"row has no {missing}; cannot form its stratum")
    return "|".join(str(row[column]) for column in by)


def allocate(
    sizes: Mapping[str, int], total: int, min_per_stratum: int = 1
) -> dict[str, int]:
    """Split ``total`` across strata proportionally, by largest remainders.

    A stratum never receives more than it has, and the leftover from any such stratum
    is redistributed rather than lost, so the panel reaches ``total`` whenever the
    population allows it.

    ``min_per_stratum`` is the one departure from strict proportionality, and it is
    deliberate: under pure proportional allocation a stratum holding 0.3% of a view
    rounds to zero, and D-044 requires E3 to be reported **per content type**, which a
    stratum that is not in the panel makes impossible. The floor keeps every non-empty
    stratum present; the summary records how few rows it got, so nobody reads a
    three-row slice as a result.

    Args:
        sizes: Stratum -> how many rows it has.
        total: Total to allocate.
        min_per_stratum: Floor for any non-empty stratum, budget permitting.

    Returns:
        Stratum -> allocation.

    Raises:
        PanelError: If ``total`` is negative.
    """
    if total < 0:
        raise PanelError(f"total must be non-negative, got {total}")
    population = sum(sizes.values())
    if population <= total:
        return dict(sizes)

    remaining = total
    quota: dict[str, int] = {stratum: 0 for stratum in sizes}
    open_strata = {s: n for s, n in sizes.items() if n > 0}
    # Iterate because capping a small stratum frees budget for the others.
    while remaining > 0 and open_strata:
        pool = sum(open_strata.values())
        exact = {s: remaining * n / pool for s, n in open_strata.items()}
        floors = {s: int(v) for s, v in exact.items()}
        for stratum, value in floors.items():
            quota[stratum] += value
        given = sum(floors.values())
        leftover = remaining - given
        order = sorted(
            open_strata, key=lambda s: (-(exact[s] - floors[s]), _key(s, "allocate"))
        )
        for stratum in order[:leftover]:
            quota[stratum] += 1
        remaining = 0
        # Claw back anything over a stratum's population and re-open the rest.
        spilled = 0
        for stratum in list(open_strata):
            if quota[stratum] > sizes[stratum]:
                spilled += quota[stratum] - sizes[stratum]
                quota[stratum] = sizes[stratum]
            if quota[stratum] >= sizes[stratum]:
                open_strata.pop(stratum)
        remaining = spilled

    # Lift any non-empty stratum off zero, paying for it from the largest allocations.
    starved = [s for s, n in sizes.items() if n > 0 and quota[s] < min(min_per_stratum, n)]
    for stratum in sorted(starved, key=lambda s: (sizes[s], _key(s, "floor"))):
        need = min(min_per_stratum, sizes[stratum]) - quota[stratum]
        while need > 0:
            donor = max(
                (s for s in quota if s != stratum and quota[s] > min_per_stratum),
                key=lambda s: (quota[s], _key(s, "donor")),
                default=None,
            )
            if donor is None:
                break
            quota[donor] -= 1
            quota[stratum] += 1
            need -= 1
    return quota


def _take(
    rows: Sequence[Mapping[str, Any]], n: int, salt: str, id_key: str
) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: _key(str(row[id_key]), salt))
    return [dict(row) for row in ordered[:n]]


def sample_per_stratum(
    rows: Sequence[Mapping[str, Any]],
    by: Sequence[str],
    cap: int,
    set_id: str,
    id_key: str = "image_id",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take at most ``cap`` rows from every stratum.

    Args:
        rows: The view's manifest rows.
        by: Columns forming the stratum.
        cap: Maximum rows per stratum.
        set_id: Used as the sampling salt, so two views never draw the same subset.
        id_key: Column holding the image id.

    Returns:
        ``(panel_rows, summary)``.

    Raises:
        PanelError: If ``rows`` is empty or ``cap`` is not positive.
    """
    if not rows:
        raise PanelError(f"{set_id}: no rows to draw a panel from")
    if cap <= 0:
        raise PanelError(f"{set_id}: cap must be positive, got {cap}")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(stratum_of(row, by), []).append(row)

    panel: list[dict[str, Any]] = []
    per_stratum: dict[str, dict[str, int]] = {}
    for stratum in sorted(grouped):
        available = grouped[stratum]
        taken = _take(available, min(cap, len(available)), f"{set_id}:{stratum}", id_key)
        panel.extend(taken)
        per_stratum[stratum] = {"available": len(available), "taken": len(taken)}

    return panel, {
        "set_id": set_id,
        "mode": "per_stratum",
        "cap": cap,
        "stratified_by": list(by),
        "population": len(rows),
        "panel": len(panel),
        "by_stratum": per_stratum,
    }


def sample_total(
    rows: Sequence[Mapping[str, Any]],
    by: Sequence[str],
    total: int,
    set_id: str,
    id_key: str = "image_id",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Take ``total`` rows overall, allocated proportionally across the strata.

    Args:
        rows: The view's manifest rows.
        by: Columns forming the stratum.
        total: Total panel size.
        set_id: Sampling salt.
        id_key: Column holding the image id.

    Returns:
        ``(panel_rows, summary)``.

    Raises:
        PanelError: If ``rows`` is empty or ``total`` is not positive.
    """
    if not rows:
        raise PanelError(f"{set_id}: no rows to draw a panel from")
    if total <= 0:
        raise PanelError(f"{set_id}: total must be positive, got {total}")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(stratum_of(row, by), []).append(row)

    quota = allocate({s: len(r) for s, r in grouped.items()}, total)
    panel: list[dict[str, Any]] = []
    per_stratum: dict[str, dict[str, int]] = {}
    for stratum in sorted(grouped):
        taken = _take(grouped[stratum], quota[stratum], f"{set_id}:{stratum}", id_key)
        panel.extend(taken)
        per_stratum[stratum] = {"available": len(grouped[stratum]), "taken": len(taken)}

    return panel, {
        "set_id": set_id,
        "mode": "total",
        "cap": total,
        "stratified_by": list(by),
        "population": len(rows),
        "panel": len(panel),
        "by_stratum": per_stratum,
    }


def draw(
    rows: Sequence[Mapping[str, Any]], set_id: str, id_key: str = "image_id",
    min_side: int = EVAL_CROP,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Draw the panel for a view using its approved entry in :data:`PANEL_CAPS`.

    Rows too small for the evaluation crop are removed **before** the draw, so the cap
    is filled from images that can actually be scored. I-047: the first panel was drawn
    without this check and 375 of its 12,150 rows could not be centre-cropped to 320,
    which surfaced as a crash in the S1 pass rather than as a shortfall in the draw.

    Args:
        rows: The view's manifest rows.
        set_id: A key of :data:`PANEL_CAPS`.
        id_key: Column holding the image id.
        min_side: Minimum shorter side. Defaults to :data:`EVAL_CROP`; ``0`` disables it.

    Returns:
        ``(panel_rows, summary)``. The summary carries ``min_side`` and the counts.

    Raises:
        PanelError: If the set has no approved cap. Choosing one here would be an
            unapproved change to the evaluation plan (CLAUDE.md section 9b).
    """
    if set_id not in PANEL_CAPS:
        raise PanelError(
            f"{set_id!r} has no approved panel cap; PANEL_CAPS has "
            f"{sorted(PANEL_CAPS)}. Adding one is a plan change, not a default."
        )
    eligible, dropped, unmeasured = filter_by_min_side(rows, min_side)
    spec = PANEL_CAPS[set_id]
    if spec["mode"] == "per_stratum":
        panel, summary = sample_per_stratum(
            eligible, spec["by"], spec["cap"], set_id, id_key
        )
    else:
        panel, summary = sample_total(eligible, spec["by"], spec["cap"], set_id, id_key)
    summary["min_side"] = {
        "threshold": min_side,
        "eligible": len(eligible),
        "dropped": len(dropped),
        "unmeasured": unmeasured,
        "rule": "shorter side >= the evaluation crop (I-047); label-blind, no backfill",
    }
    return panel, summary
