"""Train/validation splits that content cannot cross.

The unit of a split here is never an image. It is the **piece of content** an image
came from, because the corpus contains many images per piece of content and every one
of them carries the same scene:

* ``t_ctrl`` stores two crops of each DIV2K photograph, so the two crops of source
  ``0001`` are the same picture twice;
* each real has twelve reconstructions, one per decoder member, and a reconstruction
  is the real's content run through a decoder;
* each SD1.5 generation was produced from a **COCO caption**, and that caption belongs
  to a specific COCO photograph.

Splitting by image would put a real in train and its own reconstruction in validation,
and the validation AUC would then measure memorisation. So the split is assigned to a
group -- ``source_image_id`` for reals, ``source_real_id`` for pair rows, the captioned
COCO image for generations -- and every row inherits its group's side.

**The caption link, not the JPEG-matching link (D-046).** ``gen_matched`` rows carry a
``source_real_id``, but that is the real whose *JPEG settings* were copied, chosen by
the matcher and often unrelated to the picture. The content link is the caption:
generation ``coco_000000000009`` was written from the caption of COCO image
``000000000009``. This module follows the caption and ignores ``source_real_id``, which
is the whole reason :func:`caption_source_stem` exists.

Stratification is by COCO supercategory, using the **rarest** supercategory present in
an image (:func:`stratum_label`). Ranking by the most frequent one instead would label
almost the whole corpus ``person``, which appears in the majority of COCO images and
would stratify nothing; the rarest-present rule spreads the tail categories evenly
across the two sides, which is what stratification is for.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import PurePosixPath
from typing import Any

#: The approved split seed (D-046). Changing it re-draws the split, which invalidates
#: every calibration and every threshold fitted on the old one.
SPLIT_SEED = 2026

#: The approved train share. Validation carries the remaining 20%.
TRAIN_FRACTION = 0.8

TRAIN = "train"
VAL = "val"

#: Stratum for a real with no COCO instance annotation at all. COCO has images whose
#: only content is unannotated background; they are a stratum of their own rather than
#: being folded into a category they do not belong to.
NO_CATEGORY = "none"

#: Prefix the P1 generation manifest uses for the COCO image its caption came from.
GENERATION_ID_PREFIX = "coco_"


class SplitError(ValueError):
    """Raised when a split is malformed or content crosses it."""


def coco_stem(relative_path: str) -> str:
    """COCO file stem of a source path, e.g. ``000000000009.jpg`` -> ``000000000009``.

    Args:
        relative_path: A ``source_relative_path`` value from the reals manifest.

    Returns:
        The stem, with no directory and no suffix.
    """
    return PurePosixPath(str(relative_path).replace("\\", "/")).stem


def caption_source_stem(generation_image_id: str) -> str:
    """COCO stem of the caption a generation was written from.

    The P1 generation manifest names each row ``coco_<stem>`` precisely because the
    caption's provenance had to survive into the file name.

    Args:
        generation_image_id: An ``image_id`` from ``p1_gen_manifest.csv``, or a
            ``gen_id`` from ``p1_gen_matched_manifest.csv``.

    Returns:
        The COCO stem.

    Raises:
        SplitError: If the id does not carry the expected prefix. Guessing here would
            silently detach a generation from its caption and put it on whichever side
            of the split happened to be convenient.
    """
    text = str(generation_image_id)
    if not text.startswith(GENERATION_ID_PREFIX):
        raise SplitError(
            f"generation id {text!r} does not start with {GENERATION_ID_PREFIX!r}, so "
            "the COCO image its caption came from cannot be determined. The split must "
            "follow the caption, not the JPEG-matching partner (D-046)."
        )
    return text[len(GENERATION_ID_PREFIX) :]


def stratum_label(
    supercategories: Iterable[str], corpus_frequency: Mapping[str, int]
) -> str:
    """Single stratum for an image that may carry several supercategories.

    The rarest supercategory present wins. Ties are broken alphabetically so the label
    does not depend on set iteration order.

    Args:
        supercategories: Supercategories annotated in the image, possibly empty.
        corpus_frequency: How often each supercategory occurs across the corpus.

    Returns:
        The chosen supercategory, or :data:`NO_CATEGORY` when there is none.
    """
    present = sorted(str(name) for name in supercategories)
    if not present:
        return NO_CATEGORY
    return min(present, key=lambda name: (corpus_frequency.get(name, 0), name))


def corpus_supercategory_frequency(
    per_image: Mapping[str, Iterable[str]],
) -> dict[str, int]:
    """Count how many images carry each supercategory.

    Args:
        per_image: Group id -> its supercategories.

    Returns:
        Supercategory -> number of images carrying it.
    """
    counter: Counter[str] = Counter()
    for names in per_image.values():
        counter.update(sorted({str(name) for name in names}))
    return dict(counter)


def assign_group_splits(
    groups: Mapping[str, str],
    train_fraction: float = TRAIN_FRACTION,
    seed: int = SPLIT_SEED,
) -> dict[str, str]:
    """Assign each group to ``train`` or ``val``, stratified and seeded.

    Each stratum is shuffled independently under a seed derived from the global seed
    and the stratum name, so adding or removing a stratum cannot re-draw the others.

    Args:
        groups: Group id -> stratum label.
        train_fraction: Share of each stratum's groups that goes to train.
        seed: Base seed.

    Returns:
        Group id -> ``"train"`` or ``"val"``.

    Raises:
        SplitError: If ``train_fraction`` is not strictly between 0 and 1.
    """
    if not 0.0 < train_fraction < 1.0:
        raise SplitError(f"train_fraction must be in (0, 1), got {train_fraction!r}")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for group_id, stratum in groups.items():
        by_stratum[str(stratum)].append(str(group_id))

    assignment: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        members = sorted(by_stratum[stratum])
        rng = random.Random(f"{seed}:{stratum}")
        rng.shuffle(members)
        n_train = round(len(members) * train_fraction)
        # A stratum of one would otherwise contribute nothing to validation; a stratum
        # of two must still give one to each side.
        if len(members) >= 2:
            n_train = min(max(n_train, 1), len(members) - 1)
        for index, group_id in enumerate(members):
            assignment[group_id] = TRAIN if index < n_train else VAL
    return assignment


def apply_split(
    rows: Sequence[Mapping[str, Any]],
    group_of: Mapping[str, str],
    group_key: str,
    *,
    row_label: str = "row",
) -> list[dict[str, Any]]:
    """Copy rows with a ``split`` column filled in from their group.

    Args:
        rows: Manifest rows.
        group_of: Group id -> split.
        group_key: Column holding the group id.
        row_label: What a row is called in the error message.

    Returns:
        New rows, each with ``split`` set.

    Raises:
        SplitError: If a row's group has no assignment. Defaulting it to train would
            hide a join that silently failed.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        group_id = str(row[group_key])
        if group_id not in group_of:
            raise SplitError(
                f"{row_label} with {group_key}={group_id!r} has no split assignment; "
                "the join to the reals manifest failed for it."
            )
        new = dict(row)
        new["split"] = group_of[group_id]
        out.append(new)
    return out


def verify_no_content_crosses(
    rows_by_set: Mapping[str, Sequence[Mapping[str, Any]]],
    content_key: str = "content_id",
) -> dict[str, Any]:
    """Check that no piece of content appears on both sides, across all sets together.

    This is the assertion the whole module exists for, and it is deliberately run over
    the union of the sets rather than one at a time: a real in train and its own
    reconstruction in validation is exactly the failure that a per-set check misses.

    Args:
        rows_by_set: Set id -> rows carrying ``split`` and ``content_key``.
        content_key: Column holding the content identity.

    Returns:
        A report with the per-set counts and the (empty) list of offenders.

    Raises:
        SplitError: If any content id occurs in both splits.
    """
    sides: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = {}
    for set_id, rows in rows_by_set.items():
        per_set: Counter[str] = Counter()
        for row in rows:
            split = str(row["split"])
            if split not in (TRAIN, VAL):
                raise SplitError(
                    f"{set_id}: split must be {TRAIN!r} or {VAL!r}, got {split!r}"
                )
            sides[str(row[content_key])].add(split)
            per_set[split] += 1
        counts[set_id] = {TRAIN: per_set[TRAIN], VAL: per_set[VAL]}

    crossing = sorted(cid for cid, s in sides.items() if len(s) > 1)
    if crossing:
        raise SplitError(
            f"{len(crossing)} content id(s) appear in both splits, e.g. "
            f"{crossing[:5]}. Training on one crop, one reconstruction or one "
            "caption of a picture and validating on another is leakage."
        )
    return {
        "content_key": content_key,
        "distinct_content": len(sides),
        "crossing": [],
        "counts": counts,
    }


def summarise_split(
    groups: Mapping[str, str], assignment: Mapping[str, str]
) -> dict[str, Any]:
    """Per-stratum train/val counts, for the report and the data card.

    Args:
        groups: Group id -> stratum.
        assignment: Group id -> split.

    Returns:
        A JSON-ready summary.
    """
    per_stratum: dict[str, Counter[str]] = defaultdict(Counter)
    for group_id, stratum in groups.items():
        per_stratum[str(stratum)][assignment[str(group_id)]] += 1
    total = Counter(assignment.values())
    return {
        "seed": SPLIT_SEED,
        "train_fraction": TRAIN_FRACTION,
        "groups": len(groups),
        "totals": {TRAIN: total[TRAIN], VAL: total[VAL]},
        "by_stratum": {
            stratum: {TRAIN: counts[TRAIN], VAL: counts[VAL]}
            for stratum, counts in sorted(per_stratum.items())
        },
    }
