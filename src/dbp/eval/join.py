"""Joining streams by image id, with hard checks and no silent drops (I-056).

Every number in Session E combines arrays that came from different files: S1 caches,
S2 score CSVs, twelve S3 probe caches, image lists, the split. The first version of
``p2_fit_heads.py`` joined them by **array length** -- a stream whose validation array
was a different length from S1's was silently left out of both fusions, and two arrays
of equal length in different orders would have been fused row-misaligned. Nothing
raised in either case.

So nothing in Session E combines arrays by position any more. Every join goes through
:func:`join_streams` against an explicit, ordered list of target ids, and it raises --
listing the first ten offending ids -- on:

* a duplicate id in the target or in any stream;
* a target id missing from any stream;
* an id a stream holds that the target does not, unless that stream is declared a
  superset (an S3 probe cache covers the panel **and** the training subset, and the
  panel is selected from it by id, explicitly);
* a target id whose label differs between two label sources (:func:`agreed_labels`).

The output is in target order, so the order a file happens to be written in cannot
change any result.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

#: How many offending ids an error lists.
SHOW = 10


class JoinError(ValueError):
    """Raised when streams cannot be joined exactly by id."""


def _first(ids: Iterable[str]) -> list[str]:
    return list(ids)[:SHOW]


def _duplicates(ids: Sequence[str]) -> list[str]:
    return sorted(k for k, n in Counter(ids).items() if n > 1)


def check_unique(name: str, ids: Sequence[str]) -> None:
    """Raise if ``ids`` holds any id twice."""
    dup = _duplicates([str(i) for i in ids])
    if dup:
        raise JoinError(f"{name}: {len(dup)} duplicate id(s), first {SHOW}: {_first(dup)}")


def join_streams(
    target_ids: Sequence[str],
    streams: Mapping[str, tuple[Sequence[str], Any]],
    *,
    supersets: Iterable[str] = (),
    expected_count: int | None = None,
    what: str = "join",
) -> dict[str, np.ndarray]:
    """Align every stream to ``target_ids`` by id.

    Args:
        target_ids: The rows wanted, in the order wanted.
        streams: Name -> ``(ids, values)``; ``values`` has one entry (or row) per id.
        supersets: Streams allowed to hold ids outside the target. Every other stream
            must hold exactly the target's id set.
        expected_count: If given, the target must have exactly this many rows.
        what: Names the join in error messages.

    Returns:
        Name -> values reordered to ``target_ids``.

    Raises:
        JoinError: On any duplicate, missing or unexpected id, or a wrong count.
    """
    target = [str(i) for i in target_ids]
    check_unique(f"{what}: target", target)
    if expected_count is not None and len(target) != expected_count:
        raise JoinError(f"{what}: {len(target)} target row(s), expected {expected_count}")
    if not streams:
        raise JoinError(f"{what}: no streams to join")
    wanted = set(target)
    allowed_extra = set(supersets)
    out: dict[str, np.ndarray] = {}
    for name, (ids, values) in streams.items():
        ids = [str(i) for i in ids]
        array = np.asarray(values)
        if array.shape[0] != len(ids):
            raise JoinError(f"{what}: {name} has {len(ids)} id(s) but {array.shape[0]} value(s)")
        check_unique(f"{what}: {name}", ids)
        index = {image_id: position for position, image_id in enumerate(ids)}
        missing = [i for i in target if i not in index]
        if missing:
            raise JoinError(
                f"{what}: {name} is missing {len(missing)} of {len(target)} id(s), "
                f"first {SHOW}: {_first(missing)}"
            )
        if name not in allowed_extra:
            extra = sorted(set(ids) - wanted)
            if extra:
                raise JoinError(
                    f"{what}: {name} holds {len(extra)} id(s) the target does not, "
                    f"first {SHOW}: {_first(extra)}"
                )
        out[name] = array[[index[i] for i in target]]
    return out


def agreed_labels(
    target_ids: Sequence[str],
    sources: Mapping[str, Mapping[str, str]],
    *,
    what: str = "labels",
) -> np.ndarray:
    """0/1 labels for ``target_ids``, required to agree across every source.

    Args:
        target_ids: The rows.
        sources: Source name -> ``{image_id: "real" | "fake"}``. Every source must
            label every target id.

    Returns:
        ``1`` for fake, ``0`` for real, in target order.

    Raises:
        JoinError: If a source lacks an id, gives a value other than real/fake, or two
            sources disagree.
    """
    if not sources:
        raise JoinError(f"{what}: no label source")
    target = [str(i) for i in target_ids]
    for name, mapping in sources.items():
        missing = [i for i in target if i not in mapping]
        if missing:
            raise JoinError(f"{what}: {name} has no label for {len(missing)} id(s), "
                            f"first {SHOW}: {_first(missing)}")
        odd = [i for i in target if mapping[i] not in ("real", "fake")]
        if odd:
            raise JoinError(f"{what}: {name} has labels other than real/fake for "
                            f"{len(odd)} id(s), first {SHOW}: {_first(odd)}")
    names = list(sources)
    reference = sources[names[0]]
    for name in names[1:]:
        disagree = [i for i in target if sources[name][i] != reference[i]]
        if disagree:
            raise JoinError(
                f"{what}: {name} and {names[0]} disagree on {len(disagree)} id(s), "
                f"first {SHOW}: {_first(disagree)}"
            )
    return np.array([1 if reference[i] == "fake" else 0 for i in target], dtype=int)


def require_streams(used: Iterable[str], configured: Iterable[str], what: str) -> None:
    """Raise unless exactly the configured streams were used."""
    used_set, configured_set = set(used), set(configured)
    if used_set != configured_set:
        raise JoinError(
            f"{what}: streams used {sorted(used_set)} differ from the configured "
            f"{sorted(configured_set)} (missing {sorted(configured_set - used_set)}, "
            f"unexpected {sorted(used_set - configured_set)})"
        )
