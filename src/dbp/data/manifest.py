"""CSV manifests: the record of what a built dataset actually contains.

Every dataset this project builds is accompanied by a manifest. The manifest, not the
directory of images, is the artefact the thesis cites: it is small, diffable, and
carries a digest for each file, so a set can be checked for drift without re-reading
gigabytes of pixels.

Three properties are enforced here rather than left to each build script:

- **Atomic writes.** A manifest written by a job the 12-hour Kaggle limit killed must
  either be the previous complete file or the new complete file, never a truncated one.
- **Stable ids.** An image's id is derived from its path inside its source set, so two
  machines that list a directory in different orders still agree on every id -- which
  is what makes the seeded JPEG-matched view reproducible.
- **Resumable appends.** Long builds are sharded. :func:`append_rows` adds a shard's
  rows and writes the header only when creating the file, so a rerun that skips
  finished shards produces the same manifest as an uninterrupted run.
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

#: Columns every image manifest starts with. Build scripts append their own after these.
BASE_COLUMNS: tuple[str, ...] = (
    "image_id",
    "set_id",
    "label",
    "relative_path",
    "width",
    "height",
    "format",
    "is_jpeg",
    "jpeg_quality",
    "subsampling",
    "sha256",
)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Hex SHA-256 of a file, read in chunks so a large image never lands in memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_image_id(relative_path: str | Path, set_id: str = "", length: int = 16) -> str:
    """A deterministic id for an image, derived from its path within its source set.

    Args:
        relative_path: Path relative to the set's root. Converted to POSIX form first,
            so a manifest built on Windows and one built on Linux agree.
        set_id: Included in the digest so the same relative path in two sets does not
            collide.
        length: Hex characters to keep. 16 is 64 bits; at P1 scale the chance of a
            collision is negligible and the id stays readable in a CSV.

    Returns:
        A lowercase hex id of ``length`` characters.
    """
    text = f"{set_id}/{Path(relative_path).as_posix()}"
    return hashlib.blake2b(text.encode("utf-8"), digest_size=32).hexdigest()[:length]


def _normalise(row: Mapping[str, Any], columns: Sequence[str]) -> dict[str, str]:
    """Render one row as strings in column order, with None written as an empty cell."""
    unknown = set(row) - set(columns)
    if unknown:
        raise ValueError(f"row has column(s) not in the manifest schema: {sorted(unknown)}")
    return {
        column: "" if row.get(column) is None else str(row.get(column, ""))
        for column in columns
    }


def write_manifest(
    rows: Iterable[Mapping[str, Any]], path: str | Path, columns: Sequence[str]
) -> Path:
    """Write a manifest atomically: temp file in the same directory, then ``os.replace``.

    Args:
        rows: The rows, each a mapping whose keys are a subset of ``columns``.
        path: Destination CSV.
        columns: Column order. Rows may omit columns; they may not invent them.

    Returns:
        The destination path.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    handle_fd, tmp = tempfile.mkstemp(dir=str(dest.parent), suffix=".csv.tmp")
    os.close(handle_fd)
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns))
            writer.writeheader()
            for row in rows:
                writer.writerow(_normalise(row, columns))
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return dest


def append_rows(
    rows: Iterable[Mapping[str, Any]], path: str | Path, columns: Sequence[str]
) -> Path:
    """Append rows to a manifest, creating it with a header if it does not exist.

    Used by the sharded builders. Appending is not atomic in the way
    :func:`write_manifest` is, so a shard is written only after its images are on disk:
    a manifest row can then never describe a file that is not there, though a killed
    run can leave a file that has no row yet. The next run re-does that shard, which is
    safe because the reconstruction is deterministic.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    exists = dest.is_file() and dest.stat().st_size > 0
    with dest.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(_normalise(row, columns))
    return dest


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a manifest back as a list of string-valued dicts."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def completed_ids(path: str | Path, column: str = "image_id") -> set[str]:
    """Ids already present in a manifest, so a resumed build can skip them.

    Returns an empty set if the manifest does not exist yet, which is the state a
    first run is in.
    """
    manifest = Path(path)
    if not manifest.is_file():
        return set()
    return {row[column] for row in read_manifest(manifest) if row.get(column)}


def manifest_digest(path: str | Path) -> str:
    """SHA-256 of the manifest file itself, to cite one exact build in the log."""
    return file_sha256(path)


def summarise_by(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    """Count rows per distinct value of ``key``, for the tables in a data card."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get(key, ""))] = counts.get(str(row.get(key, "")), 0) + 1
    return dict(sorted(counts.items()))


def compression_table(
    rows: Sequence[Mapping[str, Any]], label_column: str = "label"
) -> dict[str, dict[str, Any]]:
    """Build the ``label x compression`` table every data card must carry (CLAUDE.md 6b).

    Args:
        rows: Manifest rows carrying ``format``, ``is_jpeg``, ``jpeg_quality`` and
            ``subsampling``.
        label_column: Which column holds the class label.

    Returns:
        ``{label: {n, jpeg, non_jpeg, formats, quality_min, quality_median,
        quality_max, subsampling}}``. Counts are plain ints and the quality summary is
        None when a label has no JPEGs at all, so the absence is visible rather than
        rendered as a zero.
    """
    table: dict[str, dict[str, Any]] = {}
    for row in rows:
        label = str(row.get(label_column, ""))
        entry = table.setdefault(
            label,
            {
                "n": 0,
                "jpeg": 0,
                "non_jpeg": 0,
                "formats": {},
                "_qualities": [],
                "subsampling": {},
            },
        )
        entry["n"] += 1
        image_format = str(row.get("format", ""))
        entry["formats"][image_format] = entry["formats"].get(image_format, 0) + 1
        is_jpeg = str(row.get("is_jpeg", "")).strip().lower() in ("true", "1", "yes")
        entry["jpeg" if is_jpeg else "non_jpeg"] += 1
        subsampling = str(row.get("subsampling", ""))
        entry["subsampling"][subsampling] = entry["subsampling"].get(subsampling, 0) + 1
        quality = row.get("jpeg_quality")
        if is_jpeg and quality not in (None, ""):
            entry["_qualities"].append(int(quality))

    for entry in table.values():
        qualities = sorted(entry.pop("_qualities"))
        entry["formats"] = dict(sorted(entry["formats"].items()))
        entry["subsampling"] = dict(sorted(entry["subsampling"].items()))
        if qualities:
            middle = len(qualities) // 2
            entry["quality_min"] = qualities[0]
            entry["quality_median"] = (
                qualities[middle]
                if len(qualities) % 2
                else (qualities[middle - 1] + qualities[middle]) // 2
            )
            entry["quality_max"] = qualities[-1]
        else:
            entry["quality_min"] = None
            entry["quality_median"] = None
            entry["quality_max"] = None
    return dict(sorted(table.items()))
