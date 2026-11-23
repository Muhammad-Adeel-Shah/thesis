"""P5 test panel: the zip-directory listing (no network; local zips in memory)."""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import p5_fetch_panel as F  # noqa: E402


class CountingFile:
    """A local file with the remote reader's interface: one exact range per read."""

    def __init__(self, data: bytes) -> None:
        self.data, self.ranges = data, []

    def _fetch_range(self, start: int, end: int) -> bytes:
        self.ranges.append((start, end))
        return self.data[start:end]


def listing(data: bytes) -> tuple[dict, CountingFile]:
    remote = CountingFile(data)
    reader = F.RangeReader(remote)
    out = F.central_directory(reader, len(data))
    assert reader.downloaded == out["bytes_read"]
    return out, remote


def make_zip(names: list[str], payload: bytes = b"x" * 5000, comment: bytes = b"") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as z:
        for name in names:
            z.writestr(name, payload)
        z.comment = comment
    return buffer.getvalue()


def test_listing_matches_zipfile_and_never_reads_member_data():
    names = ["T2V/Sora/a_0_0to100.mp4", "T2V/Sora/b.mp4", "T2V/Sora/sub/", "c d é.mp4"]
    data = make_zip(names)
    out, remote = listing(data)
    reference = zipfile.ZipFile(io.BytesIO(data)).infolist()
    assert [m["name"] for m in out["members"]] == [i.filename for i in reference]
    assert [m["size"] for m in out["members"]] == [i.file_size for i in reference]
    assert not out["zip64"] and out["tail_in_directory"]
    # every byte read lies in the central directory or the end records
    for start, _ in remote.ranges:
        assert start >= out["cd_offset"]


def test_zip64_directory_with_more_than_65535_entries():
    data = make_zip([f"v{i:06d}.mp4" for i in range(65_600)], payload=b"")
    out, _ = listing(data)
    assert out["zip64"] and len(out["members"]) == 65_600
    assert out["members"][-1]["name"] == "v065599.mp4"


def test_a_zip_comment_grows_the_tail_read():
    data = make_zip(["a.mp4"], comment=b"c" * 500)
    out, remote = listing(data)
    assert [m["name"] for m in out["members"]] == ["a.mp4"]
    assert len(remote.ranges) == 3          # 128 B tail, the full tail, the directory


def test_not_a_zip_is_refused():
    with pytest.raises(F.ZipListError, match="end-of-central-directory"):
        listing(b"\0" * 4096)
