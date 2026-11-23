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


# ------------------------------------------------------------------------- the draw ---

def fake_listing(n_sources: int = 30, clips_per_source: int = 2, closed: int = 40):
    real = [f"src{s:03d}_{k}_{10 * k}to{10 * k + 9}.mp4"
            for s in range(n_sources) for k in range(clips_per_source)]
    names = {"AIGVDBench/Real/Real.zip": [f"videos/{n}" for n in real],
             "AIGVDBench/OpenSource/T2V/G1.zip": [f"videos/{n}" for n in real],
             "AIGVDBench/OpenSource/I2V/G2.zip": [f"videos/{n}" for n in real],
             "AIGVDBench/ClosedSource/C1.zip": [f"{i:04d}.mp4" for i in range(closed)],
             "AIGVDBench/ClosedSource/C2.zip": [f"{i:04d}.mp4" for i in range(closed)]}
    return names, real


PARAMS = {"salt": "2026:p5-panel:", "calibration": 6, "closed_reals": 6, "open_pool": 12,
          "per_slice": 4, "stop_after_ineligible": 3}


def test_draw_pools_are_disjoint_by_source_video_and_deterministic():
    names, real = fake_listing()
    test_ids = real[:50]
    draw = F.draw_panel(names, test_ids, PARAMS)
    assert draw == F.draw_panel(dict(reversed(list(names.items()))), list(reversed(test_ids)),
                                PARAMS)
    pools = draw["pools"]
    sources = {p: {F.source_of(n) for n in v} for p, v in pools.items()}
    assert not sources["calibration"] & sources["closed_reals"]
    assert not sources["open"] & (sources["calibration"] | sources["closed_reals"])
    assert all(len(pools[p]) >= PARAMS[k] for p, k in
               (("calibration", "calibration"), ("closed_reals", "closed_reals"),
                ("open", "open_pool")))
    assert set().union(*map(set, pools.values())) <= set(test_ids)
    # whole groups: a source's clips are all in one pool
    for pool, members in pools.items():
        for name in members:
            sibling = [n for n in test_ids if F.source_of(n) == F.source_of(name)]
            assert set(sibling) <= set(members), (pool, name)


def test_open_slices_draw_only_the_open_pool_and_closed_slices_share_one_order():
    names, real = fake_listing()
    draw = F.draw_panel(names, real, PARAMS)
    open_pool = set(draw["pools"]["open"])
    g1 = draw["slices"]["AIGVDBench/OpenSource/T2V/G1.zip"]["candidates"]
    g2 = draw["slices"]["AIGVDBench/OpenSource/I2V/G2.zip"]["candidates"]
    assert set(g1) == set(g2) == open_pool and g1 != g2          # each zip its own order
    c1 = draw["slices"]["AIGVDBench/ClosedSource/C1.zip"]["candidates"]
    c2 = draw["slices"]["AIGVDBench/ClosedSource/C2.zip"]["candidates"]
    assert c1 == c2 and not set(c1) & set(real)


def test_draw_refuses_a_universe_too_small_for_the_pools():
    names, real = fake_listing(n_sources=5)
    with pytest.raises(SystemExit, match="cannot fill the pools"):
        F.draw_panel(names, real, PARAMS)


def test_test_ids_are_read_by_a_known_key_or_refused(tmp_path):
    good = tmp_path / "test.jsonl"
    good.write_text('{"Video_id": "a_0_0to9.mp4", "Prompt": "x"}\n{"Video_id": "b_0_0to9"}\n',
                    encoding="utf-8")
    assert F.read_test_ids(good) == (["a_0_0to9.mp4", "b_0_0to9.mp4"], "Video_id")
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"clip": "a.mp4"}\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="no id key"):
        F.read_test_ids(bad)


# ------------------------------------------------------------- fetch, gate, dedup ---

@pytest.mark.parametrize("method", [zipfile.ZIP_DEFLATED, zipfile.ZIP_STORED])
def test_read_member_inflates_and_checks_the_crc(method):
    payload = bytes(range(256)) * 300
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", method) as z:
        z.writestr("videos/a.mp4", payload)
        z.writestr("videos/b.mp4", b"other")
    data = buffer.getvalue()
    listing, _ = listing_of(data)
    member = next(m for m in listing["members"] if m["name"] == "videos/a.mp4")
    assert F.read_member(CountingFile(data), member) == payload
    with pytest.raises(F.ZipListError, match="CRC"):
        F.read_member(CountingFile(data), {**member, "crc32": "00000000"})


def listing_of(data: bytes):
    return listing(data)


def test_separability_auc_is_direction_free_and_scores_categories():
    labels = [1, 1, 1, 0, 0, 0]
    assert F.separability_auc(labels, [1, 2, 3, 4, 5, 6], False) == 1.0
    assert F.separability_auc(labels, [5, 5, 5, 5, 5, 5], False) == 0.5
    assert F.separability_auc(labels, ["h264"] * 3 + ["vp9"] * 3, True) == 1.0
    assert F.separability_auc([1, 1], [1, 2], False) is None          # one class only


def test_near_duplicates_match_brute_force_and_survive_the_last_row():
    """The dry-run of 2026-11-24 crashed on the last row (an empty comparison)."""
    import random

    rng = random.Random(7)
    base = [rng.getrandbits(64) for _ in range(3)]
    rows = [{"phash": [f"{h:016x}" for h in base]},
            {"phash": [f"{h ^ 0b111:016x}" for h in base]},          # 3 bits off each
            {"phash": [f"{rng.getrandbits(64):016x}" for _ in range(3)]},
            {"phash": [f"{h ^ (1 << 40) - 1:016x}" for h in base]}]   # far
    got = F.near_duplicates(rows, 6)
    want = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows))
            if all(bin(int(a, 16) ^ int(b, 16)).count("1") <= 6
                   for a, b in zip(rows[i]["phash"], rows[j]["phash"], strict=True))]
    assert got == want == [(0, 1)]
    assert F.near_duplicates(rows[:1], 6) == []
