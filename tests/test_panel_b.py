"""Panel B draw rules (P3 pre-registration §2, P3_PLAN §1): one test per rule."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbp.data.phash import hamming_distance  # noqa: E402
from dbp.eval.panel_b import (  # noqa: E402
    HashIndex,
    PanelBError,
    coco_id,
    draw_view,
    e3_number,
    hamming_to_all,
    order_key,
    source_id,
)


def _hex(i: int) -> str:
    """Hashes far apart from each other: a different 16-bit block per index."""
    return f"{(0xFFFF << (16 * (i % 4))) ^ (i * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFFFFFF):016x}"


def _row(image_id: str, label: str, path: str, *, side: int = 512, sha: str | None = None,
         phash: str | None = None, **extra) -> dict:
    return {"image_id": image_id, "label": label, "relative_path": path, "width": side,
            "height": side, "sha256": sha or f"{abs(hash(image_id)):064x}"[:64].rjust(64, "0"),
            "phash": phash or _hex(abs(hash(image_id)) % 10_000), **extra}


def _e7(n: int = 30) -> list[dict]:
    rows = []
    for i in range(n):
        cid = f"{i:012d}"
        rows.append(_row(f"r{i}", "real", f"real/{cid}.jpg", source_vae="real",
                         phash=f"{i * 7919:016x}"))
        for k, vae in enumerate(("FLUX.1", "sdxl-vae")):
            rows.append(_row(f"{vae}{i}", "fake", f"{vae}/val2017/{cid}.jpg", source_vae=vae,
                             phash=f"{(i * 104729 + k * 1299709) << 20:016x}"[-16:]))
    return rows


E7_SLOTS = [("FLUX.1", "fake", "source_vae", "FLUX.1", 5, "test"),
            ("sdxl-vae", "fake", "source_vae", "sdxl-vae", 5, "test"),
            ("real-calibration", "real", None, None, 3, "calibration"),
            ("real-test", "real", None, None, 5, "test")]


def test_vectorised_hamming_matches_the_scalar_one() -> None:
    rng = np.random.default_rng(0)
    values = rng.integers(0, 2**63, 50, dtype=np.uint64)
    probe = int(rng.integers(0, 2**63))
    expected = [hamming_distance(probe, int(v)) for v in values]
    np.testing.assert_array_equal(hamming_to_all(probe, values), expected)


def test_identities() -> None:
    row = {"relative_path": "sdxl-vae/val2017/000000167128.jpg"}
    assert coco_id(row) == "000000167128"
    assert source_id("e7_dda_coco", row) == "coco:000000167128"
    assert source_id("e4_wildrf_matched", {"relative_path": "test\\reddit\\0_real\\1.jpg"}) == \
        "path:test/reddit/0_real/1.jpg"
    assert e3_number({"relative_path": "images/Raw/Hybrid Images/real/181.jpg"}) == \
        ("Hybrid Images", "181")
    with pytest.raises(PanelBError):
        e3_number({"relative_path": "181.jpg"})


def test_e7_never_reuses_a_coco_id_and_never_reuses_a_p2_one() -> None:
    rows = _e7()
    p2 = [r for r in rows if r["image_id"] in ("r0", "FLUX.11")]      # COCO ids 0 and 1
    panel, summary = draw_view("e7_dda_coco", rows, p2, HashIndex(), HashIndex(),
                               slots_spec=E7_SLOTS)
    ids = [coco_id(r) for r in panel]
    assert len(ids) == len(set(ids))                                    # one id, once
    assert not {"000000000000", "000000000001"} & set(ids)              # P2 sources out
    assert summary["rejections"]["source image in the P2 panel"] >= 1
    assert sum(1 for r in panel if r["role"] == "calibration") == 3


def test_the_draw_is_deterministic_and_seeded() -> None:
    rows = _e7()
    a, _ = draw_view("e7_dda_coco", rows, [], HashIndex(), HashIndex(), slots_spec=E7_SLOTS)
    b, _ = draw_view("e7_dda_coco", list(reversed(rows)), [], HashIndex(), HashIndex(),
                     slots_spec=E7_SLOTS)
    assert [r["image_id"] for r in a] == [r["image_id"] for r in b]
    c, _ = draw_view("e7_dda_coco", rows, [], HashIndex(), HashIndex(), slots_spec=E7_SLOTS,
                     seed=7)
    assert [r["image_id"] for r in a] != [r["image_id"] for r in c]
    assert order_key("v", "x") != order_key("w", "x")


def test_the_calibration_pool_is_the_first_reals_in_seeded_order() -> None:
    rows = _e7()
    panel, _ = draw_view("e7_dda_coco", rows, [], HashIndex(), HashIndex(), slots_spec=E7_SLOTS)
    reals = [r for r in panel if r["label"] == "real"]
    keys = [int(r["order_key"], 16) for r in reals]
    cal = [k for r, k in zip(reals, keys, strict=True) if r["role"] == "calibration"]
    test = [k for r, k in zip(reals, keys, strict=True) if r["role"] == "test"]
    assert max(cal) < min(test)


def test_small_images_and_reference_duplicates_are_rejected_and_counted() -> None:
    rows = [_row("big", "fake", "a/1.png", side=400), _row("small", "fake", "a/2.png", side=319),
            _row("bytes", "fake", "a/3.png", sha="ab" * 32),
            _row("near", "fake", "a/4.png", phash="ffffffffffffff00")]
    refs = HashIndex()
    refs.add("t_real", "ab" * 32, None)
    refs.add("p2_panel", None, "ffffffffffffff03")                     # 2 bits from "near"
    slots = [("fake", "fake", None, None, None, "test")]
    panel, summary = draw_view("e4_wildrf_matched", rows, [], refs, HashIndex(), slots_spec=slots)
    assert [r["image_id"] for r in panel] == ["big"]
    rejected = summary["rejections"]
    assert rejected["shorter side < 320 or unmeasured"] == 1
    assert rejected["duplicate of a reference (sha: t_real)"] == 1
    assert rejected["duplicate of a reference (phash: p2_panel)"] == 1


def test_a_near_duplicate_inside_panel_b_is_rejected_across_views() -> None:
    accepted = HashIndex()
    slots = [("fake", "fake", None, None, None, "test")]
    first, _ = draw_view("e4_wildrf_matched", [_row("a", "fake", "x/1.png", phash="0f" * 8)],
                         [], HashIndex(), accepted, slots_spec=slots)
    second, summary = draw_view("e5_chameleon_pipeline",
                                [_row("b", "fake", "y/1.jpg", phash="0f" * 7 + "0e")],
                                [], HashIndex(), accepted, slots_spec=slots)
    assert len(first) == 1 and second == []
    assert summary["rejections"]["duplicate of a Panel B row"] == 1


def test_capacities_are_respected_and_shortfalls_reported() -> None:
    rows = [_row(f"f{i}", "fake", f"g/{i}.jpg", generator="GoT", phash=_hex(i)) for i in range(3)]
    slots = [("GoT", "fake", "generator", "GoT", 2, "test"),
             ("NOVA", "fake", "generator", "NOVA", 4, "test")]
    panel, summary = draw_view("e8_evalgen", rows, [], HashIndex(), HashIndex(), slots_spec=slots)
    assert len(panel) == 2 and summary["rejections"]["slot full"] == 1
    assert summary["slots"]["NOVA"]["shortfall"] == 4


def test_e3_rows_carry_the_p2_number_flag() -> None:
    rows = [_row("r5", "real", "images/Raw/Hybrid Images/real/5.jpg", source_label="real"),
            _row("g7", "fake", "images/Raw/Hybrid Images/GPT-Image2-fake/7.png",
                 source_label="GPT-Image2-fake")]
    p2 = [_row("p", "fake", "images/Raw/Hybrid Images/Nano-Banana2-fake/5.png",
               source_label="Nano-Banana2-fake")]
    slots = [("GPT-Image2-fake", "fake", "source_label", "GPT-Image2-fake", None, "test"),
             ("real-test", "real", None, None, None, "test")]
    panel, summary = draw_view("e3_matched", rows, p2, HashIndex(), HashIndex(), slots_spec=slots)
    flags = {r["image_id"]: r["e3_number_in_p2"] for r in panel}
    assert flags == {"r5": "true", "g7": "false"}                      # number 5, any label
    assert summary["e3_number_in_p2"]["real|real|true"] == 1


def test_duplicate_manifest_ids_are_refused() -> None:
    rows = [_row("x", "fake", "a/1.png"), _row("x", "fake", "a/2.png")]
    with pytest.raises(PanelBError, match="duplicate image ids"):
        draw_view("e4_wildrf_matched", rows, [], HashIndex(), HashIndex(),
                  slots_spec=[("fake", "fake", None, None, None, "test")])
