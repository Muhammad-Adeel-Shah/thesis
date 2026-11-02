"""Tests for the JPEG-matched generation variant (P1-B6b, issue I-023).

This is the training-side half of the compression-confound fix, so the properties are
the same ones `test_eval_sets.py` establishes for the evaluation side, plus the two the
review asked for explicitly: the sampling is deterministic under its seed, and the
quality round-trips exactly.

The one that matters most is the last: after matching, the never-compressed generations
and the JPEG reals must no longer be separable by compression history alone.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from dbp.data.jpeg import encode_jpeg_bytes, jpeg_metadata, sample_matched_pair
from dbp.data.manifest import file_sha256, read_manifest, write_manifest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_matcher():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "p1_match_generations", SCRIPTS / "p1_match_generations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matcher = _load_matcher()

REAL_COLUMNS = (
    "image_id",
    "set_id",
    "label",
    "source_relative_path",
    "is_jpeg",
    "jpeg_quality",
    "subsampling",
)
GEN_COLUMNS = ("image_id", "set_id", "label", "relative_path")


def _photo(height: int, width: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    base = (
        115.0
        + 47.0 * np.sin(xs / (10.0 + seed % 6))
        + 36.0 * np.cos(ys / (13.0 + seed % 8))
        + rng.normal(0, 3.5, size=(height, width))
    )
    stack = np.stack([base, base * 0.88 + 17.0, base * 0.72 + 35.0], axis=-1)
    return np.clip(stack, 0, 255).astype(np.uint8)


@pytest.fixture
def corpus(tmp_path: Path) -> dict:
    """A T-real manifest of JPEG reals and a generation set of never-compressed PNGs."""
    reports = tmp_path / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    # Reals: a deliberately lumpy joint distribution, so independent sampling of
    # quality and subsampling would produce combinations that never occur here.
    pairs = [(72, "4:2:0")] * 10 + [(95, "4:4:4")] * 6
    real_rows = []
    for index, (quality, subsampling) in enumerate(pairs):
        real_rows.append(
            {
                "image_id": f"real_{index:04d}",
                "set_id": "t_real",
                "label": "real",
                "source_relative_path": f"train2017/{index:012d}.jpg",
                "is_jpeg": True,
                "jpeg_quality": quality,
                "subsampling": subsampling,
            }
        )
    reals_manifest = reports / "p1_reals_manifest_t_real.csv"
    write_manifest(real_rows, reals_manifest, REAL_COLUMNS)

    gen_root = tmp_path / "outputs" / "p1" / "generated"
    gen_dir = gen_root / "g_sd15"
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen_rows = []
    for index in range(12):
        gen_id = f"coco_{index:012d}"
        path = gen_dir / f"{gen_id}.png"
        Image.fromarray(_photo(96 + index, 112 + index, seed=200 + index), mode="RGB").save(path)
        gen_rows.append(
            {
                "image_id": gen_id,
                "set_id": "g_sd15",
                "label": "fake",
                "relative_path": path.relative_to(gen_root.parent).as_posix(),
            }
        )
    gen_manifest = reports / "p1_gen_manifest.csv"
    write_manifest(gen_rows, gen_manifest, GEN_COLUMNS)

    return {
        "tmp": tmp_path,
        "reports": reports,
        "gen_manifest": gen_manifest,
        "reals_manifest": reals_manifest,
        "gen_root": gen_root,
        "real_rows": real_rows,
    }


def _run(corpus, monkeypatch, extra: list[str] | None = None) -> int:
    monkeypatch.setattr(matcher, "THESIS_DBP", corpus["tmp"])
    argv = [
        "p1_match_generations.py",
        "--gen-manifest",
        str(corpus["gen_manifest"]),
        "--reals-manifest",
        str(corpus["reals_manifest"]),
        "--gen-root",
        str(corpus["gen_root"]),
        "--out-root",
        str(corpus["tmp"] / "outputs" / "p1" / "gen_matched"),
        *(extra or []),
    ]
    old = sys.argv
    sys.argv = argv
    try:
        return matcher.main()
    finally:
        sys.argv = old


# ---------------------------------------------------------------------------
# The two properties the review asked for by name
# ---------------------------------------------------------------------------


def test_sampling_is_deterministic_under_the_seed():
    reference = [(72, "4:2:0")] * 10 + [(95, "4:4:4")] * 6
    first = sample_matched_pair("coco_000000000007", reference, salt="gen_matched")
    again = sample_matched_pair("coco_000000000007", reference, salt="gen_matched")
    assert first == again
    # And a different id draws independently of it.
    others = {
        sample_matched_pair(f"coco_{i:012d}", reference, salt="gen_matched")[1] for i in range(40)
    }
    assert len(others) > 1, "every generation drew the same quality"


def test_quality_round_trips_exactly(tmp_path):
    """What the matcher writes must read back as the quality it asked for."""
    image = _photo(128, 160, seed=7)
    for quality, subsampling in [(72, 2), (80, 1), (95, 0)]:
        path = tmp_path / f"q{quality}_s{subsampling}.jpg"
        path.write_bytes(encode_jpeg_bytes(image, quality, subsampling))
        assert jpeg_metadata(path)["jpeg_quality"] == quality


def test_joint_sampling_never_invents_an_unobserved_combination():
    """Independent draws would pair q=95 with 4:2:0, which occurs in no real here."""
    reference = [(72, "4:2:0")] * 10 + [(95, "4:4:4")] * 6
    observed = set(reference)
    drawn = {
        sample_matched_pair(f"coco_{i:012d}", reference, salt="gen_matched")[1:]
        for i in range(400)
    }
    assert drawn <= observed, f"invented combination(s): {sorted(drawn - observed)}"
    assert len(drawn) == 2, "the fixture's two combinations should both appear"


def test_sampling_reproduces_the_reference_distribution():
    reference = [(72, "4:2:0")] * 80 + [(95, "4:4:4")] * 20
    draws = [
        sample_matched_pair(f"g_{i:06d}", reference, salt="gen_matched")[1] for i in range(2000)
    ]
    share = draws.count(72) / len(draws)
    assert 0.75 < share < 0.85, f"expected ~0.80 at q=72, got {share:.3f}"


def test_empty_reference_is_refused():
    with pytest.raises(ValueError, match="no reference"):
        sample_matched_pair("coco_0", [], salt="gen_matched")


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_every_generation_gets_a_matched_jpeg(corpus, monkeypatch):
    assert _run(corpus, monkeypatch) == 0

    rows = read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv")
    assert len(rows) == 12
    assert set(rows[0]) == {
        "gen_id",
        "source_real_id",
        "pairing",
        "matched_quality",
        "subsampling",
        "seed",
        "png_sha256",
        "jpg_sha256",
        "png_path",
        "jpg_path",
        "width",
        "height",
    }
    for row in rows:
        assert Path(row["jpg_path"]).is_file()
        assert jpeg_metadata(row["jpg_path"])["is_jpeg"] is True
        assert row["source_real_id"].startswith("real_")


def test_the_png_variant_is_kept_and_untouched(corpus, monkeypatch):
    """P2 reports raw-vs-matched as an ablation, so the raw variant has to survive."""
    before = {
        path: file_sha256(path)
        for path in sorted(corpus["gen_root"].rglob("*.png"))
    }
    _run(corpus, monkeypatch)
    after = {
        path: file_sha256(path)
        for path in sorted(corpus["gen_root"].rglob("*.png"))
    }
    assert after == before and before, "the PNG variant was modified or removed"

    rows = read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv")
    for row in rows:
        assert file_sha256(row["png_path"]) == row["png_sha256"]


def test_matching_never_resizes(corpus, monkeypatch):
    _run(corpus, monkeypatch)
    for row in read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv"):
        with Image.open(row["png_path"]) as png, Image.open(row["jpg_path"]) as jpg:
            assert png.size == jpg.size, "the matched variant was resized (D-001)"
            assert (int(row["width"]), int(row["height"])) == png.size


def test_matched_parameters_come_only_from_observed_reals(corpus, monkeypatch):
    _run(corpus, monkeypatch)
    observed = {
        (int(r["jpeg_quality"]), r["subsampling"]) for r in corpus["real_rows"]
    }
    rows = read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv")
    drawn = {(int(r["matched_quality"]), r["subsampling"]) for r in rows}
    assert drawn <= observed


def test_the_compression_confound_is_gone_after_matching(corpus, monkeypatch):
    """The point of the whole exercise: reals and fakes now share a JPEG history."""
    from dbp.data.jpeg import compression_is_confounded_with_label

    rows = read_manifest(corpus["gen_manifest"])
    before, reason_before = compression_is_confounded_with_label(
        {"real": {"jpeg": 16, "non_jpeg": 0}, "fake": {"jpeg": 0, "non_jpeg": len(rows)}}
    )
    assert before is True and "CONFOUNDED" in reason_before

    _run(corpus, monkeypatch)
    matched = read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv")
    after, reason_after = compression_is_confounded_with_label(
        {"real": {"jpeg": 16, "non_jpeg": 0}, "fake": {"jpeg": len(matched), "non_jpeg": 0}}
    )
    assert after is False, reason_after


def test_rebuilding_gives_byte_identical_jpegs(corpus, monkeypatch):
    _run(corpus, monkeypatch)
    first = {
        row["gen_id"]: row["jpg_sha256"]
        for row in read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv")
    }
    _run(corpus, monkeypatch)
    second = {
        row["gen_id"]: row["jpg_sha256"]
        for row in read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv")
    }
    assert first == second, "the matched generation set is not reproducible"


def test_exact_coco_partners_are_preferred_over_a_random_draw(corpus, monkeypatch):
    """Generations and reals here share COCO ids 0..11, so all 12 pair exactly."""
    _run(corpus, monkeypatch)
    summary = json.loads(
        (corpus["reports"] / "p1_gen_matched_manifest.json").read_text(encoding="utf-8")
    )
    assert summary["pairing_exact"] == 12
    assert summary["pairing_sampled"] == 0
    assert summary["png_variant_kept"] is True
    assert summary["resized"] is False


def test_a_reals_manifest_without_jpegs_is_refused(corpus, monkeypatch, tmp_path):
    """T-ctrl is lossless PNG: matching against it would invent a distribution."""
    lossless = corpus["reports"] / "p1_reals_manifest_t_ctrl.csv"
    write_manifest(
        [
            {
                "image_id": f"ctrl_{i}",
                "set_id": "t_ctrl",
                "label": "real",
                "source_relative_path": f"DIV2K/{i}.png",
                "is_jpeg": False,
                "jpeg_quality": None,
                "subsampling": "not-jpeg",
            }
            for i in range(4)
        ],
        lossless,
        REAL_COLUMNS,
    )
    corpus["reals_manifest"] = lossless
    assert _run(corpus, monkeypatch) == 2


# --------------------------------------------------------------------------- #
# Greyscale exclusions (D-039). A greyscale real has no chroma channels, so it
# cannot define a (quality, subsampling) pair for a colour image. Thirteen
# generations failed on exactly this before the reference was filtered.
# --------------------------------------------------------------------------- #


class _Log:
    """Collects log lines so a test can assert the exclusion was reported, not hidden."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, message, *args):
        self.lines.append(message % args if args else message)

    warning = info
    error = info


def _real(index: int, quality: int, subsampling: str, excluded: str = "") -> dict:
    return {
        "image_id": f"real_{index:04d}",
        "set_id": "t_real",
        "label": "real",
        "source_relative_path": f"train2017/{index:012d}.jpg",
        "is_jpeg": True,
        "jpeg_quality": quality,
        "subsampling": subsampling,
        "excluded_reason": excluded,
    }


def test_reference_drops_unmappable_subsampling():
    """`unknown(-1)` is what Pillow reports for a greyscale JPEG; it cannot be encoded."""
    rows = [_real(0, 96, "4:4:4"), _real(1, 96, "unknown(-1)"), _real(2, 80, "4:2:0")]
    log = _Log()
    reference, usable, exclusions = matcher.build_reference(rows, log)

    assert len(reference) == 2
    assert [row["image_id"] for row in usable] == ["real_0000", "real_0002"]
    assert exclusions["unmappable_subsampling"] == 1
    assert any("unmappable_subsampling" in line for line in log.lines)


def test_reference_drops_rows_flagged_by_the_corpus():
    rows = [_real(0, 96, "4:4:4"), _real(1, 96, "4:4:4", excluded="greyscale")]
    reference, usable, exclusions = matcher.build_reference(rows, _Log())

    assert len(reference) == 1
    assert [row["image_id"] for row in usable] == ["real_0000"]
    assert exclusions["excluded_reason"] == 1


def test_reference_refuses_to_invent_a_distribution():
    """No eligible real means no matched view -- never a guessed setting."""
    with pytest.raises(ValueError, match="no eligible JPEG rows"):
        matcher.build_reference([_real(0, 96, "unknown(-1)")], _Log())


def test_every_reference_pair_can_be_encoded(corpus):
    """The property the failures violated: each drawn label maps to an encoder setting."""
    rows = [_real(i, 96, "4:4:4") for i in range(3)] + [_real(9, 96, "unknown(-1)")]
    reference, _, _ = matcher.build_reference(rows, _Log())
    from dbp.data.jpeg import subsampling_code

    for _quality, subsampling in reference:
        assert subsampling_code(subsampling) in (0, 1, 2)


def test_an_excluded_exact_partner_becomes_a_sampled_colour_partner(corpus, monkeypatch):
    """The generation is kept, and the swap is recorded rather than left implicit.

    Dropping it would shrink the training set for a reason that has nothing to do with
    the generator, so it draws a colour partner by seed instead (D-039).
    """
    rows = list(corpus["real_rows"])
    # real_0003 is the exact partner of coco_000000000003.
    for row in rows:
        row["excluded_reason"] = "greyscale" if row["image_id"] == "real_0003" else ""
    write_manifest(rows, corpus["reals_manifest"], (*REAL_COLUMNS, "excluded_reason"))

    assert _run(corpus, monkeypatch) == 0

    matched = read_manifest(corpus["reports"] / "p1_gen_matched_manifest.csv")
    by_id = {row["gen_id"]: row for row in matched}
    swapped = by_id["coco_000000000003"]

    assert len(matched) == 12, "no generation is lost to an excluded partner"
    assert swapped["pairing"] == "sampled_after_exclusion"
    assert swapped["source_real_id"] != "real_0003"

    summary = json.loads((corpus["reports"] / "p1_gen_matched_manifest.json").read_text())
    assert summary["pairing_counts"]["sampled_after_exclusion"] == 1
    assert summary["reference_excluded"]["excluded_reason"] == 1
    assert summary["failures"] == []
