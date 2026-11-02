"""JPEG forensics tests.

Two things are being established.

**That the quality estimator works.** It reads the quantisation tables and scores them
against the Annex K tables scaled by the IJG rule. The constants in the module are
written down from the standard, so the test does not take them on trust: it encodes at
a known quality with Pillow and requires the estimate to come back.

**That the JPEG-matched view is reproducible and safe.** The parameters drawn for an
image must depend only on its id and the set's salt -- not on the order of a directory
listing, not on a process-level random seed -- and real images must never be
re-encoded, because replacing a genuine compression history with a synthetic one is the
same error as leaving the confound in place.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from dbp.data.jpeg import (
    ANNEX_K_CHROMINANCE,
    ANNEX_K_LUMINANCE,
    NOT_JPEG,
    SUBSAMPLING_LABELS,
    compression_is_confounded_with_label,
    encode_jpeg_bytes,
    estimate_quality_from_tables,
    jpeg_metadata,
    sample_matched_pair,
    scaled_table,
    seeded_rng,
    subsampling_code,
)


def _photo_like(height: int = 160, width: int = 192) -> np.ndarray:
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    base = 110.0 + 55.0 * np.sin(xs / 11.0) + 45.0 * np.cos(ys / 15.0)
    stack = np.stack([base, base * 0.9 + 15.0, base * 0.75 + 35.0], axis=-1)
    return np.clip(stack, 0, 255).astype(np.uint8)


def _write(tmp_path, img, **kwargs):
    path = tmp_path / f"q{kwargs.get('quality', 0)}_s{kwargs.get('subsampling', 0)}.jpg"
    path.write_bytes(encode_jpeg_bytes(img, **kwargs))
    return path


# ---------------------------------------------------------------------------
# Quality estimation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quality", [30, 50, 60, 75, 85, 90, 95])
@pytest.mark.parametrize("subsampling", sorted(SUBSAMPLING_LABELS))
def test_estimated_quality_recovers_the_encoder_setting(tmp_path, quality, subsampling):
    path = _write(tmp_path, _photo_like(), quality=quality, subsampling=subsampling)
    metadata = jpeg_metadata(path)
    assert metadata["is_jpeg"] is True
    assert metadata["jpeg_quality"] == quality
    assert metadata["subsampling"] == SUBSAMPLING_LABELS[subsampling]


def test_scaled_table_endpoints_follow_the_ijg_rule():
    """At quality 50 the scale is 1, so the table is the baseline unchanged."""
    assert scaled_table(ANNEX_K_LUMINANCE, 50) == tuple(ANNEX_K_LUMINANCE)
    assert scaled_table(ANNEX_K_CHROMINANCE, 50) == tuple(ANNEX_K_CHROMINANCE)
    # At 100 every entry saturates at the minimum of 1; at 1 most saturate at 255.
    assert set(scaled_table(ANNEX_K_LUMINANCE, 100)) == {1}
    assert max(scaled_table(ANNEX_K_LUMINANCE, 1)) == 255


def test_scaled_table_rejects_a_quality_outside_the_range():
    for bad in (0, 101, -5):
        with pytest.raises(ValueError, match="quality must be in 1..100"):
            scaled_table(ANNEX_K_LUMINANCE, bad)


def test_estimator_returns_none_without_a_luminance_table():
    assert estimate_quality_from_tables({}) is None
    assert estimate_quality_from_tables({1: list(ANNEX_K_CHROMINANCE)}) is None


def test_a_png_is_described_completely_rather_than_left_blank(tmp_path):
    path = tmp_path / "x.png"
    Image.fromarray(_photo_like(), mode="RGB").save(path)
    metadata = jpeg_metadata(path)
    assert metadata["is_jpeg"] is False
    assert metadata["format"] == "PNG"
    assert metadata["jpeg_quality"] is None
    assert metadata["subsampling"] == NOT_JPEG
    assert (metadata["width"], metadata["height"]) == (192, 160)


def test_metadata_reports_the_true_pixel_size(tmp_path):
    img = _photo_like(height=101, width=203)
    path = _write(tmp_path, img, quality=90, subsampling=0)
    metadata = jpeg_metadata(path)
    assert (metadata["width"], metadata["height"]) == (203, 101)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_encoding_is_deterministic_within_one_build():
    """I-010 is about libjpeg differing ACROSS machines; on one machine it must not."""
    img = _photo_like()
    assert encode_jpeg_bytes(img, 75, 0) == encode_jpeg_bytes(img, 75, 0)


def test_encoding_at_different_settings_produces_different_bytes():
    img = _photo_like()
    assert encode_jpeg_bytes(img, 75, 0) != encode_jpeg_bytes(img, 90, 0)
    assert encode_jpeg_bytes(img, 75, 0) != encode_jpeg_bytes(img, 75, 2)


def test_encoding_never_changes_the_image_size():
    img = _photo_like(height=137, width=211)
    import io as _io

    with Image.open(_io.BytesIO(encode_jpeg_bytes(img, 80, 2))) as decoded:
        assert decoded.size == (211, 137), "the matched view must not resize (D-001)"


def test_encoder_rejects_bad_input():
    img = _photo_like()
    with pytest.raises(TypeError, match="uint8"):
        encode_jpeg_bytes(img.astype(np.float32), 75)
    with pytest.raises(ValueError, match="HWC RGB"):
        encode_jpeg_bytes(img[:, :, 0], 75)
    with pytest.raises(ValueError, match="quality must be in"):
        encode_jpeg_bytes(img, 0)
    with pytest.raises(ValueError, match="subsampling must be"):
        encode_jpeg_bytes(img, 75, subsampling=7)


def test_subsampling_labels_round_trip():
    for code, label in SUBSAMPLING_LABELS.items():
        assert subsampling_code(label) == code
    with pytest.raises(ValueError, match="unknown chroma subsampling"):
        subsampling_code("4:1:1")


# ---------------------------------------------------------------------------
# The JPEG-matched view
# ---------------------------------------------------------------------------


def test_matched_parameters_depend_only_on_the_image_id_and_salt():
    reference = [(75, "4:4:4"), (80, "4:2:0"), (85, "4:4:4"), (90, "4:2:0"), (95, "4:4:4")]
    first = sample_matched_pair("img_0001", reference, salt="e3_raw")
    again = sample_matched_pair("img_0001", reference, salt="e3_raw")
    assert first == again, "a rebuilt matched view must be byte-identical"


def test_matched_parameters_differ_between_images_and_between_sets():
    reference = [(quality, "4:4:4") for quality in range(60, 100)]
    draws = {
        sample_matched_pair(f"img_{i:04d}", reference, salt="e3_raw")[1] for i in range(40)
    }
    assert len(draws) > 1, "every image drew the same parameters"

    # Two sets may happen to draw the same pair for one id, so the property that is
    # actually required is that the salt changes the stream.
    assert seeded_rng("img_0001", "e3_raw").integers(0, 2**32) != seeded_rng(
        "img_0001", "e4_wildrf"
    ).integers(0, 2**32)


def test_matched_parameters_reproduce_the_real_distribution():
    """Drawing from the observed pairs, repeats included, keeps the shape of it."""
    reference = [(75, "4:2:0")] * 80 + [(95, "4:4:4")] * 20
    qualities = [
        sample_matched_pair(f"img_{i:05d}", reference, salt="e3_raw")[1] for i in range(2000)
    ]
    share_75 = qualities.count(75) / len(qualities)
    assert 0.75 < share_75 < 0.85, f"expected ~0.80 of draws at q=75, got {share_75:.3f}"


def test_the_pair_is_drawn_jointly_not_independently():
    """Independent draws would produce q=95 with 4:2:0, which occurs in no real here."""
    reference = [(75, "4:2:0")] * 80 + [(95, "4:4:4")] * 20
    drawn = {
        sample_matched_pair(f"img_{i:05d}", reference, salt="e3_raw")[1:] for i in range(500)
    }
    assert drawn <= set(reference), f"invented combination(s): {sorted(drawn - set(reference))}"


def test_the_returned_index_names_the_reference_image():
    reference = [(70, "4:2:0"), (80, "4:2:2"), (90, "4:4:4")]
    index, quality, subsampling = sample_matched_pair("img_0001", reference, salt="e3_raw")
    assert reference[index] == (quality, subsampling)


def test_matched_parameters_refuse_an_empty_reference():
    with pytest.raises(ValueError, match="no reference"):
        sample_matched_pair("img_0001", [], salt="e3_raw")


def test_seeded_rng_is_stable_across_processes():
    """BLAKE2b, not hash(): PYTHONHASHSEED would otherwise change the view per run."""
    expected = int(seeded_rng("stable_check", "salt").integers(0, 2**63))
    assert int(seeded_rng("stable_check", "salt").integers(0, 2**63)) == expected


# ---------------------------------------------------------------------------
# Detecting the confound in the first place
# ---------------------------------------------------------------------------


def test_e3_style_split_is_reported_as_confounded():
    """Measured E3 `Raw/`: 726 real JPEGs, 0 real PNGs; 1361 fake PNGs, 91 fake JPEGs."""
    confounded, reason = compression_is_confounded_with_label(
        {"real": {"jpeg": 726, "non_jpeg": 0}, "fake": {"jpeg": 91, "non_jpeg": 1361}}
    )
    assert confounded is True
    assert "CONFOUNDED" in reason
    assert "93.7%" in reason or "93.8%" in reason


def test_a_balanced_set_is_not_flagged():
    confounded, reason = compression_is_confounded_with_label(
        {"real": {"jpeg": 480, "non_jpeg": 20}, "fake": {"jpeg": 470, "non_jpeg": 30}}
    )
    assert confounded is False
    assert "not confounded" in reason


def test_a_single_label_cannot_be_confounded():
    confounded, reason = compression_is_confounded_with_label({"real": {"jpeg": 10, "non_jpeg": 0}})
    assert confounded is False
    assert "fewer than two labels" in reason


# --------------------------------------------------------------------------- #
# D-042: how an image was compressed, not only whether. The compressed-vs-not
# test reported "not confounded" on E7 while every FLUX.1 reconstruction was
# 4:2:0 and every real 4:4:4 -- a perfect classifier on that subset.
# --------------------------------------------------------------------------- #

from dbp.data.jpeg import (  # noqa: E402
    SEPARABILITY_LIFT_THRESHOLD,
    parameter_separability,
    separability_by_stratum,
)


def _rows(spec: list[tuple[str, str, str, int]]) -> list[dict]:
    """(label, stratum, subsampling, count) -> manifest-shaped rows."""
    out = []
    for label, stratum, subsampling, count in spec:
        out += [{"label": label, "source_vae": stratum, "subsampling": subsampling}] * count
    return out


def test_a_parameter_that_gives_the_label_away_is_caught():
    rows = _rows([("fake", "flux", "4:2:0", 50), ("real", "real", "4:4:4", 50)])
    result = parameter_separability(rows, "subsampling")

    assert result["accuracy"] == 1.0
    assert result["baseline"] == 0.5
    assert result["separable"] is True
    assert result["values"]["4:2:0"] == {"fake": 50}


def test_a_shared_parameter_is_not_flagged():
    rows = _rows([("fake", "a", "4:4:4", 50), ("real", "real", "4:4:4", 50)])
    result = parameter_separability(rows, "subsampling")

    assert result["accuracy"] == result["baseline"] == 0.5
    assert result["lift"] == 0.0
    assert result["separable"] is False


def test_accuracy_alone_would_mislead_on_an_unbalanced_set():
    """95% accuracy from always saying 'fake' is not a finding, which is why lift is used."""
    rows = _rows([("fake", "a", "4:4:4", 95), ("real", "real", "4:4:4", 5)])
    result = parameter_separability(rows, "subsampling")

    assert result["accuracy"] == 0.95
    assert result["baseline"] == 0.95
    assert result["separable"] is False


def test_a_single_label_is_undefined_rather_than_perfect():
    """E8 has no real class. 'Perfectly separable' would be a nonsense verdict."""
    result = parameter_separability(_rows([("fake", "a", "4:4:4", 10)]), "subsampling")

    assert result["separable"] is False
    assert result["accuracy"] is None
    assert "fewer than two labels" in result["reason"]


def test_the_pooled_figure_hides_what_the_strata_reveal():
    """The E7 shape: one subset separable, five not, and a pooled lift near zero."""
    rows = _rows(
        [("fake", "FLUX.1", "4:2:0", 4971)]
        + [("fake", name, "4:4:4", 5000) for name in ("ema", "mse", "sdxl", "sd21", "sd35")]
        + [("real", "real", "4:4:4", 5000)]
    )

    pooled = parameter_separability(rows, "subsampling")
    assert pooled["separable"] is False, "pooled, this is invisible"
    assert abs(pooled["lift"]) < 0.01

    per = separability_by_stratum(rows, "subsampling", "source_vae")
    assert per["separable_strata"] == ["FLUX.1"]
    assert per["by_stratum"]["FLUX.1"]["accuracy"] == 1.0
    assert per["by_stratum"]["ema"]["separable"] is False


def test_each_stratum_is_scored_against_the_shared_real_class():
    rows = _rows([("fake", "a", "4:2:0", 10), ("fake", "b", "4:4:4", 10),
                  ("real", "real", "4:4:4", 10)])
    per = separability_by_stratum(rows, "subsampling", "source_vae")

    assert per["by_stratum"]["a"]["n"] == 20, "10 fakes + the 10 shared reals"
    assert per["by_stratum"]["b"]["n"] == 20


def test_the_threshold_is_a_named_constant_not_a_literal():
    assert 0 < SEPARABILITY_LIFT_THRESHOLD < 0.5
