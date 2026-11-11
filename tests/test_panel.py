"""Tests for the shared evaluation panel (B1)."""

from __future__ import annotations

import pytest

from dbp.eval import panel
from dbp.eval.panel import (
    PANEL_CAPS,
    PanelError,
    allocate,
    draw,
    sample_per_stratum,
    sample_total,
    stratum_of,
)


def _rows(spec: dict[str, int], **columns: str) -> list[dict[str, str]]:
    """Rows shaped like a view manifest: ``spec`` maps a stratum value to a count."""
    out = []
    for value, count in spec.items():
        for i in range(count):
            out.append({"image_id": f"{value}_{i:05d}", "stratum": value, **columns})
    return out


def test_stratum_joins_the_named_columns() -> None:
    row = {"generator": "GPT", "content_type": "texture"}
    assert stratum_of(row, ("generator", "content_type")) == "GPT|texture"


def test_a_missing_stratum_column_is_an_error_not_an_unknown_bucket() -> None:
    with pytest.raises(PanelError, match="cannot form its stratum"):
        stratum_of({"generator": "GPT"}, ("generator", "content_type"))


def test_allocate_returns_everything_when_the_population_fits() -> None:
    assert allocate({"a": 10, "b": 5}, 100) == {"a": 10, "b": 5}


def test_allocate_is_proportional_and_sums_to_the_total() -> None:
    quota = allocate({"a": 600, "b": 300, "c": 100}, 100)
    assert sum(quota.values()) == 100
    assert quota == {"a": 60, "b": 30, "c": 10}


def test_allocate_never_over_draws_a_small_stratum_and_redistributes() -> None:
    """A tiny stratum must not be asked for more rows than it has."""
    quota = allocate({"big": 100, "tiny": 2}, 100)
    assert quota["tiny"] == 2
    assert quota["big"] == 98
    assert sum(quota.values()) == 100


def test_allocate_keeps_a_rare_stratum_present() -> None:
    quota = allocate({"big": 10000, "rare": 40}, 1000)
    assert quota["rare"] > 0, "a rare stratum was rounded out of the panel"


def test_allocate_lifts_a_stratum_that_would_round_to_zero() -> None:
    """D-044 reports E3 per content type; a stratum absent from the panel cannot be."""
    quota = allocate({"big": 1000, "rare": 3}, 100)
    assert quota["rare"] >= 1
    assert quota["rare"] <= 3
    assert sum(quota.values()) == 100


def test_allocate_rejects_a_negative_total() -> None:
    with pytest.raises(PanelError, match="non-negative"):
        allocate({"a": 5}, -1)


def test_per_stratum_caps_each_stratum_independently() -> None:
    rows = _rows({"flux": 4971, "sdxl": 5000, "real": 5000})
    panel, summary = sample_per_stratum(rows, ("stratum",), 500, "e7_dda_coco")
    assert len(panel) == 1500
    assert summary["by_stratum"]["flux"] == {"available": 4971, "taken": 500}
    assert summary["population"] == 14971


def test_per_stratum_takes_everything_from_a_stratum_smaller_than_the_cap() -> None:
    rows = _rows({"big": 900, "small": 40})
    panel, summary = sample_per_stratum(rows, ("stratum",), 500, "s")
    assert len(panel) == 540
    assert summary["by_stratum"]["small"]["taken"] == 40


def test_total_mode_respects_the_overall_cap() -> None:
    rows = _rows({"a": 1000, "b": 800, "c": 378})
    panel, summary = sample_total(rows, ("stratum",), 1500, "e3_matched")
    assert len(panel) == 1500
    assert summary["mode"] == "total"
    assert sum(v["taken"] for v in summary["by_stratum"].values()) == 1500


def test_sampling_is_reproducible() -> None:
    rows = _rows({"a": 500, "b": 500})
    first, _ = sample_per_stratum(rows, ("stratum",), 100, "set")
    second, _ = sample_per_stratum(rows, ("stratum",), 100, "set")
    assert [r["image_id"] for r in first] == [r["image_id"] for r in second]


def test_sampling_does_not_depend_on_the_input_order() -> None:
    """A view rebuilt in a different file order must give the same panel."""
    rows = _rows({"a": 400, "b": 400})
    forward, _ = sample_per_stratum(rows, ("stratum",), 50, "set")
    backward, _ = sample_per_stratum(list(reversed(rows)), ("stratum",), 50, "set")
    assert sorted(r["image_id"] for r in forward) == sorted(
        r["image_id"] for r in backward
    )


def test_two_views_draw_different_subsets_from_identical_ids() -> None:
    rows = _rows({"a": 400})
    left, _ = sample_per_stratum(rows, ("stratum",), 50, "e3_matched")
    right, _ = sample_per_stratum(rows, ("stratum",), 50, "e4_wildrf_matched")
    assert {r["image_id"] for r in left} != {r["image_id"] for r in right}


def test_the_panel_contains_no_duplicates() -> None:
    rows = _rows({"a": 300, "b": 300, "c": 300})
    panel, _ = sample_total(rows, ("stratum",), 400, "set")
    ids = [r["image_id"] for r in panel]
    assert len(ids) == len(set(ids))


def test_multi_column_strata_are_kept_apart() -> None:
    rows = [
        {"image_id": f"{g}_{c}_{i}", "generator": g, "content_type": c}
        for g in ("GPT", "Nano")
        for c in ("texture", "structure")
        for i in range(100)
    ]
    _, summary = sample_total(rows, ("generator", "content_type"), 200, "e3_matched")
    assert set(summary["by_stratum"]) == {
        "GPT|texture", "GPT|structure", "Nano|texture", "Nano|structure",
    }


def test_draw_uses_the_approved_cap_for_a_known_set() -> None:
    rows = _rows({"real": 3000, "fake": 3000}, label="x")
    for row in rows:
        row["label"] = row["stratum"]
    panel, summary = draw(rows, "e5_chameleon_pipeline")
    assert summary["cap"] == PANEL_CAPS["e5_chameleon_pipeline"]["cap"] == 2000
    assert len(panel) == 4000


def test_draw_refuses_a_set_with_no_approved_cap() -> None:
    """Inventing a cap here would be an unapproved change to the evaluation plan."""
    with pytest.raises(PanelError, match="no approved panel cap"):
        draw(_rows({"a": 10}), "e9_something_new")


def test_empty_input_is_an_error() -> None:
    with pytest.raises(PanelError, match="no rows"):
        sample_per_stratum([], ("stratum",), 10, "set")


def test_a_non_positive_cap_is_an_error() -> None:
    with pytest.raises(PanelError, match="cap must be positive"):
        sample_per_stratum(_rows({"a": 5}), ("stratum",), 0, "set")
    with pytest.raises(PanelError, match="total must be positive"):
        sample_total(_rows({"a": 5}), ("stratum",), 0, "set")


def test_the_approved_caps_add_up_to_the_budgeted_panel() -> None:
    """What the caps allocate: 12,150. The evaluation-crop rule (I-047) then
    removes 375 of them, leaving the 11,775 the panel actually holds."""
    expected = (
        PANEL_CAPS["e7_dda_coco"]["cap"] * 7        # 6 source VAEs + real
        + PANEL_CAPS["e8_evalgen"]["cap"] * 5       # 5 generators, no reals
        + PANEL_CAPS["e3_matched"]["cap"]
        + PANEL_CAPS["e4_wildrf_matched"]["cap"]
        + PANEL_CAPS["e5_chameleon_pipeline"]["cap"] * 2
        + PANEL_CAPS["t_ctrl_diagnostic"]["cap"] * 2
    )
    assert expected == 12150


def test_every_approved_cap_names_columns_the_manifests_actually_have() -> None:
    """E3's columns are `source_label` and `subset`.

    The first version of PANEL_CAPS said `generator` and `content_type`, which are
    plausible and wrong, and the E3 panel refused to build. These are the names
    measured from the committed P1 eval manifests.
    """
    assert PANEL_CAPS["e7_dda_coco"]["by"] == ("source_vae",)
    assert PANEL_CAPS["e8_evalgen"]["by"] == ("generator",)
    assert PANEL_CAPS["e3_matched"]["by"] == ("source_label", "subset")
    assert PANEL_CAPS["e4_wildrf_matched"]["by"] == ("platform", "label")
    assert PANEL_CAPS["e5_chameleon_pipeline"]["by"] == ("label",)
    assert PANEL_CAPS["t_ctrl_diagnostic"]["by"] == ("label",)


def test_e3_strata_are_generator_crossed_with_content_type() -> None:
    rows = [
        {"image_id": f"{g}_{c}_{i}", "source_label": g, "subset": c}
        for g in ("GPT-Image2-fake", "Nano-Banana2-fake", "real")
        for c in ("Texture Images", "Structure Images", "Hybrid Images")
        for i in range(60)
    ]
    panel, summary = draw(rows, "e3_matched")
    assert len(summary["by_stratum"]) == 9
    assert "GPT-Image2-fake|Texture Images" in summary["by_stratum"]
    assert len(panel) == min(1500, len(rows))


# ------------------------------------------- the luma view and diagnostic-only ---


def test_the_luma_view_is_not_a_panel_cap() -> None:
    """PANEL_CAPS defines the 12,150 rows and the frozen digest. A seventh key would
    demand a sampler CSV that does not exist and would change the digest."""
    assert not set(panel.LUMA_VIEWS) & set(panel.PANEL_CAPS)


def test_every_luma_view_names_a_real_parent() -> None:
    for name, spec in panel.LUMA_VIEWS.items():
        assert spec["parent"] in panel.PANEL_CAPS, name


def test_the_displaced_slice_belongs_to_the_parent_view() -> None:
    """§8 displaces `e7_dda_coco/FLUX.1`, so the name must be the parent's, not the
    luma view's, or the substitution would never match a scored slice."""
    for spec in panel.LUMA_VIEWS.values():
        for displaced in spec["replaces_in_gate"]:
            assert displaced.split("/", 1)[0] == spec["parent"]


def test_the_displaced_subset_is_one_the_luma_view_carries() -> None:
    for spec in panel.LUMA_VIEWS.values():
        for displaced in spec["replaces_in_gate"]:
            assert displaced.split("/", 1)[1] in spec["subsets"]


def test_gate_slices_replaced_by_luma_collects_them() -> None:
    assert panel.gate_slices_replaced_by_luma() == frozenset({"e7_dda_coco/FLUX.1"})


def test_a_luma_id_round_trips() -> None:
    assert panel.parent_image_id(panel.luma_image_id("abc123")) == "abc123"


def test_a_luma_id_is_recognisable() -> None:
    assert panel.is_luma_id(panel.luma_image_id("abc123"))
    assert not panel.is_luma_id("abc123")


def test_double_suffixing_is_refused() -> None:
    with pytest.raises(panel.PanelError, match="already a luma id"):
        panel.luma_image_id(panel.luma_image_id("abc123"))


def test_a_standard_id_has_no_parent() -> None:
    """Returning it unchanged would let a standard row be joined to luma metadata."""
    with pytest.raises(panel.PanelError, match="not a luma id"):
        panel.parent_image_id("abc123")


def test_t_ctrl_is_the_diagnostic_only_set() -> None:
    assert set(panel.DIAGNOSTIC_ONLY_SETS) == {"t_ctrl"}


def test_diagnostic_only_sets_are_not_panel_views() -> None:
    """`t_ctrl` names a P1 set; `t_ctrl_diagnostic` names its panel. Confusing the two
    would either withhold nothing or empty the panel."""
    assert not panel.DIAGNOSTIC_ONLY_SETS & set(panel.PANEL_CAPS)


# ------------------------------------------------- I-047: the evaluation crop ---
# A panel row whose shorter side is below the evaluation crop cannot be centre-cropped
# to it, so it cannot be scored at all. The first panel was drawn without this check and
# 375 of its 12,150 rows were unscoreable -- discovered as a crash in the S1 pass.


def test_shorter_side_is_the_smaller_dimension() -> None:
    assert panel.shorter_side({"width": "422", "height": "317"}) == 317
    assert panel.shorter_side({"width": "317", "height": "422"}) == 317


def test_shorter_side_accepts_a_float_string() -> None:
    assert panel.shorter_side({"width": "500.0", "height": "400.0"}) == 400


def test_a_row_without_dimensions_has_no_measurable_side() -> None:
    """None is not zero: an unrecorded size says nothing about the image."""
    assert panel.shorter_side({"image_id": "x"}) is None


def test_a_non_numeric_dimension_is_unmeasurable() -> None:
    assert panel.shorter_side({"width": "", "height": "320"}) is None
    assert panel.shorter_side({"width": "wide", "height": "320"}) is None


def test_a_zero_dimension_is_unmeasurable_rather_than_tiny() -> None:
    assert panel.shorter_side({"width": "0", "height": "320"}) is None


def _sized(*pairs: tuple[int, int], label: str = "fake") -> list[dict[str, str]]:
    return [{"image_id": f"i{n}", "width": str(w), "height": str(h), "label": label}
            for n, (w, h) in enumerate(pairs)]


def test_filter_splits_on_the_shorter_side() -> None:
    kept, dropped, unmeasured = panel.filter_by_min_side(
        _sized((400, 400), (422, 317), (320, 320)), 320
    )
    assert [r["image_id"] for r in kept] == ["i0", "i2"]
    assert [r["image_id"] for r in dropped] == ["i1"]
    assert unmeasured == 0


def test_the_threshold_is_inclusive() -> None:
    kept, dropped, _ = panel.filter_by_min_side(_sized((320, 320)), 320)
    assert kept and not dropped


def test_the_filter_is_label_blind() -> None:
    """The same pixels must give the same verdict whatever the row is labelled."""
    for label in ("real", "fake"):
        kept, dropped, _ = panel.filter_by_min_side(
            _sized((400, 400), (200, 200), label=label), 320
        )
        assert len(kept) == 1 and len(dropped) == 1


def test_a_zero_threshold_disables_the_filter() -> None:
    rows = _sized((10, 10))
    kept, dropped, unmeasured = panel.filter_by_min_side(rows, 0)
    assert kept == rows and not dropped and unmeasured == 0


def test_unmeasured_rows_are_kept_and_counted() -> None:
    """Dropping them would silently empty a view whose manifest records no size."""
    rows = [{"image_id": "a"}, *_sized((200, 200))]
    kept, dropped, unmeasured = panel.filter_by_min_side(rows, 320)
    assert [r["image_id"] for r in kept] == ["a"]
    assert len(dropped) == 1 and unmeasured == 1


def test_the_default_evaluation_crop_is_320() -> None:
    assert panel.EVAL_CROP == 320


def _e8(n_ok: int, n_small: int) -> list[dict[str, str]]:
    rows = []
    for i in range(n_ok):
        rows.append({"image_id": f"ok{i}", "generator": "Flux",
                     "width": "512", "height": "512", "label": "fake"})
    for i in range(n_small):
        rows.append({"image_id": f"sm{i}", "generator": "Flux",
                     "width": "512", "height": "200", "label": "fake"})
    return rows


def test_draw_excludes_rows_too_small_for_the_evaluation_crop() -> None:
    """Fails against the pre-I-047 draw, which sampled from every row."""
    drawn, _ = panel.draw(_e8(5, 5), "e8_evalgen")
    assert len(drawn) == 5
    assert all(r["image_id"].startswith("ok") for r in drawn)
    assert all(panel.shorter_side(r) >= panel.EVAL_CROP for r in drawn)


def test_draw_reports_what_the_size_rule_removed() -> None:
    _, summary = panel.draw(_e8(5, 3), "e8_evalgen")
    assert summary["min_side"]["threshold"] == panel.EVAL_CROP
    assert summary["min_side"]["dropped"] == 3
    assert summary["min_side"]["eligible"] == 5
    assert summary["min_side"]["unmeasured"] == 0


def test_draw_can_be_asked_not_to_filter() -> None:
    drawn, summary = panel.draw(_e8(5, 5), "e8_evalgen", min_side=0)
    assert len(drawn) == 10
    assert summary["min_side"]["dropped"] == 0


def test_draw_keeps_rows_whose_manifest_records_no_size() -> None:
    rows = [{"image_id": f"x{i}", "generator": "Flux", "label": "fake"} for i in range(4)]
    drawn, summary = panel.draw(rows, "e8_evalgen")
    assert len(drawn) == 4
    assert summary["min_side"]["unmeasured"] == 4
