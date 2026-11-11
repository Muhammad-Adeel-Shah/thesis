"""Tests for the grouped, stratified train/val split (B1, D-046)."""

from __future__ import annotations

import pytest

from dbp.data.splits import (
    NO_CATEGORY,
    SPLIT_SEED,
    TRAIN,
    TRAIN_FRACTION,
    VAL,
    SplitError,
    apply_split,
    assign_group_splits,
    caption_source_stem,
    coco_stem,
    corpus_supercategory_frequency,
    stratum_label,
    summarise_split,
    verify_no_content_crosses,
)


def test_coco_stem_strips_directory_and_suffix() -> None:
    assert coco_stem("000000000009.jpg") == "000000000009"
    assert coco_stem("train2017/000000000025.jpg") == "000000000025"
    assert coco_stem("train2017\\000000000030.jpg") == "000000000030"
    assert coco_stem("0001.png") == "0001"


def test_caption_source_stem_follows_the_caption() -> None:
    assert caption_source_stem("coco_000000000009") == "000000000009"


def test_caption_source_stem_refuses_to_guess() -> None:
    """A gen row that lost its prefix must fail loudly, not land on a default side."""
    with pytest.raises(SplitError, match="does not start with"):
        caption_source_stem("2ab2fdc72c37441b")


def test_stratum_label_picks_the_rarest_present() -> None:
    frequency = {"person": 1600, "vehicle": 300, "appliance": 20}
    assert stratum_label({"person", "vehicle", "appliance"}, frequency) == "appliance"
    assert stratum_label({"person", "vehicle"}, frequency) == "vehicle"
    assert stratum_label({"person"}, frequency) == "person"


def test_stratum_label_breaks_ties_alphabetically() -> None:
    frequency = {"food": 10, "animal": 10}
    assert stratum_label({"food", "animal"}, frequency) == "animal"
    assert stratum_label(["animal", "food"], frequency) == "animal"


def test_unannotated_images_form_their_own_stratum() -> None:
    assert stratum_label(set(), {"person": 5}) == NO_CATEGORY


def test_corpus_frequency_counts_images_not_annotations() -> None:
    per_image = {"a": ["person", "person", "vehicle"], "b": ["person"], "c": []}
    assert corpus_supercategory_frequency(per_image) == {"person": 2, "vehicle": 1}


def _groups(n: int, strata: int = 4) -> dict[str, str]:
    return {f"g{i:04d}": f"s{i % strata}" for i in range(n)}


def test_split_is_roughly_the_approved_fraction() -> None:
    assignment = assign_group_splits(_groups(1000))
    train = sum(1 for side in assignment.values() if side == TRAIN)
    assert abs(train / 1000 - TRAIN_FRACTION) < 0.01


def test_every_stratum_is_split_not_just_the_corpus() -> None:
    groups = _groups(400, strata=8)
    summary = summarise_split(groups, assign_group_splits(groups))
    for stratum, counts in summary["by_stratum"].items():
        total = counts[TRAIN] + counts[VAL]
        assert counts[VAL] > 0, f"{stratum} contributed nothing to validation"
        assert abs(counts[TRAIN] / total - TRAIN_FRACTION) < 0.1, stratum


def test_the_same_seed_gives_the_same_split() -> None:
    groups = _groups(500)
    assert assign_group_splits(groups) == assign_group_splits(groups)


def test_a_different_seed_gives_a_different_split() -> None:
    groups = _groups(500)
    assert assign_group_splits(groups) != assign_group_splits(groups, seed=SPLIT_SEED + 1)


def test_adding_a_stratum_does_not_redraw_the_others() -> None:
    """Per-stratum seeding: a new supercategory must not invalidate old calibrations."""
    groups = _groups(300, strata=3)
    before = assign_group_splits(groups)
    extended = {**groups, **{f"x{i}": "brand_new" for i in range(20)}}
    after = assign_group_splits(extended)
    assert all(after[g] == before[g] for g in groups)


def test_a_stratum_of_two_gives_one_to_each_side() -> None:
    assignment = assign_group_splits({"a": "tiny", "b": "tiny"})
    assert sorted(assignment.values()) == [TRAIN, VAL]


def test_a_stratum_of_one_goes_to_train() -> None:
    assert assign_group_splits({"a": "alone"}) == {"a": TRAIN}


def test_train_fraction_must_be_a_fraction() -> None:
    with pytest.raises(SplitError, match="must be in"):
        assign_group_splits(_groups(10), train_fraction=1.0)


def test_apply_split_fills_the_column_from_the_group() -> None:
    rows = [{"image_id": "i1", "source_image_id": "g0"},
            {"image_id": "i2", "source_image_id": "g0"},
            {"image_id": "i3", "source_image_id": "g1"}]
    out = apply_split(rows, {"g0": TRAIN, "g1": VAL}, "source_image_id")
    assert [r["split"] for r in out] == [TRAIN, TRAIN, VAL]
    assert "split" not in rows[0], "input rows were mutated"


def test_apply_split_refuses_an_unjoined_row() -> None:
    with pytest.raises(SplitError, match="no split assignment"):
        apply_split([{"source_image_id": "missing"}], {"g0": TRAIN}, "source_image_id")


def test_verify_catches_a_real_and_its_reconstruction_on_opposite_sides() -> None:
    """The failure the whole module exists to prevent."""
    reals = [{"split": TRAIN, "content_id": "000000000009"}]
    pairs = [{"split": VAL, "content_id": "000000000009"}]
    with pytest.raises(SplitError, match="both splits"):
        verify_no_content_crosses({"reals": reals, "pairs": pairs})


def test_verify_catches_a_generation_placed_by_its_matching_partner() -> None:
    """D-046: gen_matched must follow the caption, not source_real_id."""
    reals = [{"split": TRAIN, "content_id": "000000000009"},
             {"split": VAL, "content_id": "000000000025"}]
    # Written from image ...009's caption but JPEG-matched to ...025.
    gen = [{"split": VAL, "content_id": "000000000009"}]
    with pytest.raises(SplitError, match="both splits"):
        verify_no_content_crosses({"reals": reals, "gen": gen})


def test_verify_passes_and_reports_counts_when_nothing_crosses() -> None:
    reals = [{"split": TRAIN, "content_id": "a"}, {"split": VAL, "content_id": "b"}]
    pairs = [{"split": TRAIN, "content_id": "a"}, {"split": VAL, "content_id": "b"}]
    report = verify_no_content_crosses({"reals": reals, "pairs": pairs})
    assert report["crossing"] == []
    assert report["distinct_content"] == 2
    assert report["counts"]["pairs"] == {TRAIN: 1, VAL: 1}


def test_verify_rejects_an_unknown_split_value() -> None:
    with pytest.raises(SplitError, match="split must be"):
        verify_no_content_crosses({"s": [{"split": "test", "content_id": "a"}]})


def test_summary_totals_match_the_assignment() -> None:
    groups = _groups(250, strata=5)
    assignment = assign_group_splits(groups)
    summary = summarise_split(groups, assignment)
    assert summary["seed"] == SPLIT_SEED
    assert summary["groups"] == 250
    assert summary["totals"][TRAIN] + summary["totals"][VAL] == 250
    assert sum(c[TRAIN] for c in summary["by_stratum"].values()) == summary["totals"][TRAIN]
