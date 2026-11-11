"""Tests for the pre-registered analysis (B8).

Two things are checked here, and both are about a number not quietly meaning something
other than what the pre-registration says it means:

* §8's substitution. On E7's FLUX.1 subset a rule reading only chroma subsampling is a
  perfect classifier, so the standard slice is reported and the luma-only slice is what
  enters M1 and M2. If the luma companion was never scored, the verdict must be refused
  rather than computed over one slice fewer.
* The exact-duplicate caveat. WildRF distributes the same bytes under more than one
  path. The primary number keeps them, because an evaluation set is used as distributed,
  so the duplication has to be stated and quantified instead of removed silently.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("p2_analyze", SCRIPTS / "p2_analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load()


# ------------------------------------------------------------- slice naming ---


def test_an_e7_subset_row_is_named_view_slash_subset() -> None:
    """The §8 displacement list says `e7_dda_coco/FLUX.1`, so slice_key must agree."""
    row = {"view": "e7_dda_coco", "subset": "FLUX.1", "generator": "FLUX.1"}
    assert analyze.slice_key(row) == "e7_dda_coco/FLUX.1"


def test_a_luma_row_is_named_under_its_own_view() -> None:
    row = {"view": "e7_dda_coco_luma", "subset": "FLUX.1", "generator": "FLUX.1"}
    assert analyze.slice_key(row) == "e7_dda_coco_luma/FLUX.1"


def test_the_displaced_slice_name_is_one_slice_key_can_produce() -> None:
    from dbp.eval.panel import gate_slices_replaced_by_luma

    for name in gate_slices_replaced_by_luma():
        view, subset = name.split("/", 1)
        assert analyze.slice_key({"view": view, "subset": subset, "generator": ""}) == name


def test_the_e7_trivial_baseline_reaches_the_flux_slice() -> None:
    assert "4:2:0" in analyze.baseline_for("e7_dda_coco/FLUX.1")


def test_the_luma_slice_also_carries_the_e7_baseline() -> None:
    """The baseline is printed beside the luma number too: it is what the luma view
    exists to defuse, and D-044 wants it adjacent either way."""
    assert analyze.baseline_for("e7_dda_coco_luma/FLUX.1") != ""


# --------------------------------------------------------- exact duplicates ---


def _rows(ids: list[str]) -> list[dict[str, str]]:
    return [{"image_id": i} for i in ids]


def test_identical_hashes_form_a_group() -> None:
    hashes = {"a": "h1", "b": "h1", "c": "h2"}
    assert analyze.exact_duplicate_groups(_rows(["a", "b", "c"]), hashes) == [["a", "b"]]


def test_a_unique_hash_forms_no_group() -> None:
    hashes = {"a": "h1", "b": "h2"}
    assert analyze.exact_duplicate_groups(_rows(["a", "b"]), hashes) == []


def test_rows_without_a_hash_are_not_grouped() -> None:
    """A missing hash is unknown, not equal to another missing hash."""
    assert analyze.exact_duplicate_groups(_rows(["a", "b"]), {}) == []


def test_groups_are_sorted_so_the_kept_row_is_deterministic() -> None:
    hashes = {"b": "h1", "a": "h1"}
    assert analyze.exact_duplicate_groups(_rows(["b", "a"]), hashes) == [["a", "b"]]


def test_a_group_of_three_is_found_whole() -> None:
    hashes = {"a": "h", "b": "h", "c": "h"}
    assert analyze.exact_duplicate_groups(_rows(["a", "b", "c"]), hashes) == [["a", "b", "c"]]


def test_deduplicate_keeps_the_first_of_each_group() -> None:
    rows = _rows(["a", "b", "c"])
    assert [r["image_id"] for r in analyze.deduplicate(rows, [["a", "b"]])] == ["a", "c"]


def test_deduplicate_keeps_one_row_from_a_group_of_three() -> None:
    rows = _rows(["a", "b", "c", "d"])
    out = analyze.deduplicate(rows, [["a", "b", "c"]])
    assert [r["image_id"] for r in out] == ["a", "d"]


def test_deduplicate_with_no_groups_changes_nothing() -> None:
    rows = _rows(["a", "b"])
    assert analyze.deduplicate(rows, []) == rows


def test_deduplicate_preserves_order() -> None:
    rows = _rows(["z", "a", "b"])
    assert [r["image_id"] for r in analyze.deduplicate(rows, [["a", "b"]])] == ["z", "a"]


# ------------------------------------------------------------ panel hashes ---


def test_panel_hashes_are_read_by_image_id(tmp_path: Path) -> None:
    path = tmp_path / "frozen.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "sha256"])
        writer.writeheader()
        writer.writerow({"image_id": "a", "sha256": "f" * 64})
    assert analyze.load_panel_hashes(path) == {"a": "f" * 64}


def test_a_blank_hash_is_dropped_rather_than_stored(tmp_path: Path) -> None:
    """A blank would otherwise group every unhashed row together as duplicates."""
    path = tmp_path / "frozen.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "sha256"])
        writer.writeheader()
        writer.writerow({"image_id": "a", "sha256": ""})
        writer.writerow({"image_id": "b", "sha256": ""})
    assert analyze.load_panel_hashes(path) == {}


def test_a_missing_hash_file_is_empty_rather_than_fatal(tmp_path: Path) -> None:
    assert analyze.load_panel_hashes(tmp_path / "absent.csv") == {}


def test_the_caveat_column_is_in_the_metrics_schema() -> None:
    assert "caveat" in analyze.METRIC_COLUMNS
