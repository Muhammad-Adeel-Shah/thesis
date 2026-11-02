"""I-056: streams are joined by image id, with hard checks and no silent drops.

The first ``p2_fit_heads.py`` joined streams by array length: a stream of a different
length was silently left out of both fusions, and equal lengths in different orders
would have been fused row-misaligned. These tests pin the replacement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dbp.eval.fusion import (  # noqa: E402
    FUSION_STREAMS,
    fit_by_id,
    fit_stack_fusion,
    score_by_id,
)
from dbp.eval.join import JoinError, agreed_labels, join_streams, require_streams  # noqa: E402

N = 120


def _tables(seed: int = 0) -> tuple[list[str], np.ndarray, dict[str, tuple[list[str], np.ndarray]]]:
    rng = np.random.default_rng(seed)
    ids = [f"img{i:04d}" for i in range(N)]
    labels = np.array([i % 2 for i in range(N)])
    tables = {
        name: (list(ids), labels * (1.0 + k) + rng.normal(0, 1.0, N))
        for k, name in enumerate(FUSION_STREAMS)
    }
    return ids, labels, tables


def _fit(ids, labels, tables):
    return fit_by_id("stack", (ids, labels), tables, ids, fitted_on="val_test")


def _shuffled(ids: list[str], values: np.ndarray, seed: int = 7):
    order = np.random.default_rng(seed).permutation(len(ids))
    return [ids[i] for i in order], values[order]


# --------------------------------------------- (a) shuffled stream: same result --

def test_a_shuffled_stream_gives_an_identical_fit_and_identical_scores() -> None:
    ids, labels, tables = _tables()
    reference = _fit(ids, labels, tables)
    shuffled = dict(tables, s2=_shuffled(*tables["s2"]))
    refit = _fit(ids, labels, shuffled)
    assert refit.coefficients == pytest.approx(reference.coefficients)
    a = score_by_id(reference, tables, ids, "panel")
    b = score_by_id(reference, shuffled, ids, "panel")
    np.testing.assert_allclose(a, b)


def test_the_old_positional_join_would_have_changed_the_result() -> None:
    """What I-056 was: the same arrays fused by position after one stream is reordered."""
    ids, labels, tables = _tables()
    positional = {name: values for name, (_, values) in tables.items()}
    reordered = dict(positional, s2=_shuffled(*tables["s2"])[1])
    a = fit_stack_fusion(labels, positional, "val")
    b = fit_stack_fusion(labels, reordered, "val")
    assert a.coefficients != pytest.approx(b.coefficients)


# ------------------------------------------------------- (b) missing id: error --

def test_b_one_missing_id_raises_and_names_it() -> None:
    ids, labels, tables = _tables()
    s3_ids, s3_values = tables["s3_lr"]
    tables["s3_lr"] = (s3_ids[1:], s3_values[1:])
    with pytest.raises(JoinError, match="s3_lr is missing 1 .*img0000"):
        _fit(ids, labels, tables)


# ----------------------------------------------------- (c) duplicate id: error --

def test_c_one_duplicate_id_raises_and_names_it() -> None:
    ids, labels, tables = _tables()
    s1_ids, s1_values = tables["s1"]
    tables["s1"] = (s1_ids + ["img0005"], np.append(s1_values, 0.0))
    with pytest.raises(JoinError, match="s1: 1 duplicate id.*img0005"):
        _fit(ids, labels, tables)


def test_a_duplicate_in_the_target_raises() -> None:
    ids, labels, tables = _tables()
    with pytest.raises(JoinError, match="target: 1 duplicate"):
        join_streams(ids + ["img0003"], {"s1": tables["s1"]})


# --------------------------------------------------- (d) missing stream: error --

def test_d_one_missing_stream_raises_for_the_fit() -> None:
    ids, labels, tables = _tables()
    del tables["s3_set"]
    with pytest.raises(JoinError, match="s3_set"):
        _fit(ids, labels, tables)


def test_d_one_missing_stream_raises_for_the_score() -> None:
    ids, labels, tables = _tables()
    fusion = _fit(ids, labels, tables)
    del tables["aeroblade"]
    with pytest.raises(JoinError, match="aeroblade"):
        score_by_id(fusion, tables, ids, "panel")


def test_an_unconfigured_extra_stream_is_refused() -> None:
    ids, labels, tables = _tables()
    tables["s4"] = tables["s1"]
    with pytest.raises(JoinError, match="unexpected \\['s4'\\]"):
        _fit(ids, labels, tables)


def test_the_fusion_records_exactly_the_configured_streams() -> None:
    ids, labels, tables = _tables()
    assert set(_fit(ids, labels, tables).streams) == set(FUSION_STREAMS)


# ------------------------------------------------------------ other hard checks --

def test_an_extra_id_in_a_non_superset_stream_raises() -> None:
    ids, _, tables = _tables()
    s1_ids, s1_values = tables["s1"]
    with pytest.raises(JoinError, match="holds 1 id"):
        join_streams(ids[:-1], {"s1": (s1_ids, s1_values)})


def test_a_declared_superset_may_hold_more_ids() -> None:
    ids, _, tables = _tables()
    out = join_streams(ids[:10], {"s3": tables["s3_lr"]}, supersets=("s3",))
    np.testing.assert_allclose(out["s3"], tables["s3_lr"][1][:10])


def test_the_expected_count_is_enforced() -> None:
    ids, _, tables = _tables()
    with pytest.raises(JoinError, match="expected 11775"):
        join_streams(ids, {"s1": tables["s1"]}, expected_count=11775)


def test_labels_must_agree_across_sources() -> None:
    ids = ["a", "b", "c"]
    good = {"list": {"a": "real", "b": "fake", "c": "fake"}}
    assert agreed_labels(ids, dict(good, split=dict(good["list"]))).tolist() == [0, 1, 1]
    with pytest.raises(JoinError, match="disagree on 1 id.*'b'"):
        agreed_labels(ids, dict(good, split={"a": "real", "b": "real", "c": "fake"}))
    with pytest.raises(JoinError, match="no label for 1"):
        agreed_labels(ids, dict(good, split={"a": "real", "b": "fake"}))


def test_require_streams_lists_missing_and_unexpected() -> None:
    with pytest.raises(JoinError, match="missing \\['s2'\\], unexpected \\['x'\\]"):
        require_streams(["s1", "x"], ["s1", "s2"], "t")
