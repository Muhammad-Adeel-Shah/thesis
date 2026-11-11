"""EXPLORATORY (post-verdict) X1-X3: the helpers the tables rest on.

* the vectorised slice-calibrated threshold is exactly ``calibrate_on_reals`` (I-062's
  rule), ties included;
* the vectorised AUC is exactly ``roc_auc``;
* paired resampling really pairs (a reconstruction moves with its real);
* the X3b member orders are fixed, start at sd15 and add one new lineage per step;
* LaTeX cells are escaped.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.complementarity import GateSlice  # noqa: E402
from dbp.eval.metrics import roc_auc  # noqa: E402
from dbp.eval.thresholds import calibrate_on_reals  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("p2_exploratory_x_test",
                                                  ROOT / "scripts" / "p2_exploratory_x.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["p2_exploratory_x_test"] = module
    spec.loader.exec_module(module)
    return module


x = _load()


@pytest.mark.parametrize("n", [10, 19, 20, 491, 1864])
def test_slice_threshold_is_calibrate_on_reals(n: int) -> None:
    rng = np.random.default_rng(n)
    reals = np.column_stack([rng.normal(size=n), np.round(rng.normal(size=n), 1),
                             np.zeros(n), np.r_[np.zeros(n - 3), [5.0, 5.0, 5.0]]])
    vectorised = x.slice_thresholds(reals)
    for k in range(reals.shape[1]):
        assert vectorised[k] == calibrate_on_reals(reals[:, k], 0.05, "t").threshold


def test_vectorised_auc_is_roc_auc() -> None:
    rng = np.random.default_rng(1)
    reals, fakes = rng.normal(size=(50, 3)), rng.normal(0.5, 1, size=(40, 3))
    fakes[:, 2] = np.round(fakes[:, 2])
    reals[:, 2] = np.round(reals[:, 2])
    y = np.r_[np.zeros(50), np.ones(40)]
    expected = [roc_auc(y, np.r_[reals[:, k], fakes[:, k]]) for k in range(3)]
    np.testing.assert_allclose(x.auc_columns(reals, fakes), expected)


def test_paired_resampling_moves_a_reconstruction_with_its_real() -> None:
    n = 30
    units = tuple(f"img{i}" for i in range(n))
    item = GateSlice("pairs", np.arange(n, 2 * n), units, "pair", np.arange(n), units, "pair")
    for k, replicate in enumerate(x.draws([item], resamples=20)):
        fakes, reals = replicate[0]
        np.testing.assert_array_equal(fakes - n, reals)
        if k == 0:  # the data: every real exactly once (in unit order)
            np.testing.assert_array_equal(np.sort(reals), np.arange(n))


def test_x3b_orders_are_the_bank_and_add_one_lineage_per_step() -> None:
    bank = load_bank_config(ROOT / "configs" / "decoder_bank.yaml")
    lineage = {s.key: s.lineage for s in bank.distinct_members()}
    n_lineages = len(set(lineage.values()))
    assert n_lineages == 8
    for order in x.X3B_ORDERS.values():
        assert sorted(order) == sorted(lineage) and order[0] == "sd15"
        seen = [lineage[m] for m in order[:n_lineages]]
        assert len(set(seen)) == n_lineages            # a new lineage at every step to k=8
    near, far = x.X3B_ORDERS.values()
    assert near != far and x.X3B_KS == (1, 2, 4, 6, 8, 12)


def test_latex_cells_are_escaped() -> None:
    assert x.tex_escape("s2_full & 5% {x}") == r"s2\_full \& 5\% \{x\}"
    assert x.tex_escape("0.9†") == r"0.9\textsuperscript{\dag}"
    assert x.tex_escape("a\\b") == r"a\textbackslash{}b"


def test_tables_are_labelled_and_written_twice(tmp_path: Path) -> None:
    x.write_table(tmp_path, "t", ["a", "b_c"], [{"a": 1, "b_c": "x_y"}], "cap", ["a", "b_c"])
    csv_text = (tmp_path / "t.csv").read_text(encoding="utf-8")
    tex = (tmp_path / "t.tex").read_text(encoding="utf-8")
    assert csv_text.startswith("# EXPLORATORY (post-verdict)")
    assert "\\toprule" in tex and "x\\_y" in tex and "EXPLORATORY (post-verdict)" in tex
