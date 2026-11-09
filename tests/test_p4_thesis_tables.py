"""scripts/p4_thesis_tables.py restates the committed P4 analysis exactly, all descriptive."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_p4_tables", ROOT / "scripts" /
                                                  "p4_thesis_tables.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_p4_tables"] = module
    spec.loader.exec_module(module)
    return module


T = _module()
A = json.loads(T.ANALYSIS.read_text(encoding="utf-8"))
P3 = json.loads(T.P3_FULL.read_text(encoding="utf-8"))


def test_t9a_restates_the_macro_and_slice_aucs() -> None:
    rows = T.t9a(A)
    assert [r["model"] for r in rows][:4] == ["DDA", "CO-SPY", "FIRE",
                                             "AEROBLADE (ours, from the paper)"]
    dda = rows[0]
    assert dda["macro AUC (4 unseen)"] == T.ci(A["rows"]["dda_native"]["macro_auc_unseen"], 3)
    assert dda["E5 Chameleon"] == T.ci(
        A["rows"]["dda_native"]["slices"]["e5_chameleon_pipeline/fake"]["auc"], 3)
    assert all(r["status"] == T.DESCRIPTIVE for r in rows) and len(rows) == 8


def test_t9c_is_ours_minus_the_baseline() -> None:
    rows = T.t9c(A)
    pair = A["paired"]["s2d_p35_seed0-fire_native"]["macro_auc_unseen"]
    row = next(r for r in rows if r["ours"] == "S2-D s0" and r["minus baseline"] == "FIRE")
    assert row["delta macro AUC, points"] == T.ci_points(pair)
    ours = A["reference"]["s2d_p35_seed0"]["macro_auc_unseen"]["point"]
    base = A["rows"]["fire_native"]["macro_auc_unseen"]["point"]
    assert abs(pair["point"] - (ours - base)) < 1e-12 and len(rows) == 15


def test_t9d_h5_uses_the_committed_p3_h5_numbers() -> None:
    rows = T.t9d_h5(A, P3)
    decision = P3["confirmatory"]["H5"]["decision"]
    assert rows[0]["S2-D s0 minus its clean, points (P3 H5)"] == \
        f"{100 * decision['jpeg50']['difference']:+.1f}"
    curve = P3["confirmatory"]["H5"]["curves"]["s2d_p35_seed0"]["jpeg50"]
    assert A["h5"]["jpeg50"]["s2d_seed0_macro_auc"] == {**curve}


def test_every_table_is_descriptive_and_e7_states_the_home_advantage() -> None:
    for build in (T.t9b, T.t9d_e7, T.t9d_e8, T.t9d_crop308):
        assert all(r["status"] == T.DESCRIPTIVE for r in build(A))
    assert "home benchmark" in T.HOME and "upper bound" in T.CAVEATS
