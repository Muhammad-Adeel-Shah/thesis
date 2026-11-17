"""scripts/p3b_thesis_outputs.py restates the committed P3b, P4 and P3 records exactly."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_p3b_outputs", ROOT / "scripts" /
                                                  "p3b_thesis_outputs.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_p3b_outputs"] = module
    spec.loader.exec_module(module)
    return module


T = _module()
P3B = json.loads(T.P3B.read_text(encoding="utf-8"))
P4 = json.loads(T.P4.read_text(encoding="utf-8"))
P3 = json.loads(T.P3_FULL.read_text(encoding="utf-8"))


def test_table_10_restates_every_question_and_verdict() -> None:
    rows = T.question_rows(P3B)
    assert [(r["question"], r["rule"]) for r in rows] == [
        ("Q1", "matches"), ("Q1", "beats"), ("Q2", "matches"), ("Q2", "beats"),
        ("Q3", "gain"), ("Q4", "pass"), ("Q5", "method (Q)"), ("Q5", "method (S)"),
        ("Q6", "pass")]
    q1 = rows[0]
    assert q1["role"] == "primary"
    assert q1["seed 0"] == T.points(P3B["seeds"]["0"]["paired"]["dda_native"]["macro_auc_unseen"])
    assert q1["verdict"] == P3B["questions"]["Q1"]["matches"]["verdict"]
    q4 = next(r for r in rows if r["question"] == "Q4")
    assert q4["verdict"] == P3B["questions"]["Q4"]["verdict"]
    assert all(r["seed 1 rule"] in ("pass", "fail") and r["pass rule (pre-registered)"]
               for r in rows)


def test_figure_a_values_are_the_records_and_q1_is_their_difference() -> None:
    entries = {label: m for label, _, m in T.macro_entries(P3B, P4)}
    assert entries["DDA"] == P4["rows"]["dda_native"]["macro_auc_unseen"]
    assert entries["S2-D ViT-B s1"] == P4["reference"]["s2d_p35_seed1"]["macro_auc_unseen"]
    for seed in T.SEEDS:
        vitl = entries[f"S2-D ViT-L s{seed}"]
        pair = P3B["seeds"][seed]["paired"]["dda_native"]["macro_auc_unseen"]
        assert abs((vitl["point"] - entries["DDA"]["point"]) - pair["point"]) < 1e-12


def test_figure_b_uses_dda_h5_vitb_seed0_and_both_vitl_seeds() -> None:
    series = {label: values for label, _, _, values in T.robustness_series(P3B, P4, P3)}
    assert list(series) == ["DDA", "S2-D ViT-B s0", "S2-D ViT-L s0", "S2-D ViT-L s1"]
    assert series["DDA"][1] == P4["h5"]["jpeg50"]["macro_auc"]
    assert series["S2-D ViT-B s0"][2] == P3["confirmatory"]["H5"]["curves"]["s2d_p35_seed0"][
        "resize050"]
    clean = P3B["seeds"]["1"]["robust"]["clean"]["macro_auc"]
    drop = P3B["seeds"]["1"]["robust"]["q6"]["decision"]["jpeg50"]["difference"]
    assert abs(series["S2-D ViT-L s1"][1]["point"] - clean["point"] - drop) < 1e-12


def test_the_figures_stay_within_the_chart_rule(tmp_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    paths = T.figure_macro(T.macro_entries(P3B, P4), tmp_path)
    paths += T.figure_robustness(T.robustness_series(P3B, P4, P3), tmp_path)
    assert sorted(p.name for p in paths) == sorted(
        f"{n}.{s}" for n in ("fig_p3b_macro_auc", "fig_p3b_robustness") for s in ("png", "svg"))
    assert all(0 < p.stat().st_size <= 1_000_000 for p in paths)
