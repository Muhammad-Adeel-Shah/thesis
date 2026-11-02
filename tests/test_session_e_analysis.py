"""Session E analysis: the I-061 fixes and Amendment 2's gate, one regression each.

* I-061 (a): E7's and E3's slices hold no reals of their own; each is scored against
  its view's real pool, so it has an AUC and enters M2.
* I-061 (b): the bootstrap resamples the *macro* M1/M2, fakes within slice and reals
  within the view's pool, a luma row moving with its parent.
* Amendment 2: the best single is re-maximised in every replicate; P-A's twelve slices
  are asserted; thresholds must be the reals-only rule; GO needs both seeds.
* No silent drops: a non-finite score raises, in the analysis and in complementarity.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.eval.complementarity import (  # noqa: E402
    GateSlice,
    error_contingency,
    jaccard_of_misses,
    macro_bootstrap,
    pairwise,
)
from dbp.eval.metrics import MetricError, roc_auc  # noqa: E402


def _load():
    spec = importlib.util.spec_from_file_location("p2_analyze_e",
                                                  ROOT / "scripts" / "p2_analyze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


analyze = _load()
LOG = logging.getLogger("test")

# --------------------------------------------------------------- a P-A shaped panel --

P_A_FAKES = {
    ("e7_dda_coco", "sdxl-vae", ""): "out-of-bank (excluded)",
    ("e7_dda_coco", "stable-diffusion-3.5-large", ""): "out-of-bank (excluded)",
    ("e7_dda_coco", "FLUX.1", ""): "out-of-bank (excluded)",
    ("e7_dda_coco", "sd-vae-ft-ema", ""): "in-bank",
    ("e3_matched", "GPT-Image2-fake", "GPT-Image2-fake"): "",
    ("e3_matched", "Nano-Banana2-fake", "Nano-Banana2-fake"): "",
    ("e4_wildrf_matched", "reddit", ""): "",
    ("e5_chameleon_pipeline", "", ""): "",
    **{("e8_evalgen", g, g): "" for g in ("Flux", "GoT", "Infinity", "NOVA", "OmniGen")},
    ("t_ctrl_diagnostic", "sd15", ""): "",
}
REAL_VIEWS = ("e7_dda_coco", "e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline",
              "t_ctrl_diagnostic")


def _panel(n: int = 12, seed: int = 0, shift: dict[str, float] | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    shift = shift or {}
    rows: list[dict] = []

    def add(image_id, parent, view, subset, generator, label, in_bank):
        fake = label == "fake"
        row = {"image_id": image_id, "parent_id": parent, "view": view, "subset": subset,
               "generator": generator, "content_type": "", "label": label,
               "in_bank": in_bank}
        for stream in analyze.ALL_STREAMS:
            row[stream] = f"{rng.normal(2.0 * fake + shift.get(stream, 0.0) * fake, 1.0):.6f}"
        rows.append(row)

    for view in REAL_VIEWS:
        for i in range(n):
            add(f"{view}-r{i}", f"{view}-r{i}", view, "real" if view == "e7_dda_coco" else "",
                "", "real", "real" if view == "e7_dda_coco" else "")
    for i in range(n):  # luma reals, one per E7 real
        add(f"e7_dda_coco-r{i}__luma", f"e7_dda_coco-r{i}", "e7_dda_coco_luma", "real", "",
            "real", "real")
    for (view, subset, generator), status in P_A_FAKES.items():
        for i in range(n):
            image_id = f"{view}-{subset}-f{i}"
            add(image_id, image_id, view, subset, generator, "fake", status)
            if subset == "FLUX.1":
                add(f"{image_id}__luma", image_id, "e7_dda_coco_luma", subset, "", "fake",
                    status)
    return rows


def _heads(run_id: str = "p_a_sd1_seed0", val_auc: dict | None = None) -> dict:
    val_auc = val_auc or {"s1": 0.95, "s2": 0.9999, "s3_lr": 1.0, "s3_set": 0.97,
                          "aeroblade": 0.63}
    entry = {key: {"threshold": thr, "rule": "target_fpr_on_reals"}
             for key, thr in (("fpr_1", 2.33), ("fpr_5", 1.645))}
    return {"run_id": run_id, "protocol": run_id.rsplit("_seed", 1)[0],
            "seed": int(run_id.rsplit("_seed", 1)[1]),
            "calibrations": {s: entry for s in analyze.ALL_STREAMS},
            "streams": {s: {"val_auc": v} for s, v in val_auc.items()}}


def _write(tmp_path: Path, rows: list[dict], heads: dict) -> argparse.Namespace:
    scores = tmp_path / f"scores_{heads['run_id']}.csv"
    with scores.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    heads_path = tmp_path / f"p2_heads_{heads['run_id']}.json"
    heads_path.write_text(json.dumps(heads), encoding="utf-8")
    return argparse.Namespace(scores=scores, heads=heads_path,
                              preregistration=ROOT / "docs" / "P2_PREREGISTRATION.md",
                              panel_hashes=None, out_dir=tmp_path / "out", synthetic=True)


# ----------------------------------------------------------------- I-061 (a) slices --


def test_i061a_an_e7_slice_is_scored_against_the_views_reals() -> None:
    rows = _panel()
    slices = analyze.build_slices(rows)
    e7_reals = {k for k, r in enumerate(rows)
                if r["view"] == "e7_dda_coco" and r["label"] == "real"}
    assert set(slices["e7_dda_coco/sdxl-vae"]["reals"]) == e7_reals
    assert set(slices["e3_matched/Nano-Banana2-fake"]["reals"]) == {
        k for k, r in enumerate(rows) if r["view"] == "e3_matched" and r["label"] == "real"}
    assert slices["e8_evalgen/GoT"]["reals"] == []
    assert "e7_dda_coco/real" not in slices


def test_i061a_p_a_m2_holds_the_seven_slices_with_reals(tmp_path: Path) -> None:
    args = _write(tmp_path, _panel(), _heads())
    assert analyze.analyse(args, LOG) == 0
    gate = json.loads((tmp_path / "out" / "p_a_sd1_seed0" / "go_no_go.json").read_text())
    assert gate["gate_slices_m1"] == list(analyze.P_A_UNSEEN_SLICES)
    assert len(gate["gate_slices_m2"]) == 7
    assert "e7_dda_coco/sdxl-vae" in gate["gate_slices_m2"]
    assert "e7_dda_coco_luma/FLUX.1" in gate["gate_slices_m2"]


def test_a_missing_p_a_slice_is_an_error_not_a_smaller_macro(tmp_path: Path) -> None:
    rows = [r for r in _panel() if r["generator"] != "OmniGen"]
    with pytest.raises(analyze.AnalysisError, match="differ from section 3"):
        analyze.analyse(_write(tmp_path, rows, _heads()), LOG)


def test_without_the_luma_slice_the_verdict_is_refused(tmp_path: Path) -> None:
    rows = [r for r in _panel() if r["view"] != "e7_dda_coco_luma"]
    with pytest.raises(analyze.AnalysisError, match="no luma slice"):
        analyze.analyse(_write(tmp_path, rows, _heads()), LOG)
    # Luma reals without luma fakes are an orphan pool: also refused, never ignored.
    rows = [r for r in _panel() if not (r["view"] == "e7_dda_coco_luma" and r["label"] == "fake")]
    with pytest.raises(analyze.AnalysisError, match="belong to no slice"):
        analyze.analyse(_write(tmp_path, rows, _heads()), LOG)


def test_a_blank_or_non_finite_score_raises_naming_the_row(tmp_path: Path) -> None:
    rows = _panel()
    rows[5]["s2"] = ""
    with pytest.raises(analyze.AnalysisError, match=rows[5]["image_id"]):
        analyze.analyse(_write(tmp_path, rows, _heads()), LOG)


def test_thresholds_fitted_on_the_fakes_are_refused(tmp_path: Path) -> None:
    heads = _heads()
    heads["calibrations"]["s2"] = {k: dict(v, rule="target_fpr")
                                   for k, v in heads["calibrations"]["s2"].items()}
    with pytest.raises(analyze.AnalysisError, match="I-062"):
        analyze.analyse(_write(tmp_path, _panel(), heads), LOG)


# ------------------------------------------------------------- I-061 (b) bootstrap --


def _two_slices():
    scores = {"a": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], float),
              "b": np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], float)}
    reals = np.array([0, 1, 2, 3])
    first = GateSlice("x", np.array([4, 5]), ("u4", "u5"), "fake:x", reals,
                      ("r0", "r1", "r2", "r3"), "real:v")
    second = GateSlice("y", np.array([6, 7, 8, 9]), ("u6", "u7", "u8", "u9"), "fake:y",
                       reals, ("r0", "r1", "r2", "r3"), "real:v")
    return [first, second], scores


def test_i061b_the_point_estimate_is_the_macro_not_the_pooled_value() -> None:
    slices, scores = _two_slices()
    scores = {"a": np.array([0, 0, 0, 0, 5, -5, 5, 5, 5, 5], float)}
    boot = macro_bootstrap(slices, scores, {"a": 1.0}, resamples=10)
    assert boot["M1"]["a"][0] == pytest.approx((0.5 + 1.0) / 2)       # macro
    assert boot["M1"]["a"][0] != pytest.approx(5 / 6)                 # pooled would be this
    auc_x = roc_auc([0, 0, 0, 0, 1, 1], scores["a"][[0, 1, 2, 3, 4, 5]])
    auc_y = roc_auc([0, 0, 0, 0, 1, 1, 1, 1], scores["a"][[0, 1, 2, 3, 6, 7, 8, 9]])
    assert boot["M2"]["a"][0] == pytest.approx((auc_x + auc_y) / 2)


def test_the_bootstrap_is_seeded_and_paired() -> None:
    slices, scores = _two_slices()
    one = macro_bootstrap(slices, scores, {"a": 3.0, "b": 3.0}, resamples=50)
    two = macro_bootstrap(slices, scores, {"a": 3.0, "b": 3.0}, resamples=50)
    np.testing.assert_array_equal(one["M2"]["a"], two["M2"]["a"])
    # Paired: the same draw for both streams, so a mirrored stream mirrors every replicate.
    np.testing.assert_allclose(one["M1"]["a"].shape, one["M1"]["b"].shape)


def test_members_of_one_group_must_cover_the_same_units() -> None:
    """A luma real pool missing one parent cannot be resampled with E7's reals."""
    slices, scores = _two_slices()
    broken = GateSlice("y", np.array([6, 7]), ("u6", "u7"), "fake:y", np.array([0, 1, 2]),
                       ("r0", "r1", "r2"), "real:v")
    with pytest.raises(MetricError, match="different units"):
        macro_bootstrap([slices[0], broken], scores, {"a": 1.0, "b": 1.0}, resamples=5)


def test_a_duplicate_unit_in_a_member_raises() -> None:
    slices, scores = _two_slices()
    broken = GateSlice("x", np.array([4, 5]), ("u4", "u4"), "fake:x")
    with pytest.raises(MetricError, match="twice"):
        macro_bootstrap([broken], scores, {"a": 1.0, "b": 1.0}, resamples=5)


def test_a_non_finite_score_in_the_gate_raises() -> None:
    slices, scores = _two_slices()
    scores["a"][4] = np.nan
    with pytest.raises(MetricError, match="non-finite"):
        macro_bootstrap(slices, scores, {"a": 1.0, "b": 1.0}, resamples=5)


# ----------------------------------------------------------- Amendment 2: the gate --


def test_the_best_single_is_re_maximised_in_every_replicate() -> None:
    fused = np.array([0.80, 0.80, 0.80, 0.80])
    boot = {m: {"f": fused, "a": np.array([0.70, 0.75, 0.60, 0.79]),
                "b": np.array([0.60, 0.65, 0.78, 0.50])} for m in ("M1", "M2")}
    result = analyze.gate_result(boot, "f", ("a", "b"))
    assert result["M1"]["best_single"]["max_on_data"] == "a"
    assert result["M1"]["difference_points"] == pytest.approx(10.0)
    # Replicate 2: b (0.78) is the max, so the delta there is 2 points, not 20.
    assert result["M1"]["best_single"]["max_share_in_replicates"] == {"a": 2 / 3, "b": 1 / 3}


def test_go_needs_two_points_and_a_ci_excluding_zero() -> None:
    boot = {m: {"f": np.full(101, 0.90), "a": np.full(101, 0.85)} for m in ("M1", "M2")}
    assert analyze.gate_result(boot, "f", ("a",))["verdict"] == "GO"
    boot = {m: {"f": np.full(101, 0.90), "a": np.full(101, 0.89)} for m in ("M1", "M2")}
    assert analyze.gate_result(boot, "f", ("a",))["verdict"] == "NO-GO"   # 1 point


def test_p_a_is_go_only_if_both_seeds_pass(tmp_path: Path) -> None:
    for seed, verdict in ((0, "GO"), (1, "NO-GO")):
        directory = tmp_path / f"p_a_sd1_seed{seed}"
        directory.mkdir()
        (directory / "go_no_go.json").write_text(json.dumps({
            "run_id": f"p_a_sd1_seed{seed}",
            "governing": {"verdict": verdict, "passes_on": [], "M1": {}, "M2": {}}}))
    combined = analyze.combine_seeds([tmp_path / "p_a_sd1_seed0", tmp_path / "p_a_sd1_seed1"])
    assert combined["verdict"] == "NO-GO"
    assert combined["runs"]["p_a_sd1_seed0"]["verdict"] == "GO"


def test_the_full_analysis_writes_both_analyses(tmp_path: Path) -> None:
    rows = _panel(n=30, shift={"f_mean_rz": 3.0})
    assert analyze.analyse(_write(tmp_path, rows, _heads()), LOG) == 0
    out = tmp_path / "out" / "p_a_sd1_seed0"
    gate = json.loads((out / "go_no_go.json").read_text())
    assert gate["analysis_a"]["best_single_fixed_on_validation"] == "s3_lr"
    assert set(gate["governing"]["M1"]["best_single"]["max_over"]) == set(analyze.SINGLE_STREAMS)
    assert gate["governing"]["verdict"] in ("GO", "NO-GO")
    comp = json.loads((out / "complementarity.json").read_text())
    pair = comp["by_slice"]["e7_dda_coco/sdxl-vae"]["pairs"]["s1|s2"]
    assert "jaccard_of_misses" in pair and "mcnemar" in pair
    assert "governing_members" in comp["by_slice"]["e8_evalgen/GoT"]["f_oracle"]
    with (out / "metrics.csv").open(encoding="utf-8") as handle:
        metrics = list(csv.DictReader(handle))
    sdxl = [m for m in metrics if m["slice"] == "e7_dda_coco/sdxl-vae" and m["stream"] == "s1"]
    assert sdxl[0]["auc"] != "" and sdxl[0]["in_gate"] == "M1+M2"


# ------------------------------------------------------------ complementarity ------


def test_jaccard_of_misses() -> None:
    table = error_contingency([True, False, False, True], [True, False, True, False])
    assert jaccard_of_misses(table) == pytest.approx(1 / 3)
    assert jaccard_of_misses(error_contingency([True, True], [True, True])) is None


def test_complementarity_raises_on_a_non_finite_score_rather_than_dropping_it() -> None:
    y = np.array([0, 1, 0, 1])
    with pytest.raises(MetricError, match="non-finite"):
        pairwise(y, {"a": np.array([0.0, 1.0, np.nan, 1.0]), "b": np.zeros(4)},
                 {"a": 0.5, "b": 0.5})
