"""The P3 Panel B analysis on synthetic scores only: every rule gives the verdict it should.

No test reads a real score: the real per-image scores are analysed once, by the committed
code, after this file passes (user, 2026-10-28). Only the committed *list* of Panel B
rows (ids, views, slots; no scores) is read, to check the slot names exist.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_p3_analyze", ROOT / "scripts" /
                                                  "p3_analyze.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_p3_analyze"] = module
    spec.loader.exec_module(module)
    module.RESAMPLES = 200          # speed; the frozen 2,000 is asserted separately
    return module


A = _module()
N = 80                              # rows per test slot
CAL = {"e3_matched": 50, "e4_wildrf_matched": 100, "e5_chameleon_pipeline": 150,
       "e7_dda_coco": 100, "e7_dda_coco_luma": 30}
FAKE_SLOTS = {"e3_matched": ("GPT-Image2-fake", "Nano-Banana2-fake"),
              "e4_wildrf_matched": ("fake",), "e5_chameleon_pipeline": ("fake",),
              "e7_dda_coco": ("sd-vae-ft-ema", "sd-vae-ft-mse", "stable-diffusion-2-1",
                              "sdxl-vae", "stable-diffusion-3.5-large", "FLUX.1"),
              "e7_dda_coco_luma": ("FLUX.1",),
              "e8_evalgen": ("Flux", "GoT")}


def rows_of() -> tuple[list[dict], list[dict], list[dict]]:
    """Synthetic Panel B, luma and threshold-real lists in the committed lists' columns."""
    panel, luma = [], []
    for view, slots in FAKE_SLOTS.items():
        target = luma if view == "e7_dda_coco_luma" else panel
        for slot in slots:
            target += [{"image_id": f"{view}-{slot}-{i}", "label": "fake", "view": view,
                        "role": "test", "slot": slot} for i in range(N)]
        if view != "e8_evalgen":
            target += [{"image_id": f"{view}-r-{i}", "label": "real", "view": view,
                        "role": "test", "slot": "real-test"} for i in range(N)]
            target += [{"image_id": f"{view}-c-{i}", "label": "real", "view": view,
                        "role": "calibration", "slot": "real-calibration"}
                       for i in range(CAL[view])]
    reals = [{"image_id": f"t-{i}", "label": "real", "view": "coco_val" if i < 120 else
              "oi_held_out", "role": "threshold", "slot": ""} for i in range(200)]
    return panel, luma, reals


def scores_of(rows: list[dict], seed: int, fake_mu: float = 3.0,
              real_mu: dict[str, float] | None = None, fake_mu_by: dict | None = None
              ) -> np.ndarray:
    """Normal scores: reals around ``real_mu[view]`` (0 by default), fakes around ``fake_mu``."""
    rng = np.random.default_rng(seed)
    out = []
    for r in rows:
        if r["label"] == "real":
            mu = (real_mu or {}).get(r["view"] if r["role"] != "threshold" else "threshold", 0.0)
        else:
            mu = (fake_mu_by or {}).get((r["view"], r["slot"]), fake_mu)
        out.append(rng.normal(mu, 1.0))
    return np.asarray(out)


def panel_for(seed: int, **kwargs) -> object:
    panel, luma, reals = rows_of()
    return A.build_panel(panel, scores_of(panel, seed, **kwargs),
                         (luma, scores_of(luma, seed + 1, **kwargs)),
                         (reals, scores_of(reals, seed + 2, **kwargs)))


def test_the_frozen_constants_are_the_preregistrations() -> None:
    m = A
    assert (m.SEED, m.PRIMARY, m.SECONDARY, m.TARGET_FPR) == (2026, 0.975, 0.95, 0.05)
    assert (m.H1B_FLOOR, m.H3_MACRO, m.H3_POOL, m.H3_REDUCTION) == (0.85, 0.15, 0.25, 0.50)
    assert (m.H2_MARGIN, m.H4_GAIN, m.H5_POINTS) == (-0.02, 0.02, 0.05)
    assert (m.KSHOT_FPR, m.KSHOT_SHARE, m.ECE_BINS) == (0.075, 0.80, 15)
    source = (ROOT / "scripts" / "p3_analyze.py").read_text(encoding="utf-8")
    assert "RESAMPLES, SEED = 2000, 2026" in source
    assert m.FPR_POOLS == ("e4_wildrf_matched", "e5_chameleon_pipeline")
    assert m.KSHOT == {"e3_matched": (10, 25, 50), "e4_wildrf_matched": (10, 50, 100),
                       "e5_chameleon_pipeline": (10, 50, 100), "e7_dda_coco": (10, 50, 100)}


def test_every_named_slot_exists_in_the_committed_panel_b_list() -> None:
    """Reads only the committed list of rows (no scores)."""
    lists = ROOT / "reports/kaggle_20261027/p3_final_session1/p3_final_lists"
    slots = set()
    for name in ("panel_b.csv", "luma.csv"):
        with (lists / name).open(encoding="utf-8", newline="") as handle:
            slots |= {(r["view"], r["slot"]) for r in csv.DictReader(handle)}
    named = set(A.UNSEEN) | {k for keys in A.H1B.values() for k in keys} | \
        set(A.H1B_UPPER_BOUND.values())
    assert named <= slots, named - slots
    assert {v for v in A.KSHOT} <= {v for v, s in slots if s == "real-calibration"}


def test_a_score_file_must_match_its_list(tmp_path: Path) -> None:
    rows = [{"image_id": "a"}, {"image_id": "b"}]
    path = tmp_path / "s.csv"
    path.write_text("image_id,score\nb,0.1\na,0.2\n", encoding="utf-8")
    with pytest.raises(A.AnalysisError, match="not the list's"):
        A.scores_for(path, rows)
    path.write_text("image_id,score\na,0.1\nb,nan\n", encoding="utf-8")
    with pytest.raises(A.AnalysisError, match="non-finite"):
        A.scores_for(path, rows)


def test_calibration_and_threshold_reals_never_enter_the_test_pools() -> None:
    p = panel_for(1)
    assert len(p.real["e5_chameleon_pipeline"]) == N
    assert len(p.calibration["e5_chameleon_pipeline"]) == CAL["e5_chameleon_pipeline"]
    assert len(p.union) == 200 and len(p.coco_val) == 120
    assert "e8_evalgen" not in p.real


def test_a_paired_bootstrap_of_identical_models_has_zero_width() -> None:
    p = panel_for(2)
    ci = A.bootstrap({"a": p, "b": p}, A.groups_of(A.UNSEEN),
                     lambda q: A.macro_auc(q["a"]) - A.macro_auc(q["b"]))
    assert ci["point"] == ci["lower"] == ci["upper"] == 0.0
    again = A.bootstrap({"a": p}, A.groups_of(A.UNSEEN), lambda q: A.macro_auc(q["a"]))
    assert again == A.bootstrap({"a": p}, A.groups_of(A.UNSEEN), lambda q: A.macro_auc(q["a"]))


def test_h1b_passes_separated_folds_and_fails_the_overlapping_one() -> None:
    good = panel_for(3, fake_mu=4.0)
    bad = panel_for(4, fake_mu_by={("e7_dda_coco", "stable-diffusion-3.5-large"): 0.3})
    out = A.h1b({1: good, 2: good, 3: bad})
    assert [out["folds"][f]["verdict"] for f in "123"] == ["PASS", "PASS", "FAIL"]
    assert out["folds_passing"] == 2 and out["h1b_on_two_of_three"]
    sub = out["folds"]["3"]["subsets"]["e7_dda_coco/stable-diffusion-3.5-large"]
    assert sub["level"] == 0.975 and sub["point"] < 0.85
    assert "e7_dda_coco/FLUX.1" in out["folds"]["3"]["upper_bound"]


def _h3_panels(s2d_e5: float = 0.0) -> dict:
    # full_bank: E4/E5 reals sit far above the threshold reals -> high FPR.
    fb = panel_for(5, real_mu={"e4_wildrf_matched": 2.5, "e5_chameleon_pipeline": 2.5})
    s2d = {"e4_wildrf_matched": 0.0, "e5_chameleon_pipeline": s2d_e5}
    return {"s2d_p35_seed0": panel_for(6, real_mu=s2d), "s2d_p35_seed1": panel_for(7, real_mu=s2d),
            "full_bank_seed0": fb}


def test_h3_passes_when_both_seeds_cut_false_alarms() -> None:
    out = A.h3(_h3_panels())
    assert out["verdict"] == "PASS"
    seed = out["seeds"]["s2d_p35_seed0"]
    assert seed["macro_fpr"] <= 0.15 and seed["relative_reduction"]["level"] == 0.975
    assert set(seed["three_step"]) == {"e3_matched", "e4_wildrf_matched",
                                       "e5_chameleon_pipeline", "macro_e4_e5"}


def test_h3_fails_when_one_pool_is_above_25_percent() -> None:
    out = A.h3(_h3_panels(s2d_e5=1.5))
    assert out["verdict"] == "FAIL"
    assert not out["seeds"]["s2d_p35_seed0"]["criteria"]["no_pool_above_25"]


def test_h2_non_inferior_only_within_two_points() -> None:
    fb = panel_for(8)
    same = {"s2d_p35_seed0": panel_for(8), "s2d_p35_seed1": panel_for(8), "full_bank_seed0": fb}
    assert A.h2(same)["four_unseen_slices"]["verdict"] == "PASS"
    worse = {m: panel_for(9, fake_mu=1.0) for m in ("s2d_p35_seed0", "s2d_p35_seed1")}
    out = A.h2({**worse, "full_bank_seed0": fb})
    assert out["four_unseen_slices"]["verdict"] == "FAIL" and out["without_e3"]["verdict"] == "FAIL"


def test_h4_needs_two_points_and_a_ci_excluding_zero() -> None:
    names = {m for points in A.BANK_CURVE.values() for _, _, m in points} | {"bank_1l_seed0"}
    panels = {m: panel_for(10, fake_mu=1.0) for m in names}
    panels["s2b_seed0"] = panel_for(11, fake_mu=3.0)
    out = A.h4(panels, {"bank_1_seed0": 0.9})
    assert out["verdict"] == "PASS" and out["curves"]["near"][0]["heldout_lineage_val_auc"] == 0.9
    panels["s2b_seed0"] = panel_for(10, fake_mu=1.0)
    assert A.h4(panels, {})["verdict"] == "FAIL"


def test_h3k_quantile_controls_fpr_and_fails_when_the_pool_is_off_target() -> None:
    model = panel_for(12)
    panels = {"s2d_p35_seed0": model, "s2d_p35_seed1": panel_for(13)}
    out = A.h3k(panels)
    assert out["verdict"]["Q"] == "PASS"
    cell = out["seeds"]["s2d_p35_seed0"]["views"]["e5_chameleon_pipeline"]["Q"]["50"]
    assert cell["j"] == 1 and cell["draws"] == 1000 and cell["share_fpr_le_7_5"] >= 0.8
    single = out["seeds"]["s2d_p35_seed0"]["views"]["e4_wildrf_matched"]["Q"]["100"]
    assert single["draws"] == 1 and "single_draw_fpr" in single
    # Calibration reals far below the test reals: (Q)'s threshold is too low.
    off = panel_for(14)
    off.calibration["e4_wildrf_matched"] = off.calibration["e4_wildrf_matched"] - 3.0
    worse = A.h3k({"s2d_p35_seed0": off, "s2d_p35_seed1": panel_for(13)})
    assert worse["verdict"]["Q"] == "FAIL"


def test_h3k_shift_recentres_a_shifted_pool() -> None:
    # Test and calibration reals both sit 2 above the threshold reals: (S) shifts them back.
    shifted = {"e4_wildrf_matched": 2.0, "e5_chameleon_pipeline": 2.0}
    model = panel_for(15, real_mu=shifted)
    for view in shifted:
        model.calibration[view] = model.calibration[view] + 2.0
    out = A.kshot_view(model, "e5_chameleon_pipeline", model.union, A.k0_threshold(model))
    assert out["k0"]["fpr"] > 0.3 and out["S"]["50"]["mean_fpr"] < 0.15


def test_h5_within_five_points_and_its_failure() -> None:
    def conditions(drop: float) -> dict:
        out = {}
        for model in A.H5_MODELS:
            by = {c: panel_for(16) for c in A.CONDITIONS}
            by["jpeg50"] = panel_for(16, fake_mu=3.0 - drop)
            out[model] = by
        return out
    assert A.h5(conditions(0.0))["verdict"] == "PASS"
    out = A.h5(conditions(2.5))
    assert out["verdict"] == "FAIL" and not out["decision"]["jpeg50"]["within_5_points"]
    assert set(out["curves"]["s2d_p35_seed0"]) == set(A.CONDITIONS)


def test_ece_is_zero_for_calibrated_and_large_for_confident_errors() -> None:
    p = np.array([0.1] * 10 + [0.9] * 10)
    y = np.array([0] * 9 + [1] + [1] * 9 + [0])
    assert A.ece(p, y) == pytest.approx(0.0)
    assert A.ece(np.array([0.95] * 10), np.zeros(10)) == pytest.approx(0.95)


def test_describe_reports_every_slice_and_e8_without_an_auc() -> None:
    out = A.describe(panel_for(17))
    assert "auc" not in out["slices"]["e8_evalgen/Flux"]
    assert "tpr_k0_union" in out["slices"]["e8_evalgen/Flux"]
    assert np.isfinite(out["threshold_union"]) and np.isfinite(out["threshold_coco_only"])
    assert set(out["pools"]) >= {"e3_matched", "e4_wildrf_matched", "e5_chameleon_pipeline"}
