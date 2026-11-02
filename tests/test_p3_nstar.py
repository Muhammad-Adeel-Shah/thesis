"""The nested N* selector and H1-a (pre-registration sections 3, 4 and 6.1, frozen)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p3_select_nstar.py"
FOLD_TABLE = {1: "kl_4ch_sd1", 2: "kl_4ch_sdxl", 3: "kl_16ch", 4: "dc_ae_f32", 5: "gan_sr",
              6: "kl_32ch_flux2", 7: "kl_16ch_qwen", 8: "vq_movq_4ch"}
GRID = list(range(500, 6001, 500))


def _load():
    alias = "_nstar_test_module"
    spec = importlib.util.spec_from_file_location(alias, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module                           # dataclasses need the module known
    spec.loader.exec_module(module)
    return module


NS = _load()
sys.path.insert(0, str(ROOT / "src"))
from dbp.eval import heldout_scores as HS  # noqa: E402


def _report(fold: int, aucs: list[float], **override) -> dict:
    payload = {
        "run_id": f"lolo_f{fold}_seed0", "protocol": f"lolo_f{fold}", "seed": 0,
        "mode": "train", "status": "OK", "fixed_steps": 6000, "heldout_every": 500,
        "held_out_lineages": [FOLD_TABLE[fold]],
        "curve": [{"step": s, "val_auc": 0.99, "val_logloss": 0.05,
                   "val_tpr_at_1pct_fpr": 0.95, "heldout_auc": a,
                   "heldout_auc_by_member": {"m": a}} for s, a in zip(GRID, aucs, strict=True)],
    }
    payload.update(override)
    return payload


def _write(tmp: Path, curves: dict[int, list[float]],
           override: dict[int, dict] | None = None) -> list[Path]:
    paths = []
    for fold, aucs in curves.items():
        path = tmp / f"p2_s2_lolo_f{fold}_seed0.json"
        path.write_text(json.dumps(_report(fold, aucs, **(override or {}).get(fold, {}))),
                        encoding="utf-8")
        paths.append(path)
    return paths


def _flat(value: float) -> list[float]:
    return [value] * len(GRID)


def _folds(tmp: Path, curves: dict[int, list[float]], override: dict[int, dict] | None = None):
    return NS.load_folds(_write(tmp, curves, override), ROOT / "configs" / "p3_runs.yaml")


def test_frozen_values_match_the_preregistration() -> None:
    assert NS.N_MAX == 6000 and tuple(GRID) == NS.GRID
    assert (NS.G_MIN, NS.CI_LOWER_MIN, NS.FOLD_FLOOR, NS.CI_LEVEL) == (0.90, 0.85, 0.75, 0.975)
    assert (NS.RESAMPLES, NS.BOOTSTRAP_SEED) == (2000, 2026)


def test_nested_step_never_uses_the_folds_own_curve(tmp_path: Path) -> None:
    # Seven folds peak at step 3000; fold 8 alone peaks hard at step 6000 -- enough to
    # move the all-8 mean there, but N*(-8) must ignore it.
    curves = {f: _flat(0.9) for f in range(1, 8)}
    for f in curves:
        curves[f][GRID.index(3000)] = 0.91
    curves[8] = _flat(0.5)
    curves[8][GRID.index(6000)] = 0.99
    result = NS.select(_folds(tmp_path, curves))
    assert result["nested"]["8"]["step"] == 3000
    assert result["n_star"]["step"] == 6000
    assert 8 not in result["nested"]["8"]["folds_averaged"]
    assert result["nested"]["1"]["step"] == 6000           # fold 8 is in fold 1's average


def test_a_tie_takes_the_earliest_step_and_is_listed(tmp_path: Path) -> None:
    curves = {f: _flat(0.9) for f in range(1, 9)}
    for f in curves:
        curves[f][GRID.index(2000)] = 0.95
        curves[f][GRID.index(4500)] = 0.95
    result = NS.select(_folds(tmp_path, curves))
    assert result["n_star"]["step"] == 2000
    assert result["n_star"]["tied_steps"] == [2000, 4500]
    assert all(v["step"] == 2000 for v in result["nested"].values())


def test_ties_are_exact_not_floating_point(tmp_path: Path) -> None:
    # 0.1 + 0.2 != 0.3 in floats; as 1e-5 units the two step sums are equal.
    curves = {f: _flat(0.5) for f in range(1, 9)}
    curves[1][0], curves[2][0], curves[3][0] = 0.1, 0.2, 0.5
    curves[1][1], curves[2][1], curves[3][1] = 0.3, 0.0, 0.5
    for f in range(1, 9):
        curves[f][2:] = [0.0] * 10
    result = NS.select(_folds(tmp_path, curves))
    assert result["n_star"]["tied_steps"] == [500, 1000]


def test_more_than_five_decimals_is_refused() -> None:
    assert NS.to_units(0.99998, "x") == 99998
    with pytest.raises(NS.SelectionError, match="more than 5 decimals"):
        NS.to_units(0.999981, "x")
    with pytest.raises(NS.SelectionError, match="not a finite number"):
        NS.to_units(float("nan"), "x")


@pytest.mark.parametrize(("override", "message"), [
    ({"status": "INCOMPLETE"}, "status"),
    ({"mode": "smoke"}, "mode"),
    ({"fixed_steps": 5000}, "fixed_steps"),
    ({"held_out_lineages": ["gan_sr"]}, "held_out_lineages"),
])
def test_a_report_that_is_not_a_finished_fold_is_refused(tmp_path: Path, override,
                                                         message) -> None:
    curves = {f: _flat(0.9) for f in range(1, 9)}
    with pytest.raises(NS.SelectionError, match=message):
        _folds(tmp_path, curves, {3: override})


def test_missing_duplicate_and_off_grid_folds_are_refused(tmp_path: Path) -> None:
    curves = {f: _flat(0.9) for f in range(1, 8)}
    with pytest.raises(NS.SelectionError, match=r"folds \[8\] have no report"):
        _folds(tmp_path, curves)
    paths = _write(tmp_path, {f: _flat(0.9) for f in range(1, 9)})
    with pytest.raises(NS.SelectionError, match="fold 1 given twice"):
        NS.load_folds([*paths, paths[0]], ROOT / "configs" / "p3_runs.yaml")
    short = _report(2, _flat(0.9))
    short["curve"] = short["curve"][:-1]
    paths[1].write_text(json.dumps(short), encoding="utf-8")
    with pytest.raises(NS.SelectionError, match="not the frozen grid"):
        NS.load_folds(paths, ROOT / "configs" / "p3_runs.yaml")


def test_h1a_verdict_follows_the_frozen_rule() -> None:
    assert NS.h1a_verdict(0.95, [], None)["verdict"] == "PENDING-CI"
    assert NS.h1a_verdict(0.95, [6], None)["verdict"] == "FAIL"       # a fold below 0.75
    assert NS.h1a_verdict(0.89999, [], 0.95)["verdict"] == "FAIL"     # G below 0.90
    assert NS.h1a_verdict(0.95, [], 0.8499)["verdict"] == "FAIL"      # CI lower bound
    assert NS.h1a_verdict(0.90, [], 0.85)["verdict"] == "PASS"        # >= at both bounds


def test_fold_exactly_at_the_floor_is_not_below_it(tmp_path: Path) -> None:
    curves = {f: _flat(0.99) for f in range(1, 9)}
    curves[6] = _flat(0.75)
    table = NS.select(_folds(tmp_path, curves))["h1a_table"]
    assert not any(row["below_floor"] for row in table)


def test_bootstrap_resamples_reals_once_shared_and_fakes_per_fold() -> None:
    rng = np.random.default_rng(1)
    reals = rng.normal(0, 1, 30)
    shared = {1: reals, 2: reals + 0.1}
    fakes = {1: rng.normal(1, 1, 30), 2: rng.normal(2, 1, 45)}
    got = NS.bootstrap_g(shared, fakes, resamples=1, seed=7, level=0.975)
    # The same draw, by hand: reals once, then fold 1's fakes, then fold 2's.
    draw = np.random.default_rng(7)
    idx = draw.integers(0, 30, 30)
    aucs = []
    for f in (1, 2):
        k = fakes[f][draw.integers(0, len(fakes[f]), len(fakes[f]))]
        aucs.append(NS.roc_auc(np.r_[np.zeros(30), np.ones(len(k))], np.r_[shared[f][idx], k]))
    assert got["lower"] == pytest.approx(np.mean(aucs))
    assert got["percentiles"] == pytest.approx([1.25, 98.75])


def test_bootstrap_is_reproducible_and_bounds_a_separable_case() -> None:
    reals = {f: np.zeros(20) for f in (1, 2)}
    fakes = {f: np.ones(20) for f in (1, 2)}
    ci = NS.bootstrap_g(reals, fakes, resamples=50)
    assert ci["lower"] == ci["upper"] == 1.0
    rng = np.random.default_rng(3)
    noisy_r = {f: rng.normal(0, 1, 40) for f in (1, 2)}
    noisy_f = {f: rng.normal(0.5, 1, 40) for f in (1, 2)}
    assert NS.bootstrap_g(noisy_r, noisy_f, resamples=40) == \
        NS.bootstrap_g(noisy_r, noisy_f, resamples=40)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True,
                          text=True, cwd=ROOT, check=False)


def _scores_dir(path: Path, steps: dict[int, int], reals: np.ndarray,
                fakes: dict[int, np.ndarray], **meta) -> Path:
    """One D2-format file per fold (``write_scores``), as p3_score_heldout.py writes."""
    for f, step in steps.items():
        table = [{"image_id": f"r{i}", "member": "", "label": 0, "score": float(x)}
                 for i, x in enumerate(reals)]
        table += [{"image_id": f"r{i}", "member": "m", "label": 1, "score": float(x)}
                  for i, x in enumerate(fakes[f])]
        HS.write_scores(path / f"lolo_f{f}_seed0_step_{step:05d}.csv", table,
                        {"protocol": f"lolo_f{f}", "step": step, "limit": None, **meta})
    return path


def test_cli_selects_once_then_ci_reads_the_frozen_selection(tmp_path: Path) -> None:
    reals = np.linspace(0.0, 1.0, 40)
    fakes = {f: np.linspace(0.5, 1.5, 40) for f in range(1, 9)}
    auc = round(NS.roc_auc(np.r_[np.zeros(40), np.ones(40)], np.r_[reals, fakes[1]]), 5)
    curves = {f: _flat(auc) for f in range(1, 9)}
    paths = _write(tmp_path, curves)
    out = tmp_path / "nstar.json"
    args = ["select", *[x for p in paths for x in ("--report", str(p))], "--out", str(out)]
    first = _run(*args)
    assert first.returncode == 0, first.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["n_star"]["step"] == 500 and payload["h1a"]["verdict"] in ("PENDING-CI",
                                                                             "FAIL")
    assert payload["h1a"]["ci"] is None and payload["h1a"]["ci_status"].startswith("NOT-RUN")
    assert [i["fold"] for i in payload["inputs"]] == list(range(1, 9))
    again = _run(*args)
    assert again.returncode == 2 and "selected once" in again.stderr + again.stdout

    steps = {f: 500 for f in range(1, 9)}
    good = _scores_dir(tmp_path / "good", steps, reals, fakes)
    ci_out = tmp_path / "ci.json"
    done = _run("ci", "--nstar", str(out), "--scores-dir", str(good), "--out", str(ci_out))
    assert done.returncode == 0, done.stderr
    ci = json.loads(ci_out.read_text(encoding="utf-8"))
    assert ci["ci"]["resamples"] == 2000 and ci["ci"]["seed"] == 2026
    assert ci["verdict"] in ("PASS", "FAIL")
    assert set(ci["scores"]) == {str(f) for f in range(1, 9)}

    wrong = _scores_dir(tmp_path / "wrong", {**steps, 4: 1000}, reals, fakes)
    bad = _run("ci", "--nstar", str(out), "--scores-dir", str(wrong), "--out",
               str(tmp_path / "x.json"))
    assert bad.returncode == 1 and "no scores at its N*(-f) = 500" in bad.stderr + bad.stdout

    shifted = _scores_dir(tmp_path / "shifted", steps, reals, {**fakes, 2: fakes[2] - 0.5})
    bad = _run("ci", "--nstar", str(out), "--scores-dir", str(shifted), "--out",
               str(tmp_path / "y.json"))
    assert bad.returncode == 1 and "does not reproduce" in bad.stderr + bad.stdout

    smoke = _scores_dir(tmp_path / "smoke", steps, reals, fakes, limit=2)
    bad = _run("ci", "--nstar", str(out), "--scores-dir", str(smoke), "--out",
               str(tmp_path / "z.json"))
    assert bad.returncode == 1 and "smoke file" in bad.stderr + bad.stdout

    tampered = _scores_dir(tmp_path / "tampered", steps, reals, fakes)
    victim = tampered / "lolo_f3_seed0_step_00500.csv"
    victim.write_text(victim.read_text(encoding="utf-8").replace("0.000000", "0.000001", 1),
                      encoding="utf-8")
    bad = _run("ci", "--nstar", str(out), "--scores-dir", str(tampered), "--out",
               str(tmp_path / "w.json"))
    assert bad.returncode == 1 and "differs from its sidecar" in bad.stderr + bad.stdout
