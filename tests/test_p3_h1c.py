"""H1-c (pre-registration section 6.1): LOLO-f vs BANK-1 on lineage f, folds 2-8, paired."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p3_h1c.py"
sys.path.insert(0, str(ROOT / "src"))

from dbp.eval import heldout_scores as HS  # noqa: E402


def _load():
    alias = "_p3_h1c_test"
    spec = importlib.util.spec_from_file_location(alias, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


H1C = _load()
MEMBERS = H1C.fold_members(ROOT / "configs" / "p3_runs.yaml")
REALS = [f"r{i}" for i in range(30)]


def _table(members: list[str], shift: float, rng) -> list[dict]:
    rows = [{"image_id": r, "member": "", "label": 0, "score": float(rng.normal(0, 1))}
            for r in REALS]
    rows += [{"image_id": r, "member": m, "label": 1, "score": float(rng.normal(shift, 1))}
             for m in members for r in REALS]
    return rows


def _fixture(tmp: Path, lolo_shift: float, bank_shift: float, steps=None) -> Path:
    rng = np.random.default_rng(5)
    steps = steps or {f: 5000 for f in H1C.FOLDS}
    scores = tmp / "scores"
    for fold in H1C.FOLDS:
        HS.write_scores(scores / f"lolo_f{fold}_seed0_step_{steps[fold]:05d}.csv",
                        _table(MEMBERS[fold], lolo_shift, rng),
                        {"protocol": f"lolo_f{fold}", "step": steps[fold], "limit": None})
    bank_members = sorted({m for ms in MEMBERS.values() for m in ms} | {"sd21", "sd_ft_ema"})
    HS.write_scores(scores / "bank_1_seed0_step_05000.csv", _table(bank_members, bank_shift, rng),
                    {"run_id": "bank_1_seed0", "step": 5000})
    nstar = tmp / "nstar.json"
    nstar.write_text(json.dumps({"n_star": {"step": 5000},
                                 "nested": {str(f): {"step": s} for f, s in steps.items()}}),
                     encoding="utf-8")
    return nstar


def _run(tmp: Path, nstar: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--nstar", str(nstar), "--scores-dir",
                           str(tmp / "scores"), "--out", str(tmp / "h1c.json")],
                          capture_output=True, text=True, check=False)


def test_folds_and_members_are_the_frozen_table() -> None:
    assert H1C.FOLDS == (2, 3, 4, 5, 6, 7, 8)
    assert MEMBERS[2] == ["sdxl", "sdxl_fp16fix"] and MEMBERS[3] == ["flux", "sd35"]
    assert MEMBERS[6] == ["flux2_klein"] and MEMBERS[8] == ["movq_k22"]
    assert (H1C.MARGIN, H1C.RESAMPLES, H1C.SEED, H1C.LEVEL) == (0.02, 2000, 2026, 0.95)


def test_a_bank_that_separates_better_passes_and_the_reverse_fails(tmp_path: Path) -> None:
    nstar = _fixture(tmp_path / "a", lolo_shift=2.5, bank_shift=0.3,
                     steps={**{f: 5000 for f in H1C.FOLDS}, 6: 4500})
    done = _run(tmp_path / "a", nstar)
    assert done.returncode == 0, done.stderr
    result = json.loads((tmp_path / "a" / "h1c.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "PASS" and result["mean_difference"] > 0.02
    assert [r["fold"] for r in result["folds"]] == list(H1C.FOLDS)
    assert result["inputs"]["lolo_f6"]["step"] == 4500
    assert result["ci"]["resamples"] == 2000 and result["ci"]["lower"] > 0

    nstar = _fixture(tmp_path / "b", lolo_shift=0.3, bank_shift=2.5)
    assert _run(tmp_path / "b", nstar).returncode == 0
    result = json.loads((tmp_path / "b" / "h1c.json").read_text(encoding="utf-8"))
    assert result["verdict"] == "FAIL" and result["mean_difference"] < 0


def test_the_bootstrap_is_paired(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    reals, fakes = rng.normal(0, 1, 40), rng.normal(1, 1, 40)
    same = {f: {"lolo_real": reals, "bank_real": reals, "lolo_fake": fakes, "bank_fake": fakes}
            for f in H1C.FOLDS}
    ci = H1C.paired_bootstrap(same, resamples=50)
    assert ci["lower"] == ci["upper"] == 0.0      # identical models: every draw is 0


def test_missing_or_mismatched_inputs_are_refused(tmp_path: Path) -> None:
    nstar = _fixture(tmp_path, 1.0, 1.0)
    (tmp_path / "scores" / "bank_1_seed0_step_05000.csv").unlink()
    done = _run(tmp_path, nstar)
    assert done.returncode == 1 and "no BANK-1 scores at N* = 5000" in done.stdout + done.stderr
    with pytest.raises(H1C.H1cError, match="different reals"):
        rng = np.random.default_rng(0)
        bad = tmp_path / "bad"
        _fixture(bad, 1.0, 1.0)
        rows = _table(MEMBERS[2], 1.0, rng)
        rows[0]["image_id"] = "other"
        HS.write_scores(bad / "scores" / "lolo_f2_seed0_step_05000.csv", rows,
                        {"protocol": "lolo_f2", "step": 5000, "limit": None})
        H1C.load_inputs(bad / "scores", json.loads((bad / "nstar.json").read_text("utf-8")),
                        5000, MEMBERS)
