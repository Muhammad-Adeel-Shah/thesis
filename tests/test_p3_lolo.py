"""P3 LOLO (pre-registration section 3, frozen 2026-10-08): folds, chooser, held-out rows."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval.run_config import RunConfigError, load_run_configs  # noqa: E402

BANK = load_bank_config(ROOT / "configs" / "decoder_bank.yaml")
#: The frozen fold table, docs/P3_PLAN.md section 2.
FOLD_TABLE = {1: "kl_4ch_sd1", 2: "kl_4ch_sdxl", 3: "kl_16ch", 4: "dc_ae_f32", 5: "gan_sr",
              6: "kl_32ch_flux2", 7: "kl_16ch_qwen", 8: "vq_movq_4ch"}


def _load(name: str):
    alias = f"_lolo_test_{name}"
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.modules[alias] = module                           # dataclasses need the module known
    spec.loader.exec_module(module)
    return module


def test_the_eight_folds_are_the_frozen_table() -> None:
    runs = load_run_configs(ROOT / "configs" / "p3_runs.yaml", BANK)
    folds = {int(r.run_id.removeprefix("lolo_f")): r for r in runs.runs
             if r.run_id.startswith("lolo_f")}
    assert sorted(folds) == list(range(1, 9))
    for number, run in folds.items():
        assert run.excluded_lineages == (FOLD_TABLE[number],), number
        assert run.eval_crop == 320
        assert run.use_gen_matched is (number != 1)       # fold 1 also holds out gen_matched
    lineages = {spec.lineage for spec in BANK.distinct_members()}
    assert set(FOLD_TABLE.values()) == lineages          # every lineage held out once


def test_an_alias_is_never_split_from_its_twin() -> None:
    runs = load_run_configs(ROOT / "configs" / "p3_runs.yaml", BANK)
    fold1 = runs.by_id("lolo_f1")
    active = set(fold1.active_members(BANK))
    assert not active & {"sd15", "sd21", "sd_ft_mse", "sd_ft_ema"}


def test_p2_configs_keep_gen_matched_and_a_bad_value_is_refused(tmp_path: Path) -> None:
    for run in load_run_configs(ROOT / "configs" / "p2_runs.yaml", BANK).runs:
        assert run.use_gen_matched
    bad = tmp_path / "runs.yaml"
    bad.write_text("runs:\n  - run_id: x\n    gen_matched: 'no'\n", encoding="utf-8")
    with pytest.raises(RunConfigError, match="gen_matched must be true or false"):
        load_run_configs(bad, BANK)


def _report(root: Path, fold: str, status: str = "OK") -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"p2_s2_{fold}_seed0.json").write_text(
        json.dumps({"status": status, "mode": "train"}), encoding="utf-8")
    run_dir = root / "outputs" / "p3" / "s2" / f"{fold}_seed0"
    (run_dir / "snapshots" / "step_00500").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoint.pt").write_bytes(b"x")


def _next(here: Path, *sources: Path, extra: tuple[str, ...] = ()):
    args = [sys.executable, str(ROOT / "scripts" / "p3_lolo_next.py"),
            "--reports", str(here / "reports")]
    for source in sources:
        args += ["--resume-from", str(source)]
    result = subprocess.run([*args, *extra], capture_output=True, text=True, check=False)
    plan = json.loads((here / "reports" / "p3_lolo_plan.json").read_text(encoding="utf-8"))
    return result.returncode, plan


def test_the_chooser_picks_the_lowest_unfinished_fold_and_carries_the_rest(tmp_path) -> None:
    previous, here = tmp_path / "prev", tmp_path / "here"
    _report(previous, "lolo_f1")
    _report(previous, "lolo_f2")
    _report(previous, "lolo_f3", status="INCOMPLETE")       # interrupted: picked again
    code, plan = _next(here, previous)
    assert code == 0 and plan["selected"] == ["lolo_f3:0"]
    assert plan["complete"] == ["lolo_f1", "lolo_f2"]
    for fold in ("lolo_f1", "lolo_f2"):                      # carried forward
        assert (here / "reports" / f"p2_s2_{fold}_seed0.json").is_file()
        assert (here / "outputs/p3/s2" / f"{fold}_seed0" / "snapshots/step_00500").is_dir()
    assert not (here / "reports" / "p2_s2_lolo_f3_seed0.json").exists()


def test_two_folds_per_session_take_the_interrupted_one_first(tmp_path: Path) -> None:
    previous, here = tmp_path / "prev", tmp_path / "here"
    _report(previous, "lolo_f1")
    _report(previous, "lolo_f2", status="INCOMPLETE")
    code, plan = _next(here, previous, extra=("--count", "2"))
    assert code == 0 and plan["selected"] == ["lolo_f2:0", "lolo_f3:0"]
    for fold in range(1, 8):
        _report(here, f"lolo_f{fold}")
    code, plan = _next(here, extra=("--count", "2"))
    assert plan["selected"] == ["lolo_f8:0"]                # the last session has one fold


def test_the_chooser_override_and_all_complete(tmp_path: Path) -> None:
    here = tmp_path / "here"
    code, plan = _next(here, extra=("--fold", "lolo_f6"))
    assert code == 0 and plan["selected"] == ["lolo_f6:0"]
    for fold in range(1, 9):
        _report(here, f"lolo_f{fold}")
    code, plan = _next(here)
    assert code == 3 and plan["all_complete"] and plan["selected"] == []


def test_heldout_rows_take_validation_reconstructions_of_held_out_members(tmp_path) -> None:
    trainer = _load("p2_train_s2")
    splits, pairs = tmp_path / "splits.csv", tmp_path / "pairs.csv"
    with splits.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_id", "split"])
        writer.writerows([["a__sdxl", "val"], ["b__sdxl", "train"], ["c__sdxl", "val"],
                          ["a__sd15", "val"], ["d__sdxl", "val"]])
    with pairs.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pair_id", "image_id", "set_id", "member", "relative_path"])
        writer.writerows([["a__sdxl", "a", "t_real", "sdxl", "x"],
                          ["b__sdxl", "b", "t_real", "sdxl", "x"],     # train split
                          ["c__sdxl", "c", "t_ctrl", "sdxl", "x"],     # diagnostic-only set
                          ["a__sd15", "a", "t_real", "sd15", "x"],     # not held out
                          ["d__sdxl", "d", "t_real", "sdxl", "x"]])    # flagged real (D-039)
    rows = trainer.heldout_rows(splits, pairs, {"sdxl", "sdxl_fp16fix"}, {"a", "b", "c"})
    assert [r["pair_id"] for r in rows] == ["a__sdxl"]


def test_train_only_completion_needs_an_ok_training_report(tmp_path: Path) -> None:
    queue = _load("p2_run_queue")
    args = type("A", (), {"out_dir": tmp_path / "out", "reports_dir": tmp_path / "rep",
                          "train_only": True})()
    assert queue.is_complete("lolo_f1_seed0", args, [])[0] is False
    (tmp_path / "rep").mkdir()
    (tmp_path / "rep" / "p2_s2_lolo_f1_seed0.json").write_text(
        json.dumps({"status": "OK", "mode": "smoke"}), encoding="utf-8")
    assert queue.is_complete("lolo_f1_seed0", args, [])[0] is False   # a smoke is not a run
    (tmp_path / "rep" / "p2_s2_lolo_f1_seed0.json").write_text(
        json.dumps({"status": "OK", "mode": "train"}), encoding="utf-8")
    assert queue.is_complete("lolo_f1_seed0", args, [])[0] is True


def test_the_lolo_runbook_trains_without_selecting_and_with_the_frozen_budget() -> None:
    expect = json.loads((ROOT / "env" / "p3_lolo_expect.json").read_text(encoding="utf-8"))
    assert "--fixed-steps=6000" in expect["train_args"]              # N_max
    assert "--heldout-every=500" in expect["train_args"]
    assert "--runs-config=configs/p3_runs.yaml" in expect["train_args"]
    assert expect["folds_per_session"] == 2 and expect["devices"] == "0,1"   # one per T4
    assert expect["gpus"] == 2
    text = (ROOT / "env" / "kaggle_run_p3_lolo.py").read_text(encoding="utf-8")
    assert '"--train-only"' in text and "p2_s2_infer" not in text   # no Panel B scoring here
