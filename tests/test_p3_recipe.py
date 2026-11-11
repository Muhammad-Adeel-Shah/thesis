"""The P3 recipe and its additions (user decisions 2026-10-17: D1, D2 (i)-(iii), D3).

* D1: every non-LOLO run follows the 6,000-step cosine schedule and stops at N* = 5000;
* D2 (i): ``excluded_members`` (BANK-1 = sd15 alone), with P2 and LOLO configs unchanged;
* D2 (ii)/(iii): the per-image held-out score file and its sidecar;
* the bank-size runs: the frozen orders and decoder counts.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.decoders.registry import load_bank_config  # noqa: E402
from dbp.eval import heldout_scores as HS  # noqa: E402
from dbp.eval.run_config import RunConfigError, load_run_configs  # noqa: E402

BANK = load_bank_config(ROOT / "configs" / "decoder_bank.yaml")
P3 = load_run_configs(ROOT / "configs" / "p3_runs.yaml", BANK)
#: Pre-registration section 3 / P3_PLAN.md section 3, at the lineage level.
NEAR = ("kl_4ch_sd1", "kl_4ch_sdxl", "vq_movq_4ch", "kl_16ch", "kl_16ch_qwen",
        "kl_32ch_flux2", "dc_ae_f32", "gan_sr")
FAR = ("kl_4ch_sd1", "gan_sr", "dc_ae_f32", "kl_32ch_flux2", "kl_16ch_qwen", "kl_16ch",
       "vq_movq_4ch", "kl_4ch_sdxl")
COUNTS = {"near": {1: 1, 2: 5, 4: 8, 6: 10, 8: 12}, "far": {1: 1, 2: 4, 4: 6, 6: 9, 8: 12}}


def _trainer():
    alias = "_p3_recipe_trainer"
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / "p2_train_s2.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


# --- D1: the learning rate of a stopped run ------------------------------------------

def _reference(step: int, schedule: int) -> float:
    """The LOLO trainer's schedule, written out independently."""
    warm = min(1.0, (step + 1) / 300)
    return warm * 0.5 * (1.0 + math.cos(math.pi * min(step / schedule, 1.0)))


def test_a_stopped_run_follows_the_6000_step_schedule_at_every_step() -> None:
    import torch

    trainer = _trainer()
    param = torch.nn.Parameter(torch.zeros(1))
    optimiser = torch.optim.AdamW([param], lr=trainer.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda step: trainer.lr_factor(step, 6000))
    for step in range(1, 5001):                       # training steps 1..N* = 5000
        used = optimiser.param_groups[0]["lr"]          # what step `step` trains with
        assert used == pytest.approx(trainer.LEARNING_RATE * _reference(step - 1, 6000),
                                     rel=1e-12, abs=1e-18), step
        optimiser.step()
        scheduler.step()
    # ... and it is not a 5,000-step schedule, whose LR would be ~0 by now.
    assert _reference(4999, 6000) > 0.06 and _reference(4999, 5000) < 1e-6
    assert used == pytest.approx(trainer.LEARNING_RATE * _reference(4999, 6000))


def test_lr_factor_is_the_lolo_formula() -> None:
    trainer = _trainer()
    for step in (0, 1, 150, 299, 300, 2500, 4999, 5000, 5999, 6000, 7000):
        assert trainer.lr_factor(step, 6000) == pytest.approx(_reference(step, 6000))
        assert isinstance(trainer.lr_factor(step, 6000), float)


@pytest.mark.parametrize("extra", [["--stop-at", "5000"],
                                   ["--fixed-steps", "6000", "--stop-at", "6001"],
                                   ["--fixed-steps", "6000", "--stop-at", "-1"]])
def test_stop_at_needs_fixed_steps_and_a_step_inside_the_schedule(extra) -> None:
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "p2_train_s2.py"), "--protocol", "s2b",
         "--root", ".", "--pairs-manifest", "x.csv", *extra],
        capture_output=True, text=True, cwd=ROOT, check=False)
    assert done.returncode == 2 and "--stop-at needs --fixed-steps" in done.stderr + done.stdout


# --- D2 (i): excluded_members -----------------------------------------------------------

def _lineage_only(run) -> list[str]:
    return [s.key for s in BANK.distinct_members()
            if s.lineage not in set(run.excluded_lineages)]


def test_p2_and_lolo_configs_load_unchanged() -> None:
    p2 = load_run_configs(ROOT / "configs" / "p2_runs.yaml", BANK)
    lolo = [r for r in P3.runs if r.run_id.startswith("lolo_f")]
    assert len(lolo) == 8
    for run in [*p2.runs, *lolo]:
        assert run.excluded_members == ()
        assert run.active_members(BANK) == _lineage_only(run), run.run_id
        assert run.active_lineages(BANK) == sorted(
            {s.lineage for s in BANK.distinct_members()} - set(run.excluded_lineages))


def test_bank_1_is_sd15_alone_and_its_twin_is_never_split() -> None:
    bank_1 = P3.by_id("bank_1")
    assert bank_1.active_members(BANK) == ["sd15"]
    assert bank_1.active_lineages(BANK) == ["kl_4ch_sd1"]
    # sd_ft_mse is sd21's alias: excluding sd21 excludes it too.
    assert bank_1.subset_status(BANK, "sd_ft_mse") == "out-of-bank (excluded)"
    assert bank_1.subset_status(BANK, "sd15") == "in-bank"
    assert P3.by_id("bank_1l").active_members(BANK) == ["sd15", "sd21", "sd_ft_ema"]


@pytest.mark.parametrize(("members", "message"), [
    ("[sd_ft_mse]", "names the alias 'sd_ft_mse'; name 'sd21'"),
    ("[sd99]", "not a member of the registry"),
    ("[sd21, sd21]", "duplicates"),
    ("[sdxl]", "already excluded by its lineage"),
    ("[sd15, sd21, sd_ft_ema]", "leave no member"),
])
def test_bad_excluded_members_are_refused(tmp_path: Path, members, message) -> None:
    others = "[kl_4ch_sdxl, vq_movq_4ch, kl_16ch, kl_16ch_qwen, kl_32ch_flux2, dc_ae_f32, gan_sr]"
    path = tmp_path / "runs.yaml"
    path.write_text(f"runs:\n  - run_id: x\n    excluded_lineages: {others}\n"
                    f"    excluded_members: {members}\n", encoding="utf-8")
    with pytest.raises(RunConfigError, match=message):
        load_run_configs(path, BANK)


# --- the bank-size runs (D3) -------------------------------------------------------------

@pytest.mark.parametrize(("order", "lineages"), [("near", NEAR), ("far", FAR)])
def test_bank_k_follows_the_frozen_order_and_decoder_counts(order, lineages) -> None:
    for k in (2, 4, 6):
        run = P3.by_id(f"bank_{k}_{order}")
        assert run.active_lineages(BANK) == sorted(lineages[:k]), (order, k)
        assert len(run.active_members(BANK)) == COUNTS[order][k], (order, k)
    assert len(P3.by_id("bank_1").active_members(BANK)) == COUNTS[order][1]
    assert len(P3.by_id("s2b").active_members(BANK)) == COUNTS[order][8]


def test_every_p3_run_keeps_the_lolo_crop_and_gen_matched() -> None:
    for run in P3.runs:
        assert run.eval_crop == 320 and run.eval_crops_per_image == 1
        assert run.use_gen_matched is (run.run_id != "lolo_f1"), run.run_id


# --- D2 (ii)/(iii): the per-image score file -----------------------------------------

def _table() -> list[dict]:
    return [{"image_id": "a", "member": "", "label": 0, "score": -1.0},
            {"image_id": "b", "member": "", "label": 0, "score": 0.5},
            {"image_id": "a", "member": "sdxl", "label": 1, "score": 2.0},
            {"image_id": "b", "member": "sdxl", "label": 1, "score": 0.0},
            {"image_id": "a", "member": "dcae", "label": 1, "score": 3.0}]


def test_score_file_round_trips_with_its_sidecar(tmp_path: Path) -> None:
    out = tmp_path / "bank_1_seed0_step_05000.csv"
    HS.write_scores(out, _table(), {"run_id": "bank_1_seed0", "step": 5000,
                                    "snapshot_sha256": "x"})
    rows, meta = HS.read_scores(out)
    assert out.read_text(encoding="utf-8").splitlines()[0] == "image_id,member,label,score"
    assert [(r["image_id"], r["member"], r["label"]) for r in rows] == \
        [(r["image_id"], r["member"], r["label"]) for r in _table()]
    assert meta["step"] == 5000 and meta["reals"] == 2 and meta["reconstructions"] == 3
    assert meta["heldout_auc_by_member"] == {"dcae": 1.0, "sdxl": 0.75}  # 3 of 4 pairs
    assert meta["heldout_auc"] == pytest.approx(round(5 / 6, 5))


def test_a_tampered_or_malformed_score_file_is_refused(tmp_path: Path) -> None:
    out = tmp_path / "s.csv"
    HS.write_scores(out, _table(), {})
    out.write_text(out.read_text(encoding="utf-8") + "c,,0,1.0\n", encoding="utf-8")
    with pytest.raises(HS.HeldoutScoreError, match="differs from its sidecar"):
        HS.read_scores(out)
    bad = [*_table(), {"image_id": "c", "member": "sdxl", "label": 0, "score": 1.0}]
    HS.write_scores(out, bad, {})
    with pytest.raises(HS.HeldoutScoreError, match="label 0 with member 'sdxl'"):
        HS.read_scores(out)
    with pytest.raises(HS.HeldoutScoreError, match="non-finite"):
        HS.write_scores(out, [{"image_id": "a", "member": "", "label": 0,
                               "score": float("nan")}], {})


def test_snapshot_digest_sees_any_change(tmp_path: Path) -> None:
    snap = tmp_path / "step_05000"
    (snap / "lora").mkdir(parents=True)
    (snap / "lora" / "adapter_model.safetensors").write_bytes(b"weights")
    (snap / "head.pt").write_bytes(b"head")
    first = HS.snapshot_digest(snap)
    assert first == HS.snapshot_digest(snap)
    (snap / "head.pt").write_bytes(b"head2")
    assert HS.snapshot_digest(snap) != first
    with pytest.raises(HS.HeldoutScoreError, match="empty snapshot"):
        HS.snapshot_digest(tmp_path / "missing")


def test_the_trainer_and_the_scorer_share_one_scoring_path() -> None:
    trainer = (ROOT / "scripts" / "p2_train_s2.py").read_text(encoding="utf-8")
    scorer = (ROOT / "scripts" / "p3_score_heldout.py").read_text(encoding="utf-8")
    for text in (trainer, scorer):
        assert "score_items(" in text and "score_heldout(" in text
        assert "sample_params(" not in text.split("def validate_scores")[0].split(
            "def score_paths")[-1]
    assert json.dumps(HS.SCORE_COLUMNS) == '["image_id", "member", "label", "score"]'


# --- regression (Kaggle 2026-10-18): no unbounded step in the training session ---------

def test_the_rescoring_watchdog_ends_a_hung_process() -> None:
    import time

    script = ROOT / "scripts" / "p3_score_heldout.py"
    code = "\n".join([
        "import importlib.util, logging, sys, time",
        f"spec = importlib.util.spec_from_file_location('w', r'{script}')",
        "m = importlib.util.module_from_spec(spec); sys.modules['w'] = m",
        "spec.loader.exec_module(m)",
        "m.start_watchdog(1.0 / 3600, logging.getLogger('t'))",
        "time.sleep(60)",                                  # a 'hang' far past the deadline
    ])
    started = time.perf_counter()
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=False, timeout=50)
    assert done.returncode == 2
    assert time.perf_counter() - started < 45


def test_every_rescoring_call_in_the_runbook_has_a_deadline() -> None:
    text = (ROOT / "env" / "kaggle_run_p3_train.py").read_text(encoding="utf-8")
    calls = text.count('py("p3_score_heldout.py"')
    assert calls == 2 and text.count('"--deadline-hours", EXPECT["rescore') == 2
    expect = json.loads((ROOT / "env" / "p3_train_expect.json").read_text(encoding="utf-8"))
    assert 0 < expect["rescore_smoke_deadline_hours"] <= expect["rescore_deadline_hours"] <= 2
