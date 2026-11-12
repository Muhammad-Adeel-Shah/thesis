"""P3b scoring: the model record, the pre-flight's model check, per-list resume, fail-fast."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from dbp.backbones.loader import BACKBONES  # noqa: E402
from dbp.data.manifest import file_sha256  # noqa: E402
from dbp.eval.heldout_scores import snapshot_digest  # noqa: E402
from dbp.eval.perturb import H5_DECISION  # noqa: E402

EXPECT = ROOT / "env" / "p3b_score_expect.json"
RUNBOOK = ROOT / "env" / "kaggle_run_p3b_score.py"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------- per-list resume (scorer) ---

def _identity(score, **changes) -> dict:
    model = {"id": "vitl_s2d_p35_seed0", "lora_sha256": "a" * 64, "head_sha256": "b" * 64,
             "snapshot_sha256": "c" * 64}
    return {**score.list_identity(model, "d" * 64, "dinov3-vitl16", "clean"), **changes}


def test_a_finished_list_is_kept_and_never_scored_again(tmp_path: Path) -> None:
    score = _script("p3_final_score")
    csv_path = tmp_path / "panel_b.csv"
    assert score.kept_list(csv_path, _identity(score)) is None   # nothing recorded yet
    sha = score.write_scores(csv_path, [{"image_id": "x"}, {"image_id": "y"}], [0.5, -1.0])
    entry = {"rows": 2, "condition": "clean", "file": str(csv_path), "sha256": sha,
             "seconds": 1.0, "seconds_per_image": 0.5, "timing": {}}
    score.write_list_record(csv_path, _identity(score), entry)
    assert score.kept_list(csv_path, _identity(score))["sha256"] == sha


@pytest.mark.parametrize("change", [{"lora_sha256": "e" * 64}, {"backbone": "dinov3-vitb16"},
                                    {"list_sha256": "f" * 64}, {"condition": "jpeg50"}])
def test_a_record_of_another_model_list_or_backbone_is_refused(tmp_path: Path, change) -> None:
    score = _script("p3_final_score")
    csv_path = tmp_path / "panel_b.csv"
    sha = score.write_scores(csv_path, [{"image_id": "x"}], [0.1])
    score.write_list_record(csv_path, _identity(score, **change), {"rows": 1, "sha256": sha})
    with pytest.raises(score.ScoreError, match="refused, not overwritten"):
        score.kept_list(csv_path, _identity(score))


def test_a_changed_csv_is_refused(tmp_path: Path) -> None:
    score = _script("p3_final_score")
    csv_path = tmp_path / "luma.csv"
    sha = score.write_scores(csv_path, [{"image_id": "x"}], [0.1])
    score.write_list_record(csv_path, _identity(score), {"rows": 1, "sha256": sha})
    csv_path.write_text("image_id,score\nx,0.2\n", encoding="utf-8")
    with pytest.raises(score.ScoreError, match="not the one its record names"):
        score.kept_list(csv_path, _identity(score))


def test_the_crash_seam_is_dry_run_only(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _script("p3_final_score")
    monkeypatch.setenv("DBP_FAULT_EXIT_AFTER_LISTS", "1")
    monkeypatch.delenv("DBP_DRYRUN", raising=False)
    assert score.fault_after_lists() is None
    monkeypatch.setenv("DBP_DRYRUN", "1")
    assert score.fault_after_lists() == 1


# --------------------------------------------------------------------- model record ---

def test_the_record_compares_every_tensor_with_the_checkpoint() -> None:
    import torch

    record = _script("p3b_model_record")
    live = {"base_model.model.layer.0.q_proj.lora_A.default.weight": torch.ones(2, 3)}
    saved = {"base_model.model.layer.0.q_proj.lora_A.weight": torch.ones(2, 3)}
    head = {"weight": torch.zeros(1, 4), "bias": torch.zeros(1)}
    checkpoint = {"model": live, "head": {k: v.clone() for k, v in head.items()}}
    assert record.compare(checkpoint, saved, head) == []
    saved_off = {k: v + 1e-3 for k, v in saved.items()}
    assert any("differs" in p for p in record.compare(checkpoint, saved_off, head))
    assert any("keys differ" in p for p in record.compare(checkpoint, {}, head))


def test_the_record_needs_a_finished_vitl_report() -> None:
    record = _script("p3b_model_record")
    good = {"status": "OK", "mode": "train", "backbone": "dinov3-vitl16",
            "revision": BACKBONES["dinov3-vitl16"]["revision"], "steps_run": 5000,
            "stop_at": 5000}
    assert record.check_report(good, 5000) == []
    assert record.check_report({**good, "backbone": "dinov3-vitb16"}, 5000)
    assert record.check_report(good, 4500)


# ------------------------------------------------------------ pre-flight model check ---

def _layout(root: Path, run: str = "s2d_p35_seed0", step: int = 5000) -> Path:
    snapshot = root / "outputs/p3b/s2" / run / "snapshots" / f"step_{step:05d}"
    (snapshot / "lora").mkdir(parents=True)
    (snapshot / "lora" / "adapter_model.safetensors").write_bytes(b"lora-bytes")
    (snapshot / "lora" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (snapshot / "head.pt").write_bytes(b"head-bytes")
    return snapshot


def _model_case(tmp_path: Path, val_auc: float = 0.999):
    preflight = _script("p3b_score_preflight")
    train = tmp_path / "train"
    snapshot = _layout(train)
    report = {"status": "OK", "mode": "train", "backbone": "dinov3-vitl16",
              "revision": BACKBONES["dinov3-vitl16"]["revision"], "steps_run": 5000,
              "stop_at": 5000, "schedule_steps": 6000, "fixed_steps": 6000,
              "curve": [{"step": 5000, "loss": 0.2, "val_auc": val_auc}]}
    committed = tmp_path / "committed_report.json"
    committed.write_text(json.dumps(report), encoding="utf-8")
    (train / "reports").mkdir()
    (train / "reports" / "p2_s2_s2d_p35_seed0.json").write_bytes(committed.read_bytes())
    vitb = tmp_path / "vitb.json"
    vitb.write_text(json.dumps({"curve": [{"step": 5000, "val_auc": 0.99453}]}),
                    encoding="utf-8")
    spec = {"id": "vitl_s2d_p35_seed0", "run": "s2d_p35_seed0", "protocol": "s2d_p35",
            "seed": 0, "runs_config": "configs/p3_runs.yaml", "step": 5000,
            "training_report": str(committed), "vitb_report": str(vitb)}
    expect = {"backbone": "dinov3-vitl16", "section5": {"n_max": 6000}}
    record = {"runs": {"s2d_p35_seed0": {
        "ok": True, "lora_sha256": file_sha256(snapshot / "lora/adapter_model.safetensors"),
        "head_sha256": file_sha256(snapshot / "head.pt"),
        "snapshot_sha256": snapshot_digest(snapshot)}}}
    return preflight, spec, train, expect, record, snapshot


def test_a_model_equal_to_its_committed_record_is_planned(tmp_path: Path) -> None:
    preflight, spec, train, expect, record, _ = _model_case(tmp_path)
    model, problems = preflight.check_model(spec, train, expect, record)
    assert problems == [] and model["failure_path"] == "PASS"


def test_a_model_whose_hash_differs_from_the_record_is_refused(tmp_path: Path) -> None:
    preflight, spec, train, expect, record, snapshot = _model_case(tmp_path)
    (snapshot / "head.pt").write_bytes(b"another head")
    _, problems = preflight.check_model(spec, train, expect, record)
    assert any("differ from the committed model record" in p for p in problems)


def test_a_report_that_is_not_the_committed_one_is_refused(tmp_path: Path) -> None:
    preflight, spec, train, expect, record, _ = _model_case(tmp_path)
    (train / "reports" / "p2_s2_s2d_p35_seed0.json").write_text("{}", encoding="utf-8")
    _, problems = preflight.check_model(spec, train, expect, record)
    assert any("not the committed" in p for p in problems)


def test_a_model_that_fails_the_failure_path_is_refused(tmp_path: Path) -> None:
    preflight, spec, train, expect, record, _ = _model_case(tmp_path, val_auc=0.90)
    _, problems = preflight.check_model(spec, train, expect, record)
    assert any("failure path" in p for p in problems)


# ----------------------------------------------------------------------- fail-fast ---

def _plan(tmp_path: Path, stage: str = "clean") -> dict:
    lists = {}
    for name, n in (("panel_b", 10), ("luma", 4), ("threshold_reals", 6)):
        path = tmp_path / f"{name}.csv"
        rows = "".join(f"{name}{i},{'test' if i < 8 else 'calibration'}\n" for i in range(n))
        path.write_text("image_id,role\n" + rows, encoding="utf-8")
        lists[name] = {"path": str(path)}
    return {"stage": stage, "jobs": ["0:m1"], "lists": lists,
            "robust": {"conditions": list(H5_DECISION)}}


def _smoke(root: Path, stage: str, warm_images: int, warm_seconds: float) -> None:
    path = root / "reports/p3_final" / f"m1_{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"load_seconds": 30.0, "lists": {"panel_b": {"timing": {
        "warm_images": warm_images, "warm_seconds": warm_seconds}}}}), encoding="utf-8")


EXPECT_TIMES = {"deadline_hours": 7.0, "session_limit_hours": 12.0, "margin_hours": 0.5}


def test_the_projection_uses_the_warm_rate_and_skips_finished_lists(tmp_path: Path) -> None:
    fail = _script("p3b_failfast")
    plan, smoke = _plan(tmp_path), tmp_path / "smoke"
    _smoke(smoke, "clean", 4, 2.0)                  # 0.5 s/image after the warm-up
    record = fail.project(plan, smoke, EXPECT_TIMES, 1.0, tmp_path / "here")
    unit = record["jobs"][0]["models"][0]
    assert unit["rows_to_score"] == 20 and unit["seconds"] == pytest.approx(30.0 + 0.5 * 20)
    done = tmp_path / "here/reports/p3_final_scores/m1/clean"
    done.mkdir(parents=True)
    (done / "panel_b.done.json").write_text("{}", encoding="utf-8")
    again = fail.project(plan, smoke, EXPECT_TIMES, 1.0, tmp_path / "here")
    assert again["jobs"][0]["models"][0]["rows_to_score"] == 10 and again["ok"]
    _smoke(smoke, "robust", 4, 2.0)                 # robust: 8 test rows x 2 conditions
    robust = fail.project(_plan(tmp_path, "robust"), smoke, EXPECT_TIMES, 1.0, tmp_path / "x")
    assert robust["jobs"][0]["models"][0]["rows_to_score"] == 16


def test_a_projection_past_the_time_left_is_refused(tmp_path: Path) -> None:
    fail = _script("p3b_failfast")
    plan, smoke = _plan(tmp_path), tmp_path / "smoke"
    _smoke(smoke, "clean", 4, 4000.0)               # 1,000 s/image: 20 rows ~ 5.6 h
    record = fail.project(plan, smoke, EXPECT_TIMES, 7.0, tmp_path)   # 4.5 h left
    assert not record["ok"] and not record["jobs"][0]["fits"]


def test_a_smoke_with_no_warm_image_cannot_be_projected(tmp_path: Path) -> None:
    fail = _script("p3b_failfast")
    plan, smoke = _plan(tmp_path), tmp_path / "smoke"
    _smoke(smoke, "clean", 0, 0.0)
    with pytest.raises(ValueError, match="no image after the warm-up"):
        fail.project(plan, smoke, EXPECT_TIMES, 0.0, tmp_path)


# ------------------------------------------------------------- expect file, runbook ---

def test_the_expect_scores_the_two_frozen_models_under_their_own_ids() -> None:
    expect = json.loads(EXPECT.read_text("utf-8"))
    ids = [m["id"] for m in expect["models"]]
    assert ids == ["vitl_s2d_p35_seed0", "vitl_s2d_p35_seed1"] and expect["n_models"] == 2
    assert expect["backbone"] == "dinov3-vitl16"
    assert not any((ROOT / "reports/p3_final" / f"{i}_clean.json").exists() for i in ids)
    assert expect["smoke_limit"] > expect["batch"]          # the smoke times a warm batch
    for model in expect["models"]:
        assert (ROOT / model["training_report"]).is_file() and (ROOT / model["vitb_report"]
                                                                 ).is_file()
        assert model["step"] == 5000
    record = ROOT / expect["model_record"]["path"]
    if record.is_file():
        assert file_sha256(record) == expect["model_record"]["sha256"]
    else:
        assert expect["model_record"]["sha256"].startswith("PENDING")


def test_the_runbook_projects_before_scoring_and_resumes_per_list() -> None:
    code = RUNBOOK.read_text("utf-8")
    smoke = code.index('out["smoke"] = py("p2_run_shards.py"')
    failfast = code.index('py("p3b_failfast.py"')
    scoring = code.index('out["score"] = py("p2_run_shards.py"')
    assert code.index('py("p3b_score_preflight.py"') < smoke < failfast < scoring
    assert '"--worker-arg=--resume-lists"' in code[scoring:scoring + 200]
    assert '"--backbone", EXPECT["backbone"]' in code and '"--runtime"' in code
