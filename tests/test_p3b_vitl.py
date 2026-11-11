"""P3b ViT-L smoke tooling: the pinned backbone, its dry-run stand-in, the probe that
measures the unchanged trainer, and the smoke report's projection."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dbp.backbones.loader import BACKBONES, load_backbone  # noqa: E402


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module   # a dataclass in the module looks itself up here
    spec.loader.exec_module(module)
    return module


def test_vitl16_is_pinned_like_vitb16() -> None:
    large, base = BACKBONES["dinov3-vitl16"], BACKBONES["dinov3-vitb16"]
    assert large["revision"] == "ea8dc2863c51be0a264bab82070e3e8836b02d51"
    assert len(large["revision"]) == 40 and int(large["revision"], 16) >= 0
    assert (large["patch_size"], large["gated"], large["default"]) == (16, True, False)
    assert large["repo"] == base["repo"].replace("vitb16", "vitl16")


def test_the_dry_run_stand_in_has_vitl16s_width(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DBP_DRYRUN", "1")
    backbone = load_backbone("dinov3-vitl16", "cpu")
    assert backbone.hidden_size == 1024 and backbone.patch_size == 16
    assert backbone.config["num_register_tokens"] == 4 and backbone.config["dry_run_random_init"]
    with pytest.raises(Exception, match="CPU dry-runs only"):
        load_backbone("dinov3-vitl16", "cuda")


def test_the_probe_reports_the_guarded_trainer_hash() -> None:
    probe = _script("p3b_smoke_probe")
    guard = re.search(r'TRAINER_SHA256 = "([0-9a-f]{64})"',
                      (ROOT / "tests" / "test_p3_s2d.py").read_text("utf-8")).group(1)
    assert probe.trainer_sha256() == guard


def test_the_probe_runs_smokes_only(tmp_path: Path) -> None:
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "p3b_smoke_probe.py"),
                           "--gradient-checkpointing", "off", "--out", str(tmp_path / "p.json"),
                           "--", "--protocol", "s2d_p35"], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 2 and "smokes only" in done.stdout
    assert not (tmp_path / "p.json").exists()


EXPECT = {"backbone": "dinov3-vitl16", "smoke_steps": 50,
          "projection": {"steps": 5000, "evaluations": 10,
                         "vitb_reference": "reports/kaggle_20261023/p3_s2d_session1/"
                                           "p2_s2_s2d_p35_seed0.json"},
          "smoke_runs": [{"device": "cuda", "gradient_checkpointing": "on", "role": "recipe"},
                         {"device": "cuda", "gradient_checkpointing": "off", "role": "measure"}]}


def _smoke(scratch: Path, mode: str, probe: dict, status: str = "SMOKE-OK") -> None:
    root = scratch / f"dbp_p3b_smoke_gc_{mode}"
    (root / "reports").mkdir(parents=True)
    (root / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    (root / "reports" / "p2_s2_s2d_p35_seed0.json").write_text(
        json.dumps({"status": status, "steps_run": 50}), encoding="utf-8")


def test_the_projection_and_an_out_of_memory_measurement(tmp_path: Path) -> None:
    report = _script("p3b_smoke_report")
    _smoke(tmp_path, "on", {"step_seconds_after_warmup": 2.0, "out_of_memory": False,
                            "exit_code": 0, "peak_gpu_memory_allocated_gib": 11.2})
    _smoke(tmp_path, "off", {"step_seconds_after_warmup": None, "out_of_memory": True,
                             "exit_code": 1}, status="NO REPORT")
    expect = tmp_path / "expect.json"
    expect.write_text(json.dumps(EXPECT), encoding="utf-8")
    out = tmp_path / "out.json"
    argv = sys.argv
    sys.argv = ["x", "--expect", str(expect), "--scratch", str(tmp_path), "--out", str(out)]
    try:
        assert report.main() == 0
    finally:
        sys.argv = argv
    record = json.loads(out.read_text("utf-8"))
    projection = record["projection"]
    vitb = json.loads((ROOT / EXPECT["projection"]["vitb_reference"]).read_text("utf-8"))
    step = vitb["measured_step_seconds_train_loop"]
    factor = vitb["gpu_hours"] * 3600 / (vitb["steps_run"] * step)
    assert projection["gpu_hours_per_run"] == round(2.0 * 5000 * factor / 3600, 2)
    assert projection["vitl_over_vitb_step_time"] == round(2.0 / 0.80041, 2)
    assert record["gradient_checkpointing_needed"].startswith("yes")


def test_the_report_never_fails_the_run(tmp_path: Path) -> None:
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "p3b_smoke_report.py"),
                           "--expect", str(tmp_path / "missing.json"), "--scratch",
                           str(tmp_path), "--out", str(tmp_path / "o.json")],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0 and "report unavailable" in done.stdout


# ------------------------------------------------- P3b training session (frozen prereg) ---

TRAIN_EXPECT = ROOT / "env" / "p3b_vitl_train_expect.json"
TRAIN_RUNBOOK = ROOT / "env" / "kaggle_run_p3b_vitl_train.py"


def _vitl_report(**changes) -> dict:
    report = {"status": "OK", "mode": "train", "backbone": "dinov3-vitl16",
              "revision": BACKBONES["dinov3-vitl16"]["revision"], "steps_run": 5000,
              "curve": [{"step": s, "loss": 0.2, "val_auc": 0.99} for s in range(500, 5001, 500)]}
    report.update(changes)
    return report


def _check(report, vitb_auc: float = 0.99453) -> dict:
    check = _script("p3b_train_check")
    vitb = {"curve": [{"step": 5000, "val_auc": vitb_auc}]}
    return check.check_seed(report, vitb, "dinov3-vitl16",
                            BACKBONES["dinov3-vitl16"]["revision"], 5000)


def test_a_healthy_vitl_run_passes_the_failure_path() -> None:
    assert _check(_vitl_report())["verdict"] == "PASS"


@pytest.mark.parametrize("change, reason", [
    ({"curve": [{"step": 500, "loss": float("nan"), "val_auc": 0.9},
                {"step": 5000, "loss": 0.2, "val_auc": 0.99}]}, "NaN or divergence"),
    ({"curve": [{"step": 5000, "loss": 0.2, "val_auc": float("inf")}]}, "NaN or divergence"),
    ({"status": "FAILED"}, "status"),
    ({"backbone": "dinov3-vitb16"}, "backbone"),
    ({"steps_run": 4500}, "steps_run"),
    ({"curve": [{"step": 5000, "loss": 0.2, "val_auc": 0.944}]}, "more than 5 points below"),
])
def test_the_failure_path_stops_every_case_section_5_names(change, reason) -> None:
    result = _check(_vitl_report(**change))
    assert result["verdict"] == "FAIL" and any(reason in r for r in result["reasons"])


def test_exactly_5_points_below_is_not_more_than_5() -> None:
    at_floor = _vitl_report(curve=[{"step": 5000, "loss": 0.2, "val_auc": 0.94453}])
    assert _check(at_floor)["verdict"] == "PASS"
    assert _check(None)["reasons"][0] == "no training report"


def test_the_training_expect_changes_only_the_backbone() -> None:
    expect = json.loads(TRAIN_EXPECT.read_text("utf-8"))
    s2d = json.loads((ROOT / "env" / "p3_s2d_expect.json").read_text("utf-8"))
    assert expect["train_args"] == [*s2d["train_args"], "--backbone=dinov3-vitl16"]
    for key in ("inputs", "oi", "corpus", "gen_matched", "nstar_sha256", "choice", "imports"):
        assert expect[key] == s2d[key], key
    assert expect["queue_resolved"] == ["s2d_p35:0", "s2d_p35:1"]
    assert expect["out_dir"] == "outputs/p3b/s2" and expect["devices"] == "0,1"
    check = _script("p3b_train_check")
    assert check.stop_step(expect["train_args"]) == 5000
    # the ViT-B reports it compares against are the committed ones, at 0.99453 / 0.99423
    aucs = {s: check.val_auc_at(json.loads((ROOT / p).read_text("utf-8")), 5000)
            for s, p in expect["vitb_reports"].items()}
    assert aucs == {"0": 0.99453, "1": 0.99423}


def test_the_committed_vitb_reports_would_fail_only_on_their_backbone(tmp_path: Path) -> None:
    """The check reads real training reports: ViT-B's, posing as ViT-L's, fail on the
    backbone and the revision alone."""
    expect = json.loads(TRAIN_EXPECT.read_text("utf-8"))
    for seed, relative in expect["vitb_reports"].items():
        (tmp_path / f"p2_s2_s2d_p35_seed{seed}.json").write_bytes((ROOT / relative).read_bytes())
    out = tmp_path / "check.json"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "p3b_train_check.py"),
                           "--expect", str(TRAIN_EXPECT), "--reports-dir", str(tmp_path),
                           "--out", str(out)], capture_output=True, text=True, check=False)
    assert done.returncode == 1
    record = json.loads(out.read_text("utf-8"))
    for seed in ("0", "1"):
        reasons = record["seeds"][seed]["reasons"]
        assert len(reasons) == 1 and reasons[0].startswith("backbone 'dinov3-vitb16'"), reasons


def test_the_runbook_never_resumes_from_a_vitb_output() -> None:
    code = TRAIN_RUNBOOK.read_text("utf-8")
    assert '"notebooks/*/*/Implementation/thesis_dbp/outputs/p3b/s2"' in code
    assert "outputs/p3/s2" not in code.replace("outputs/p3b/s2", "")
    assert "p3_train_next.py" not in code
    assert '"--runtime"' in code and "p3b_train_check.py" in code


def test_the_runtime_record_on_a_cpu_host() -> None:
    preflight = _script("p2_preflight")
    recorded = preflight.check_runtime(require_cuda=False)
    assert recorded.ok and recorded.data["torch"] and "driver" in recorded.data
    assert {"torch_cuda", "cudnn", "gpus"} <= set(recorded.data)
    if not recorded.data["gpus"]:
        required = preflight.check_runtime(require_cuda=True)
        assert not required.ok and "CUDA runtime is required" in required.detail


def test_the_seed_rule() -> None:
    analyze = _script("p3b_analyze")
    assert analyze.across_seeds({"0": True, "1": True}) == "PASS"
    assert analyze.across_seeds({"0": True, "1": False}) == "seed-dependent"
    assert analyze.across_seeds({"0": False, "1": False}) == "not passed"
    assert analyze.across_seeds({"0": True, "1": None}) == "not answered"
    assert (analyze.MATCH_MARGIN, analyze.GAIN) == (-0.02, 0.02)
    assert "not evidence" in analyze.READING


def test_the_reproduction_is_exact_and_against_the_current_references() -> None:
    record = json.loads((ROOT / "reports" / "p3b_results" / "p3b_reproduce.json")
                        .read_text("utf-8"))["reproduction"]
    assert record["exact_match"] and record["checked"] == 20
    assert all(not c["differences"] for c in record["checks"])
    from dbp.data.manifest import file_sha256
    for path, sha in record["references"].items():
        assert file_sha256(ROOT / path) == sha, path
