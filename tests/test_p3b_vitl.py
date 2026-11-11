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
