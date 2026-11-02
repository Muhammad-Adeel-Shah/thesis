"""The final scoring refuses to plan unless section 5, Panel B and the 24 models are frozen."""

from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECT = json.loads((ROOT / "env" / "p3_final_expect.json").read_text(encoding="utf-8"))
RUNBOOK = ROOT / "env" / "kaggle_run_p3_final.py"


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_final_pf", ROOT / "scripts" /
                                                  "p3_final_preflight.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_final_pf"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _section5(tmp: Path) -> tuple[Path, dict, list[dict]]:
    m = _module()
    code = tmp / "code"
    _write(code / "reports/choice.json", json.dumps({"chosen": "s2b_deg_p35_seed0",
                                                     "chosen_p": 0.35}))
    _write(code / "reports/nstar.json", json.dumps({
        "n_star": {"step": 5000}, "rule": {"n_max": 6000},
        "nested": {str(f): {"step": 4500 if f == 6 else 5000} for f in range(1, 9)}}))
    _write(code / "reports/oi.csv", "image_id,split,sha256\na,train,x\nb,train,x\nc,held_out,x\n")
    sha = m.file_sha256
    s5 = {**EXPECT["section5"], "degradation_choice": "reports/choice.json",
          "degradation_choice_sha256": sha(code / "reports/choice.json"),
          "nstar": "reports/nstar.json", "nstar_sha256": sha(code / "reports/nstar.json"),
          "oi_manifest": "reports/oi.csv", "oi_manifest_sha256": sha(code / "reports/oi.csv"),
          "oi_train": 2, "oi_held_out": 1}
    models = [{"id": "lolo_f6_seed0", "kind": "snapshot", "protocol": "lolo_f6", "step": 4500},
              {"id": "s2d_p35_seed0", "kind": "snapshot", "protocol": "s2d_p35", "step": 5000},
              {"id": "full_bank_seed0", "kind": "lora_head", "protocol": "full_bank"}]
    return code, {"section5": s5}, models


def test_section5_passes_when_everything_is_committed(tmp_path: Path) -> None:
    code, expect, models = _section5(tmp_path)
    record, problems = _module().check_section5(code, expect, models)
    assert problems == [] and record["nested"]["6"] == 4500 and record["kshot_j"]["50"] == [1, True]


def test_section5_refuses_a_wrong_p_a_wrong_step_and_a_wrong_kshot_rule(tmp_path: Path) -> None:
    m = _module()
    code, expect, models = _section5(tmp_path)
    bad = copy.deepcopy(models)
    bad[0]["step"] = 5000                       # fold 6's N*(-f) is 4500
    bad[1]["protocol"] = "s2d_p70"              # not the chosen p
    _, problems = m.check_section5(code, expect, bad)
    assert any("lolo_f6_seed0 at step 5000" in p for p in problems)
    assert any("s2d_p35_seed0 is not the chosen p" in p for p in problems)
    wrong = copy.deepcopy(expect)
    wrong["section5"]["kshot"]["50"] = [2, True]
    _, problems = m.check_section5(code, wrong, models)
    assert any("k-shot" in p for p in problems)
    _write(code / "reports/choice.json", json.dumps({"chosen": "s2b_deg_p70_seed0",
                                                     "chosen_p": 0.7}))
    _, problems = m.check_section5(code, expect, models)
    assert any("not the committed file" in p for p in problems)


def _panel(tmp: Path) -> tuple[Path, dict]:
    m = _module()
    code = tmp / "code"
    rows = [{"image_id": f"i{k}", "relative_path": f"v/i{k}.png", "label": "real",
             "view": "v", "sha256": f"{k:064d}", "role": "test" if k else "calibration",
             "panel_b_slot": "s", "e3_number_in_p2": ""} for k in range(3)]
    for name, items in (("images", rows), ("luma", rows[:1])):
        path = code / f"reports/{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(items)
    from p2_materialise_panel import panel_digest
    spec = {"images": "reports/images.csv", "luma": "reports/luma.csv",
            "images_sha256": m.file_sha256(code / "reports/images.csv"),
            "luma_sha256": m.file_sha256(code / "reports/luma.csv"),
            "digest": panel_digest(rows), "luma_digest": panel_digest(rows[:1]),
            "test_digest": panel_digest(rows[1:]), "calibration_digest": panel_digest(rows[:1])}
    return code, {"panel_b": spec}


def test_panel_b_passes_only_with_the_frozen_manifests(tmp_path: Path) -> None:
    m = _module()
    code, expect = _panel(tmp_path)
    record, problems, rows, luma = m.check_panel(code, expect)
    assert problems == [] and len(rows) == 3 and len(luma) == 1
    expect["panel_b"]["digest"] = "0" * 64
    _, problems, _, _ = m.check_panel(code, expect)
    assert any("digest" in p for p in problems)
    (code / "reports/images.csv").write_text("changed\n", encoding="utf-8")
    _, problems, _, _ = m.check_panel(code, expect)
    assert any("not the committed file" in p for p in problems)


def _model(tmp: Path, report: dict) -> tuple[dict, dict]:
    root = tmp / "train"
    snapshot = root / "outputs/p3/s2/s2b_seed1/snapshots/step_05000"
    _write(snapshot / "lora/adapter_model.safetensors", "weights")
    _write(snapshot / "lora/adapter_config.json", "{}")
    _write(snapshot / "head.pt", "head")
    _write(root / "reports/p2_s2_s2b_seed1.json", json.dumps(report))
    spec = {"id": "s2b_seed1", "kind": "snapshot", "protocol": "s2b", "seed": 1,
            "source": "train", "step": 5000, "runs_config": "configs/p3_runs.yaml",
            "expect": {}}
    return spec, {"train": root}


FINISHED = {"status": "OK", "fixed_steps": 6000, "schedule_steps": 6000, "stop_at": 5000,
            "steps_run": 5000}


def test_a_model_is_hashed_and_checked_against_its_record(tmp_path: Path) -> None:
    m = _module()
    spec, sources = _model(tmp_path, FINISHED)
    record, problems = m.check_model(spec, sources, EXPECT["section5"])
    assert problems == [] and len(record["snapshot_sha256"]) == 64
    spec["expect"] = {"head_sha256": "0" * 64}
    _, problems = m.check_model(spec, sources, EXPECT["section5"])
    assert any("head_sha256" in p for p in problems)


def test_an_unfinished_or_missing_model_is_refused(tmp_path: Path) -> None:
    m = _module()
    spec, sources = _model(tmp_path, {**FINISHED, "steps_run": 4000})
    _, problems = m.check_model(spec, sources, EXPECT["section5"])
    assert any("finished run" in p for p in problems)
    _, problems = m.check_model({**spec, "source": "s2d"}, sources, EXPECT["section5"])
    assert any("not attached" in p for p in problems)


def test_complete_stages_are_carried_and_interrupted_ones_flagged(tmp_path: Path) -> None:
    m = _module()

    class Log:
        def info(self, *args: object) -> None:
            return None

    previous, here = tmp_path / "previous", tmp_path / "here"
    _write(previous / m.SUMMARIES / "a_clean.json", json.dumps({"limit": None}))
    _write(previous / m.FINAL / "a/clean/panel_b.csv", "image_id,score\n")
    _write(previous / m.FINAL / "b/clean/panel_b.csv", "image_id,score\n")  # no summary
    _write(previous / m.SUMMARIES / "c_clean.json", json.dumps({"limit": 2}))  # a smoke
    carried, incomplete = m.carry([previous], here, ["a", "b", "c"], "clean", Log())
    assert carried == ["a"] and incomplete == ["b"]
    assert m.is_complete(here, "a", "clean") and not m.is_complete(here, "c", "clean")


def test_the_expect_lists_the_24_models_with_their_committed_hashes() -> None:
    models = EXPECT["models"]
    assert len(models) == EXPECT["n_models"] == 24 == len({m["id"] for m in models})
    assert set(EXPECT["robust"]["models"]) <= {m["id"] for m in models}
    scores = ROOT / "reports" / "p3_heldout_scores"
    for model in models:
        want = model["expect"]
        if "snapshot_sha256" in want:
            sidecar = json.loads((scores / f"{model['id']}_step_{model['step']:05d}.json")
                                 .read_text(encoding="utf-8"))
            assert want["snapshot_sha256"] == sidecar["snapshot_sha256"], model["id"]
    tune = ROOT / "reports/kaggle_20261021/p3_tune"
    p2 = ROOT / "reports/kaggle_20261005"
    for model_id, folder in (("s2b_seed0", tune), ("s2b_deg_p35_seed0", tune),
                             ("full_bank_seed0", p2), ("p_a_sd1_seed0", p2)):
        summary = json.loads((folder / f"p2_s2_scores_{model_id}.json").read_text("utf-8"))
        model = next(m for m in models if m["id"] == model_id)
        assert model["expect"] == {"lora_sha256": summary["lora_sha256"],
                                   "head_sha256": summary["head_sha256"]}
    # Since 2026-10-27 every model is checked, none only recorded (user): the last four were
    # hashed from their step-5000 weight files (LoRA adapter and head), as S2-B s0 was.
    assert all(set(m["expect"]) in ({"snapshot_sha256"}, {"lora_sha256", "head_sha256"})
               for m in models), [m["id"] for m in models if not m["expect"]]
    lora = [m["expect"]["lora_sha256"] for m in models if "lora_sha256" in m["expect"]]
    assert len(lora) == len(set(lora)) == 8  # 4 P3 + 2 tuning/S2-B s0 + 2 P2, all distinct


def test_the_runbook_checks_before_it_scores_and_smokes_every_model() -> None:
    code = RUNBOOK.read_text(encoding="utf-8")
    preflight = code.index('py("p3_final_preflight.py"')
    smoke = code.index('"--worker-arg=--limit"')
    score = code.index('code = py("p2_run_shards.py"')
    assert code.index('py("p2_preflight.py"') < preflight < smoke < score
    assert 'assert PLAN["stage"] != "complete"' in code


def test_a_real_rescore_is_refused_but_a_repeated_smoke_is_not(tmp_path: Path) -> None:
    """Dry-run of 2026-10-22: a second smoke into the same scratch was refused."""
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("_final_score", ROOT / "scripts" /
                                                  "p3_final_score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    summary = tmp_path / "m_clean.json"
    assert not module.refuses_rescore(summary, 0)
    summary.write_text("{}", encoding="utf-8")
    assert module.refuses_rescore(summary, 0)
    assert not module.refuses_rescore(summary, 2)
