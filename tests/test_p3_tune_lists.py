"""Tuning scores only the rows the committed rule reads (2026-10-21); its numbers do not move."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = ROOT / "env" / "kaggle_run_p3_tune.py"
EXPECT = ROOT / "env" / "p3_tune_expect.json"
COLUMNS = ["image_id", "relative_path", "label", "view", "root"]


def _module(name: str):
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location(f"_tl_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[f"_tl_{name}"] = module
    spec.loader.exec_module(module)
    return module


def _csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _panel(tmp: Path) -> tuple[Path, Path, Path]:
    """Every view of the dev panel, a few rows each; E3 fakes of both unseen generators."""
    rows, e3 = [], []
    for view, n in (("e3_matched", 4), ("e4_wildrf_matched", 4), ("e5_chameleon_pipeline", 4),
                    ("e7_dda_coco", 3), ("e8_evalgen", 2), ("t_ctrl_diagnostic", 2)):
        for k in range(n):
            label = "fake" if k % 2 == 0 or view == "e8_evalgen" else "real"
            image_id = f"{view}_{k}"
            rows.append({"image_id": image_id, "relative_path": f"{view}/{image_id}.png",
                         "label": label, "view": view, "root": "panel"})
            if view == "e3_matched":
                e3.append({"image_id": image_id, "label": label, "source_label":
                           ("GPT-Image2-fake" if k == 0 else "Nano-Banana2-fake")
                           if label == "fake" else "real"})
    panel = tmp / "panel.csv"
    _csv(panel, COLUMNS, rows)
    manifest = tmp / "e3.csv"
    _csv(manifest, ["image_id", "label", "source_label"], e3)
    val = tmp / "s1.csv"
    _csv(val, COLUMNS, [{"image_id": f"v{k}", "relative_path": f"reals/v{k}.png",
                         "label": "real" if k < 5 else "fake", "view": "val" if k < 8 else
                         "train", "root": "corpus"} for k in range(10)])
    return panel, manifest, val


def test_the_rule_gives_identical_numbers_on_the_restricted_rows(tmp_path: Path) -> None:
    lists = _module("p3_tune_lists")
    rule = _module("p3_tune_degradation")
    panel, manifest, val = _panel(tmp_path)
    columns, kept = lists.rule_rows(panel, manifest)
    assert columns == COLUMNS
    assert {r["view"] for r in kept} == {"e3_matched", "e4_wildrf_matched",
                                         "e5_chameleon_pipeline"}
    assert len(kept) == 12
    restricted = tmp_path / "restricted.csv"
    lists.write_rows(restricted, columns, kept)
    full_rows = rule.panel_rows(panel, manifest)
    kept_rows = rule.panel_rows(restricted, manifest)
    _, reals = lists.val_reals(val)
    assert [r["image_id"] for r in reals] == [f"v{k}" for k in range(5)]
    for seed in range(5):
        rng = np.random.default_rng(seed)
        scores = {r["image_id"]: float(rng.normal() + (r["label"] == "fake")) for r in full_rows}
        val_scores = [float(x) for x in rng.normal(size=5)]
        full = rule.evaluate(full_rows, scores, val_scores)
        ids = {r["image_id"] for r in kept_rows}
        part = rule.evaluate(kept_rows, {k: v for k, v in scores.items() if k in ids},
                             val_scores)
        assert json.dumps(full, sort_keys=True) == json.dumps(part, sort_keys=True)


def test_a_list_off_the_committed_digest_is_refused(tmp_path: Path, monkeypatch) -> None:
    lists = _module("p3_tune_lists")
    panel, manifest, val = _panel(tmp_path)
    expect = tmp_path / "expect.json"
    expect.write_text(json.dumps({"derived_lists": {
        lists.PANEL_ROWS: {"rows": 12, "sha256": "0" * 64},
        lists.VAL_REALS: {"rows": 5, "sha256": "0" * 64}}}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["p3_tune_lists.py", "--panel-list", str(panel),
                                      "--val-list", str(val), "--e3-manifest", str(manifest),
                                      "--out-dir", str(tmp_path / "out"), "--expect",
                                      str(expect)])
    assert lists.main() == 1


def test_the_committed_expect_scores_only_the_rule_rows() -> None:
    expect = json.loads(EXPECT.read_text(encoding="utf-8"))
    assert expect["derived_lists"]["p3_tune_panel_rows.csv"]["rows"] == 6717
    assert expect["derived_lists"]["p3_tune_val_reals.csv"]["rows"] == 491
    assert expect["lists"] == {"panel": ["p3_tune_panel_rows.csv", None],
                               "val": ["p3_tune_val_reals.csv", None]}
    assert expect["resolve"] == ["p3_tune_panel_rows.csv", "p3_tune_val_reals.csv"]
    # The full lists are still verified by hash: the derived ones come from them.
    assert expect["inputs"]["p2_panel_images.csv"]["rows"] == 11775


def test_the_runbook_derives_first_and_gives_the_rule_the_restricted_list() -> None:
    code = RUNBOOK.read_text(encoding="utf-8")
    derive = code.index('py("p3_tune_lists.py"')
    assert derive < code.index('py("p2_preflight.py"') < code.index('code = py("p2_run_queue.py"')
    assert 'checks += ["--resolve", TUNE_LISTS / name]' in code
    assert 'LISTS += ["--list", f"{name}={TUNE_LISTS / csv_name}"]' in code
    rule = code[code.index('py("p3_tune_degradation.py"'):].split("\npy(")[0]
    assert 'TUNE_LISTS / "p3_tune_panel_rows.csv"' in rule
    assert 'S2IN / "p2_s1_images.csv"' in rule  # the rule's val list stays the full S1 list
