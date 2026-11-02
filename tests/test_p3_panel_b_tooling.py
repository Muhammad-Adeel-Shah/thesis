"""P3 Panel B tooling: fetch, view rebuild, bootstrap opt-ins (user rules 2026-10-11)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


views = _load("p3_views_test", "p3_build_eval_views.py")
fetch = _load("p3_fetch_test", "p3_fetch_eval_sets.py")
SUMMARIES = ROOT / "reports" / "kaggle_20260928"


def _summary(set_id: str) -> dict:
    return json.loads((SUMMARIES / f"p1_eval_{set_id}_summary.json").read_text(encoding="utf-8"))


def test_view_arguments_come_from_the_committed_p1_summaries() -> None:
    e5 = views.p1_arguments(_summary("e5_chameleon"), "/root")
    assert e5[:4] == ["--set-id", "e5_chameleon", "--root", "/root"]
    assert "--jpeg-match" in e5
    assert e5[e5.index("--pipeline-match-width") + 1] == "500"
    assert e5[e5.index("--pipeline-match-quality") + 1] == "75"
    e8 = views.p1_arguments(_summary("e8_evalgen"), "/root")
    assert e8[e8.index("--subsample") + 1] == "1000"
    assert e8[e8.index("--stratify-by") + 1] == "generator"
    e7 = views.p1_arguments(_summary("e7_dda_coco"), "/root")
    assert e7[e7.index("--field"):e7.index("--field") + 2] == ["--field", "source_vae=0"]
    assert "--jpeg-match" not in e7


def test_every_view_maps_to_a_committed_summary_and_a_panel_b_view() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from dbp.eval.panel_b import PANEL_B_VIEWS

    for _name, (set_id, view, manifest) in views.VIEWS.items():
        assert (SUMMARIES / f"p1_eval_{set_id}_summary.json").is_file()
        assert PANEL_B_VIEWS[view]["manifest"] == manifest
        assert (ROOT / "reports" / "p2_panel_sets" / f"p2_panel_{view}.csv").is_file()


def test_a_missing_or_changed_p2_row_stops_the_rebuild() -> None:
    rebuilt = [{"image_id": "a", "sha256": "1", "label": "real"},
               {"image_id": "b", "sha256": "2", "label": "fake"}]
    assert views.verify("v", rebuilt, [{"image_id": "a", "sha256": "1"}],
                        {"real": 1, "fake": 1})["p2_rows_reproduced"] == 1
    with pytest.raises(views.ViewError, match="1 P2 row\\(s\\) missing"):
        views.verify("v", rebuilt, [{"image_id": "z", "sha256": "9"}], None)
    with pytest.raises(views.ViewError, match="different sha256"):
        views.verify("v", rebuilt, [{"image_id": "a", "sha256": "X"}], None)
    with pytest.raises(views.ViewError, match="label counts"):
        views.verify("v", rebuilt, [], {"real": 5, "fake": 1})


def test_the_fetch_sources_are_p1s_pins() -> None:
    assert fetch.HF_SOURCES["e7"]["revision"] == "8c9330a3b374bcac46a8045a0e3c09ebcf7868fb"
    assert fetch.HF_SOURCES["e8"]["revision"] == "86c719a9bfc72b7e5d91bfbb04432774feb4e3b7"
    assert fetch.HF_SOURCES["e3"]["revision"] == "1498eead24292a9a2d134476b2c559193b68b9de"
    assert fetch.GDOWN_SOURCES["e4"]["file_id"] == "1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn"
    roots = {"e3": "e3", "e4": "wildrf/WildRF", "e7": "e7x/DDA-COCO", "e8": "e8x"}
    for name, root in roots.items():
        source = {**fetch.HF_SOURCES, **fetch.GDOWN_SOURCES}[name]
        assert source["root"] == root
        recorded = _summary(views.VIEWS[name][0])["root"]
        assert recorded.endswith(root), (name, recorded)       # P1's recorded roots


def test_local_mode_writes_roots_without_any_network(tmp_path: Path) -> None:
    (tmp_path / "e5").mkdir()
    out = tmp_path / "roots.json"
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "p3_fetch_eval_sets.py"),
                             "--tmp", str(tmp_path), "--sets", "e5", "--local",
                             f"e5={tmp_path / 'e5'}", "--roots-out", str(out)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(out.read_text())["e5"] == str(tmp_path / "e5")


def test_an_archive_member_that_escapes_is_refused(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../escape.txt", "x")
    with pytest.raises(fetch.FetchError, match="escapes"):
        fetch.extract(archive, tmp_path / "out", type("L", (), {"info": print})())


def test_the_bootstrap_opt_ins_follow_the_user_rules() -> None:
    text = (ROOT / "env" / "kaggle_bootstrap.sh").read_text(encoding="utf-8")
    assert 'GDOWN_PIN="gdown==5.2.0"' in text                     # P1's pin
    assert "pip install --no-deps" in text and '"${GDOWN_PIN}"' in text
    assert 'WITH_GDOWN="${DBP_WITH_GDOWN:-0}"' in text            # off unless asked
    assert "import gdown" in text
    assert "CORE_BEFORE" in text and "CORE_AFTER" in text and "exit 1" in text
    assert 'CPU_SESSION="${DBP_CPU_SESSION:-0}"' in text
    assert "gdown" not in (ROOT / "requirements" / "base.txt").read_text(encoding="utf-8")
    p1 = (ROOT / "env" / "kaggle_run_p1_eval.md").read_text(encoding="utf-8")
    assert '"gdown==5.2.0"' in p1                                  # the version P1 used


def test_e7_reals_come_from_coco_val2017(tmp_path: Path) -> None:
    """Regression, Kaggle 2026-10-11: DDA-COCO ships no reals; P1 added COCO val2017."""
    root, coco = tmp_path / "DDA-COCO", tmp_path / "val2017"
    (root / "FLUX.1" / "val2017").mkdir(parents=True)
    coco.mkdir()
    for n in range(3):
        (coco / f"{n:012d}.jpg").write_bytes(bytes([n]) * 10)
    log = type("L", (), {"info": staticmethod(lambda *a: None)})()
    with pytest.raises(fetch.FetchError, match="holds 3 jpg"):
        fetch.add_e7_reals(root, coco, 5000, log)
    assert fetch.add_e7_reals(root, coco, 3, log) == 3
    assert (root / "real" / "000000000002.jpg").read_bytes() == bytes([2]) * 10
    with pytest.raises(fetch.FetchError, match="already holds files"):
        fetch.add_e7_reals(root, coco, 3, log)
    assert fetch.COCO_VAL_COUNT == 5000                    # P1's E7 real count


def test_e7_without_coco_val_is_refused(tmp_path: Path) -> None:
    (tmp_path / "e7").mkdir()
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "p3_fetch_eval_sets.py"),
                             "--tmp", str(tmp_path), "--sets", "e7", "--local",
                             f"e7={tmp_path / 'e7'}", "--roots-out", str(tmp_path / "r.json")],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert "--coco-val is required" in result.stdout + result.stderr


def test_the_runbook_attaches_coco_val2017_for_e7() -> None:
    text = (ROOT / "env" / "kaggle_run_p3_panel_b.py").read_text(encoding="utf-8")
    assert '"coco_val": find("datasets/*/coco-2017-dataset/coco2017/val2017")' in text
    assert '"--coco-val", INPUTS["coco_val"]' in text
    assert "coco_val={INPUTS['coco_val']}" in text           # pre-flight requires it
