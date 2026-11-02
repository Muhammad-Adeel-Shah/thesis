"""The P2 development panel rebuilt on muhammadadeelshah (2026-10-19): byte-identical or nothing."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p3_rebuild_dev_panel.py"


def _module(path: Path, alias: str):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def test_the_committed_sets_reproduce_both_recorded_digests() -> None:
    m = _module(ROOT / "scripts" / "p2_materialise_panel.py", "_devpanel_m")
    rebuild = _module(SCRIPT, "_devpanel_rebuild")
    rows = []
    for path in sorted((ROOT / "reports" / "p2_panel_sets").glob("p2_panel_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            rows += list(csv.DictReader(handle))
    assert len(rows) == 12150
    assert m.panel_digest(rows) == rebuild.DIGEST_ALL
    expect = json.loads((ROOT / "env" / "p3_dev_panel_expect.json").read_text("utf-8"))
    assert expect["digest_all"] == rebuild.DIGEST_ALL
    assert expect["digest_scoreable"] == rebuild.DIGEST_SCOREABLE
    assert expect["inputs"]["p2_panel_images.csv"]["rows"] == 11775


def _tiny(tmp: Path, corrupt: bool = False, drop_listed: bool = False):
    """One E8-style view (relative paths under its root) and one T-ctrl view."""
    e8 = tmp / "sources" / "e8x"
    corpus = tmp / "corpus"
    sets = tmp / "sets"
    sets.mkdir(parents=True)
    rows = {"e8_evalgen": [], "t_ctrl_diagnostic": []}
    for n in range(3):
        rel = f"Gen/{n}.jpg"
        (e8 / rel).parent.mkdir(parents=True, exist_ok=True)
        (e8 / rel).write_bytes(f"e8-{n}".encode())
        rows["e8_evalgen"].append({"image_id": f"e8_{n}", "relative_path": rel, "label": "fake",
                                   "view": "e8_evalgen",
                                   "sha256": hashlib.sha256(f"e8-{n}".encode()).hexdigest()})
    rel = "reals/t_ctrl/t0.png"
    (corpus / rel).parent.mkdir(parents=True, exist_ok=True)
    (corpus / rel).write_bytes(b"tctrl")
    rows["t_ctrl_diagnostic"].append({"image_id": "t0", "relative_path": rel, "label": "real",
                                      "view": "t_ctrl_diagnostic",
                                      "sha256": hashlib.sha256(b"tctrl").hexdigest()})
    if corrupt:
        (e8 / "Gen/1.jpg").write_bytes(b"changed")
    for view, view_rows in rows.items():
        with (sets / f"p2_panel_{view}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(view_rows[0]))
            writer.writeheader()
            writer.writerows(view_rows)
    listed = [{"image_id": r["image_id"], "relative_path": f"{r['view']}/{r['image_id']}"
               f"{Path(r['relative_path']).suffix}", "label": r["label"], "view": r["view"],
               "root": "panel"} for v in rows.values() for r in v][1:]
    if drop_listed:
        listed.append({"image_id": "ghost", "relative_path": "e8_evalgen/ghost.jpg",
                       "label": "fake", "view": "e8_evalgen", "root": "panel"})
    with (tmp / "list.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(listed[0]))
        writer.writeheader()
        writer.writerows(listed)
    (tmp / "roots.json").write_text(json.dumps({"e8": str(e8)}), encoding="utf-8")
    m = _module(ROOT / "scripts" / "p2_materialise_panel.py", "_devpanel_tiny")
    everything = [r for v in rows.values() for r in v]
    ids = {x["image_id"] for x in listed}
    scoreable = [r for r in everything if r["image_id"] in ids]
    return m.panel_digest(everything), m.panel_digest(scoreable)


def _run(tmp: Path, digest_all: str, digest_scoreable: str) -> tuple[int, dict]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--p2-panel-sets", str(tmp / "sets"),
         "--roots", str(tmp / "roots.json"), "--corpus-root", str(tmp / "corpus"),
         "--panel-root", str(tmp / "panel"), "--scoreable-list", str(tmp / "list.csv"),
         "--views", "e8_evalgen,t_ctrl_diagnostic", "--expect-digest-all", digest_all,
         "--expect-digest-scoreable", digest_scoreable, "--out", str(tmp / "record.json")],
        capture_output=True, text=True, check=False)
    return done.returncode, json.loads((tmp / "record.json").read_text(encoding="utf-8"))


def test_a_byte_identical_rebuild_passes(tmp_path: Path) -> None:
    digest_all, digest_scoreable = _tiny(tmp_path)
    code, record = _run(tmp_path, digest_all, digest_scoreable)
    assert code == 0 and record["ok"] and record["rows_all"] == 4 and record["rows_scoreable"] == 3
    assert (tmp_path / "panel" / "t_ctrl_diagnostic" / "t0.png").read_bytes() == b"tctrl"


def test_a_wrong_digest_a_changed_file_or_a_missing_row_is_refused(tmp_path: Path) -> None:
    digest_all, digest_scoreable = _tiny(tmp_path / "a")
    code, record = _run(tmp_path / "a", digest_all, "0" * 64)
    assert code == 1 and not record["ok"] and "scoreable digest" in record["problems"][0]

    digest_all, digest_scoreable = _tiny(tmp_path / "b", corrupt=True)
    code, record = _run(tmp_path / "b", digest_all, digest_scoreable)
    assert code == 1 and "sha256 mismatch" in record["problems"][0]

    digest_all, digest_scoreable = _tiny(tmp_path / "c", drop_listed=True)
    code, record = _run(tmp_path / "c", digest_all, digest_scoreable)
    assert code == 1 and any("listed id(s) not in the panel sets" in p for p in record["problems"])
