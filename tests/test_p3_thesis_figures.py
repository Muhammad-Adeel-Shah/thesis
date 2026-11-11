"""The thesis tables restate the committed verdicts exactly and label every row."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_thesis_figs", ROOT / "scripts" /
                                                  "p3_thesis_figures.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_thesis_figs"] = module
    spec.loader.exec_module(module)
    return module


M = _module()
DATA = {k: json.loads(p.read_text(encoding="utf-8")) for k, p in M.SOURCES.items()}


def test_every_committed_verdict_appears_unchanged():
    rows = M.verdict_rows(DATA["h1a"], DATA["h1c"], DATA["full"])
    c = DATA["full"]["confirmatory"]
    final = {r["hypothesis"]: r["verdict"] for r in rows if r["estimate"] == "-"}
    assert final["H3"] == c["H3"]["verdict"] and final["H5"] == c["H5"]["verdict"]
    assert final["H3-k (Q)"] == c["H3k"]["verdict"]["Q"]
    assert final["H3-k (S)"] == c["H3k"]["verdict"]["S"]
    h2 = [r["verdict"] for r in rows if r["hypothesis"] == "H2" and r["estimate"] == "-"]
    assert h2 == [c["H2"]["four_unseen_slices"]["verdict"], c["H2"]["without_e3"]["verdict"]]
    single = {r["hypothesis"]: r["verdict"] for r in rows}
    assert single["H1-a"] == DATA["h1a"]["verdict"] and single["H1-c"] == DATA["h1c"]["verdict"]
    assert single["H4"] == c["H4"]["verdict"]
    assert all(r["status"] for r in rows)


def test_only_pre_registered_rows_are_marked_decisions():
    rows = M.h5_rows(DATA["full"])
    decided = [(r["model"], r["condition"]) for r in rows if r["status"] == M.DECISION]
    assert decided == [("S2-D s0", "jpeg50"), ("S2-D s0", "resize050")]
    h3k = [r for r in M.h3k_rows(DATA["full"]) if r["status"] == M.DECISION]
    assert {(r["pool"], r["k"]) for r in h3k} == {("E4", "50"), ("E5", "50")}
    assert all(r["status"] == M.DESCRIPTIVE for r in M.slice_rows(DATA["full"],
                                                                    ["s2d_p35_seed0"]))


def test_h3k_ece_table_restates_the_stored_values():
    rows = M.h3k_ece_rows(DATA["full"])
    views = DATA["full"]["confirmatory"]["H3k"]["seeds"]["s2d_p35_seed1"]["views"]
    row = next(r for r in rows if r["seed"] == "S2-D s1" and r["pool"] == "E5" and
               r["method"] == "S" and r["k"] == "50")
    assert row["ece"] == f"{views['e5_chameleon_pipeline']['S']['50']['mean_ece']:.4f}"
    assert {r["pool"] for r in rows} == {"E4", "E5"} and len(rows) == 2 * 2 * 7
    assert all(r["status"] == M.DESCRIPTIVE for r in rows)


def test_truefake_tables_restate_the_stored_values():
    tf = DATA["truefake"]
    rows = M.truefake_rows(tf)
    assert len(rows) == 4 * 2 * 10 and all(r["status"] == M.DESCRIPTIVE for r in rows)
    row = next(r for r in rows if r["model"] == "S2-D s0" and r["real_pool"] == "FFHQ" and
               r["fakes"] == "FLUX.1")
    r = tf["results"]["s2d_p35_seed0"]
    assert row["pre_share_auc"] == M.ci(r["auc"]["PreSocial"]["FFHQ"]["FLUX.1"])
    assert row["facebook_delta"] == M.ci(r["delta"]["Facebook"]["FFHQ"]["FLUX.1"])
    fpr = M.truefake_fpr_rows(tf)
    v = tf["forlab_fpr_at_k0_union_threshold"]["by_model"]["s2b_seed0"]
    first = next(f for f in fpr if f["model"] == "S2-B s0")
    assert first["twitter_fpr"] == M.pct(v["Twitter"]) and len(fpr) == 4
    assert all(f["status"] == M.DESCRIPTIVE for f in fpr)


def test_pool_tpr_weights_slices_by_their_fakes():
    desc = {"slices": {"e3_matched/a": {"fakes": 100, "tpr_k0_union": 0.2},
                       "e3_matched/b": {"fakes": 300, "tpr_k0_union": 0.6},
                       "e4_wildrf_matched/fake": {"fakes": 5, "tpr_k0_union": 1.0}}}
    assert abs(M.pool_tpr(desc, "e3_matched") - 0.5) < 1e-12


def test_markdown_cells_escape_pipes(tmp_path):
    M.write_table(tmp_path, "t", "T", [{"a": "|x| <= 1", "b": "y"}], "note")
    line = (tmp_path / "t.md").read_text(encoding="utf-8").splitlines()[-1]
    assert line == "| \\|x\\| <= 1 | y |"
