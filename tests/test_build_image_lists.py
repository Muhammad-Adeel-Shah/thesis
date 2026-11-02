"""Tests for the P2 image lists (B1).

Three properties matter here, and all three are about not wasting nine GPU-hours:
a path that does not resolve is caught before Session B, a duplicate id is refused
because it would be reconstructed and counted twice, and the S3 subset is balanced
and reproducible rather than whatever the manifest order happened to give.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
THESIS_DBP = Path(__file__).resolve().parents[1]


def _load():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "p2_build_image_lists", SCRIPTS / "p2_build_image_lists.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


lists = _load()


# ------------------------------------------------------- resolve_relative ---


def _corpus(tmp_path: Path) -> Path:
    root = tmp_path / "p1"
    (root / "reals" / "t_real").mkdir(parents=True)
    (root / "reals" / "t_real" / "aaa.png").write_bytes(b"r")
    (root / "pairs" / "sd15").mkdir(parents=True)
    (root / "pairs" / "sd15" / "aaa.png").write_bytes(b"p")
    (root / "gen_matched" / "gen_matched").mkdir(parents=True)
    (root / "gen_matched" / "gen_matched" / "coco_1.jpg").write_bytes(b"g")
    return root


def test_a_relative_path_resolves_unchanged(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    assert lists.resolve_relative(root, "reals/t_real/aaa.png", "reals") == "reals/t_real/aaa.png"


def test_an_absolute_path_from_another_session_is_recovered(tmp_path: Path) -> None:
    """p1_gen_matched_manifest.csv records absolute paths with a doubled directory."""
    root = _corpus(tmp_path)
    recorded = (
        "/kaggle/working/Implementation/thesis_dbp/outputs/p1"
        "/gen_matched/gen_matched/coco_1.jpg"
    )
    assert (
        lists.resolve_relative(root, recorded, "gen_matched")
        == "gen_matched/gen_matched/coco_1.jpg"
    )


def test_a_windows_style_path_is_recovered(tmp_path: Path) -> None:
    root = _corpus(tmp_path)
    assert lists.resolve_relative(root, r"reals\t_real\aaa.png", "reals") == "reals/t_real/aaa.png"


def test_a_genuinely_missing_file_returns_none(tmp_path: Path) -> None:
    """None, not a guess: a wrong file that happens to exist would be worse."""
    root = _corpus(tmp_path)
    assert lists.resolve_relative(root, "reals/t_real/nope.png", "reals") is None
    assert lists.resolve_relative(
        root, "/abs/gen_matched/gen_matched/nope.jpg", "gen_matched"
    ) is None


def test_resolution_does_not_cross_subdirectories(tmp_path: Path) -> None:
    """A real and its reconstruction share a basename; the subdir must disambiguate."""
    root = _corpus(tmp_path)
    assert lists.resolve_relative(root, "aaa.png", "reals") == "reals/t_real/aaa.png"
    assert lists.resolve_relative(root, "aaa.png", "pairs") == "pairs/sd15/aaa.png"


# ---------------------------------------------------------- balanced_take ---


def test_balanced_take_spreads_evenly_over_groups() -> None:
    groups = {f"m{i}": [f"m{i}_{j}" for j in range(500)] for i in range(10)}
    picked = lists.balanced_take(groups, 1000, 2026, "s")
    assert len(picked) == 1000
    per_group = {g: sum(1 for p in picked if p.startswith(f"{g}_")) for g in groups}
    assert set(per_group.values()) == {100}


def test_balanced_take_absorbs_a_small_group_into_the_others() -> None:
    groups = {"big": [f"b{i}" for i in range(500)], "tiny": ["t0", "t1"]}
    picked = lists.balanced_take(groups, 100, 2026, "s")
    assert len(picked) == 100
    assert sum(1 for p in picked if p.startswith("t")) == 2


def test_balanced_take_returns_everything_when_asked_for_too_much() -> None:
    picked = lists.balanced_take({"a": ["x", "y"], "b": ["z"]}, 500, 2026, "s")
    assert sorted(picked) == ["x", "y", "z"]


def test_balanced_take_is_reproducible_and_order_independent() -> None:
    forward = {"a": [f"a{i}" for i in range(50)], "b": [f"b{i}" for i in range(50)]}
    backward = {"b": list(reversed(forward["b"])), "a": list(reversed(forward["a"]))}
    assert lists.balanced_take(forward, 40, 2026, "s") == lists.balanced_take(
        backward, 40, 2026, "s"
    )


def test_a_different_seed_picks_a_different_subset() -> None:
    groups = {"a": [f"a{i}" for i in range(200)]}
    assert lists.balanced_take(groups, 50, 2026, "s") != lists.balanced_take(groups, 50, 2027, "s")


def test_a_different_salt_picks_a_different_subset() -> None:
    """Each protocol draws its own fakes; they must not be the same 1,000 rows."""
    groups = {"a": [f"a{i}" for i in range(200)]}
    assert lists.balanced_take(groups, 50, 2026, "p_a") != lists.balanced_take(
        groups, 50, 2026, "full"
    )


def test_balanced_take_handles_empty_groups() -> None:
    assert lists.balanced_take({}, 10, 2026, "s") == []
    assert lists.balanced_take({"a": []}, 10, 2026, "s") == []


# -------------------------------------------------------------- write_list ---


def test_write_list_refuses_a_duplicate_id(tmp_path: Path) -> None:
    """A duplicate is reconstructed twice by S3 and weighted twice in every metric."""
    rows = [
        {"image_id": "a", "relative_path": "x", "label": "real", "view": "train"},
        {"image_id": "a", "relative_path": "y", "label": "fake", "view": "train"},
    ]

    class _Log:
        def info(self, *args: object) -> None:
            pass

    with pytest.raises(lists.ImageListError, match="duplicate image_id"):
        lists.write_list(rows, tmp_path / "out.csv", _Log())


def test_write_list_reports_counts_by_view_and_label(tmp_path: Path) -> None:
    rows = [
        {"image_id": "a", "relative_path": "x", "label": "real", "view": "train"},
        {"image_id": "b", "relative_path": "y", "label": "fake", "view": "val"},
    ]

    class _Log:
        def info(self, *args: object) -> None:
            pass

    summary = lists.write_list(rows, tmp_path / "out.csv", _Log())
    assert summary["rows"] == 2
    assert summary["by_view"] == {"train": 1, "val": 1}
    assert summary["by_label"] == {"fake": 1, "real": 1}
    assert len(summary["sha256"]) == 64


# ------------------------------------------------------------ end to end ---


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def fixture(tmp_path: Path) -> dict[str, Path]:
    """A miniature corpus with real files, manifests, a split and one panel."""
    root = tmp_path / "p1"
    reports = tmp_path / "reports"
    members = ["sd15", "sd21", "sd_ft_ema", "sdxl", "dcae", "movq_k22"]

    real_rows, pair_rows, gm_rows, split_rows = [], [], [], []
    for i in range(40):
        rid = f"r{i:03d}"
        (root / "reals" / "t_real").mkdir(parents=True, exist_ok=True)
        (root / "reals" / "t_real" / f"{rid}.png").write_bytes(b"r")
        excluded = "greyscale" if i < 3 else ""
        real_rows.append({"image_id": rid, "relative_path": f"reals/t_real/{rid}.png",
                          "excluded_reason": excluded, "sha256": f"{i:064d}"})
        if not excluded:
            split_rows.append({"row_id": rid, "split": "train" if i < 32 else "val",
                               "kind": "real"})
        for member in members:
            pid = f"{rid}_{member}"
            (root / "pairs" / member).mkdir(parents=True, exist_ok=True)
            (root / "pairs" / member / f"{rid}.png").write_bytes(b"p")
            pair_rows.append({"pair_id": pid, "image_id": rid, "member": member,
                              "set_id": "t_real",
                              "relative_path": f"pairs/{member}/{rid}.png",
                              "sha256": f"{abs(hash(pid)) % 10**60:064d}"})
            if not excluded:
                split_rows.append({"row_id": pid, "split": "train" if i < 32 else "val",
                                   "kind": "fake"})

    for i in range(20):
        gid = f"coco_{i:012d}"
        (root / "gen_matched" / "gen_matched").mkdir(parents=True, exist_ok=True)
        (root / "gen_matched" / "gen_matched" / f"{gid}.jpg").write_bytes(b"g")
        gm_rows.append({
            "gen_id": gid,
            "jpg_path": f"/kaggle/working/old/outputs/p1/gen_matched/gen_matched/{gid}.jpg",
            "jpg_sha256": f"{i:064d}",
        })
        split_rows.append({"row_id": gid, "split": "train" if i < 16 else "val",
                           "kind": "fake"})

    # t_ctrl, two crops per source, plus their reconstructions: the diagnostic panel
    # is 200 lossless reals and 200 of their reconstructions, so both halves must be
    # reachable from the manifests.
    ctrl_rows = []
    for i in range(10):
        for crop in (0, 1):
            cid = f"c{i:03d}_c{crop}"
            (root / "reals" / "t_ctrl").mkdir(parents=True, exist_ok=True)
            (root / "reals" / "t_ctrl" / f"{cid}.png").write_bytes(b"c")
            ctrl_rows.append({"image_id": cid, "relative_path": f"reals/t_ctrl/{cid}.png",
                              "excluded_reason": "", "sha256": f"{abs(hash(cid)) % 10**60:064d}"})
            split_rows.append({"row_id": cid, "split": "train", "kind": "real"})
            for member in members:
                pid = f"{cid}_{member}"
                (root / "pairs" / member / f"{cid}.png").write_bytes(b"p")
                pair_rows.append({"pair_id": pid, "image_id": cid, "member": member,
                                  "set_id": "t_ctrl",
                                  "relative_path": f"pairs/{member}/{cid}.png",
                                  "sha256": f"{abs(hash(pid)) % 10**60:064d}"})
                split_rows.append({"row_id": pid, "split": "train", "kind": "fake"})

    _write(reports / "p1_reals_manifest_t_real.csv",
           ["image_id", "relative_path", "excluded_reason", "sha256"], real_rows)
    _write(reports / "p1_reals_manifest_t_ctrl.csv",
           ["image_id", "relative_path", "excluded_reason", "sha256"], ctrl_rows)
    _write(reports / "p1_pairs_manifest.csv",
           ["pair_id", "image_id", "member", "set_id", "relative_path", "sha256"], pair_rows)
    _write(reports / "p1_gen_matched_manifest.csv", ["gen_id", "jpg_path", "jpg_sha256"], gm_rows)
    _write(reports / "p2_splits.csv", ["row_id", "split", "kind"], split_rows)
    _write(reports / "p2_panel_e8_evalgen.csv",
           ["image_id", "relative_path", "label", "panel_set_id"],
           [{"image_id": f"e8_{i}", "relative_path": f"e8/{i}.png",
             "label": "fake", "panel_set_id": "e8_evalgen"} for i in range(12)])
    return {"root": root, "reports": reports}


def _run(fixture: dict[str, Path], **extra: str) -> subprocess.CompletedProcess[str]:
    argv = [
        sys.executable, str(SCRIPTS / "p2_build_image_lists.py"),
        "--root", str(fixture["root"]),
        "--manifests", str(fixture["reports"]),
        "--splits", str(fixture["reports"] / "p2_splits.csv"),
        "--pairs-manifest", str(fixture["reports"] / "p1_pairs_manifest.csv"),
        "--out-dir", str(fixture["reports"]),
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return subprocess.run(argv, capture_output=True, text=True, cwd=THESIS_DBP)


def test_end_to_end_writes_three_lists(fixture: dict[str, Path]) -> None:
    result = _run(fixture)
    assert result.returncode == 0, result.stderr
    for name in ("p2_s1_images.csv", "p2_panel_images.csv", "p2_probe_images.csv"):
        assert (fixture["reports"] / name).is_file(), name
    summary = json.loads((fixture["reports"] / "p2_image_lists_summary.json").read_text())
    assert {entry["list"] for entry in summary["lists"]} == {
        "p2_s1_images.csv", "p2_panel_images.csv", "p2_probe_images.csv",
    }


def test_flagged_reals_never_reach_any_list(fixture: dict[str, Path]) -> None:
    """D-039: the three greyscale reals are excluded, and so are their pair rows."""
    assert _run(fixture).returncode == 0
    for name in ("p2_s1_images.csv", "p2_probe_images.csv"):
        ids = {r["image_id"] for r in csv.DictReader(
            (fixture["reports"] / name).open(encoding="utf-8"))}
        assert not {"r000", "r001", "r002"} & ids, name
        assert not {i for i in ids if i.startswith(("r000_", "r001_", "r002_"))}, name


def test_a_flagged_reals_reconstructions_are_excluded_even_when_they_have_a_split(
    fixture: dict[str, Path],
) -> None:
    """The real defect this caught: the split carries them, the training lists must not.

    `p2_splits.csv` assigns a side to every pair row, flagged or not -- correctly, since
    the split is about content. The exclusion is a separate filter, and a consumer that
    honours it for the reals but not for their reconstructions puts the D-039 shortcut
    straight back: greyscale in the fake class and none in the real class.
    """
    # Give the flagged reals' pair rows a split, exactly as the real pipeline does.
    splits = fixture["reports"] / "p2_splits.csv"
    rows = list(csv.DictReader(splits.open(encoding="utf-8")))
    members = ["sd15", "sd21", "sd_ft_ema", "sdxl", "dcae", "movq_k22"]
    for rid in ("r000", "r001", "r002"):
        rows.append({"row_id": rid, "split": "train", "kind": "real"})
        rows.extend({"row_id": f"{rid}_{m}", "split": "train", "kind": "fake"}
                    for m in members)
    _write(splits, ["row_id", "split", "kind"], rows)

    assert _run(fixture).returncode == 0
    summary = json.loads((fixture["reports"] / "p2_image_lists_summary.json").read_text())
    assert summary["d039_excluded"]["reals"] == 3
    assert summary["d039_excluded"]["pair_rows"] == 3 * len(members)

    for name in ("p2_s1_images.csv", "p2_probe_images.csv"):
        ids = {r["image_id"] for r in csv.DictReader(
            (fixture["reports"] / name).open(encoding="utf-8"))}
        leaked = {i for i in ids if i.startswith(("r000_", "r001_", "r002_"))}
        assert not leaked, f"{name}: {sorted(leaked)[:5]}"


def test_gen_matched_absolute_paths_are_stored_relative(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    rows = {r["image_id"]: r["relative_path"] for r in csv.DictReader(
        (fixture["reports"] / "p2_s1_images.csv").open(encoding="utf-8"))}
    assert rows["coco_000000000000"] == "gen_matched/gen_matched/coco_000000000000.jpg"


def test_every_listed_path_exists_under_the_root(fixture: dict[str, Path]) -> None:
    """The whole point: a missing file costs a CPU cell here, not nine GPU-hours."""
    assert _run(fixture).returncode == 0
    root = fixture["root"]
    for row in csv.DictReader((fixture["reports"] / "p2_s1_images.csv").open(encoding="utf-8")):
        assert (root / row["relative_path"]).is_file(), row


def test_an_unresolvable_row_fails_the_run(fixture: dict[str, Path]) -> None:
    (fixture["root"] / "reals" / "t_real" / "r005.png").unlink()
    result = _run(fixture)
    assert result.returncode == 1
    assert "could not be resolved" in result.stderr + result.stdout


def test_the_probe_list_contains_the_panel_plus_the_subset(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    panel = {r["image_id"] for r in csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8"))}
    probe = {r["image_id"] for r in csv.DictReader(
        (fixture["reports"] / "p2_probe_images.csv").open(encoding="utf-8"))}
    assert panel <= probe
    assert len(probe) > len(panel)


def test_the_t_ctrl_diagnostic_panel_is_written(fixture: dict[str, Path]) -> None:
    """It has no eval manifest, so this script draws it through the same sampler."""
    assert _run(fixture).returncode == 0
    path = fixture["reports"] / "p2_panel_t_ctrl_diagnostic.csv"
    assert path.is_file()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows and all(r["panel_set_id"] == "t_ctrl_diagnostic" for r in rows)
    # Both halves: lossless reals and reconstructions of those same reals.
    labels = {r["label"] for r in rows}
    assert labels == {"real", "fake"}, labels
    sources = {r["source"] for r in rows if r["label"] == "fake"}
    ctrl = {r["image_id"] for r in csv.DictReader(
        (fixture["reports"] / "p1_reals_manifest_t_ctrl.csv").open(encoding="utf-8"))}
    assert sources <= ctrl, "a non-T-ctrl reconstruction reached the lossless diagnostic"


def test_no_row_is_listed_twice_even_though_the_panel_overlaps_the_subset(
    fixture: dict[str, Path],
) -> None:
    """T-ctrl rows are both panel rows and training reals; they must appear once."""
    assert _run(fixture).returncode == 0
    ids = [r["image_id"] for r in csv.DictReader(
        (fixture["reports"] / "p2_probe_images.csv").open(encoding="utf-8"))]
    assert len(ids) == len(set(ids))
    summary = json.loads((fixture["reports"] / "p2_image_lists_summary.json").read_text())
    assert summary["s3_subset"]["already_in_panel"] >= 0


def test_the_run_is_reproducible(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    first = (fixture["reports"] / "p2_probe_images.csv").read_bytes()
    assert _run(fixture).returncode == 0
    assert (fixture["reports"] / "p2_probe_images.csv").read_bytes() == first


def test_a_different_seed_changes_the_subset(fixture: dict[str, Path]) -> None:
    """Caps below the population, or the seed has nothing to choose between."""
    small = {"train_reals": 8, "train_fakes_per_protocol": 8, "val_cap": 8}
    assert _run(fixture, **small).returncode == 0
    first = (fixture["reports"] / "p2_probe_images.csv").read_bytes()
    assert _run(fixture, seed=99, **small).returncode == 0
    assert (fixture["reports"] / "p2_probe_images.csv").read_bytes() != first


def test_the_whole_population_is_taken_when_the_cap_exceeds_it(
    fixture: dict[str, Path],
) -> None:
    """With caps above the population the seed is irrelevant, and that is correct."""
    assert _run(fixture).returncode == 0
    first = (fixture["reports"] / "p2_probe_images.csv").read_bytes()
    assert _run(fixture, seed=99).returncode == 0
    assert (fixture["reports"] / "p2_probe_images.csv").read_bytes() == first


def test_the_subset_caps_are_honoured(fixture: dict[str, Path]) -> None:
    assert _run(fixture, train_reals=5, train_fakes_per_protocol=6, val_cap=4).returncode == 0
    summary = json.loads((fixture["reports"] / "p2_image_lists_summary.json").read_text())
    subset = summary["s3_subset"]
    assert subset["train_reals"] == 5
    assert all(n == 6 for n in subset["train_fakes_per_protocol"].values())
    assert subset["val"] == 4


def test_a_missing_split_file_stops_the_run(fixture: dict[str, Path]) -> None:
    (fixture["reports"] / "p2_splits.csv").unlink()
    result = _run(fixture)
    assert result.returncode == 2
    assert "p2_build_splits.py first" in result.stderr + result.stdout


def test_a_missing_root_stops_the_run(fixture: dict[str, Path]) -> None:
    argv = [
        sys.executable, str(SCRIPTS / "p2_build_image_lists.py"),
        "--root", str(fixture["root"] / "nope"),
        "--manifests", str(fixture["reports"]),
        "--splits", str(fixture["reports"] / "p2_splits.csv"),
        "--pairs-manifest", str(fixture["reports"] / "p1_pairs_manifest.csv"),
    ]
    result = subprocess.run(argv, capture_output=True, text=True, cwd=THESIS_DBP)
    assert result.returncode == 2


def test_each_protocol_draws_its_fakes_from_its_own_members(fixture: dict[str, Path]) -> None:
    """p_a_sd1 may only see the SD1 lineage; full_bank sees all six fixture members."""
    result = _run(fixture)
    assert result.returncode == 0
    summary = json.loads((fixture["reports"] / "p2_image_lists_summary.json").read_text())
    per_protocol = summary["s3_subset"]["train_fakes_per_protocol"]
    assert set(per_protocol) == {"p_a_sd1", "4ch_only", "full_bank"}
    assert all(n > 0 for n in per_protocol.values())


# ------------------------------------------------------------- two roots ---


def test_every_row_names_the_root_its_path_is_relative_to(fixture: dict[str, Path]) -> None:
    """Corpus and panel are different directories; one --root cannot resolve both."""
    assert _run(fixture).returncode == 0
    for name in ("p2_s1_images.csv", "p2_panel_images.csv", "p2_probe_images.csv"):
        rows = list(csv.DictReader((fixture["reports"] / name).open(encoding="utf-8")))
        assert rows, name
        assert all(r["root"] in {"corpus", "panel"} for r in rows), name

    probe = list(csv.DictReader(
        (fixture["reports"] / "p2_probe_images.csv").open(encoding="utf-8")))
    roots = {r["root"] for r in probe}
    assert roots == {"corpus", "panel"}, roots
    # S1's training universe is entirely corpus; the panel list entirely panel.
    assert {r["root"] for r in csv.DictReader(
        (fixture["reports"] / "p2_s1_images.csv").open(encoding="utf-8"))} == {"corpus"}
    assert {r["root"] for r in csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8"))} == {"panel"}


def test_a_materialised_panel_list_supplies_the_panel_paths(
    fixture: dict[str, Path],
) -> None:
    """--panel-list is what makes the panel rows portable between sessions."""
    panel_list = fixture["reports"] / "materialised.csv"
    _write(
        panel_list,
        ["image_id", "relative_path", "label", "view", "sha256"],
        [{"image_id": f"e8_{i}", "relative_path": f"e8_evalgen/e8_{i}.png",
          "label": "fake", "view": "e8_evalgen", "sha256": "0" * 64} for i in range(12)],
    )
    assert _run(fixture, panel_list=str(panel_list)).returncode == 0
    rows = list(csv.DictReader(
        (fixture["reports"] / "p2_panel_images.csv").open(encoding="utf-8")))
    assert len(rows) == 12
    assert all(r["relative_path"].startswith("e8_evalgen/") for r in rows)
    assert all(r["root"] == "panel" for r in rows)


def test_the_diagnostic_panel_carries_a_hash_for_every_row(
    fixture: dict[str, Path],
) -> None:
    """p2_materialise_panel.py verifies every copy against this; a blank blocks it."""
    assert _run(fixture).returncode == 0
    rows = list(csv.DictReader(
        (fixture["reports"] / "p2_panel_t_ctrl_diagnostic.csv").open(encoding="utf-8")))
    assert rows
    blank = [r["image_id"] for r in rows if not (r.get("sha256") or "").strip()]
    assert not blank, f"{len(blank)} row(s) with no sha256, e.g. {blank[:3]}"
    assert all(len(r["sha256"]) == 64 for r in rows)


def test_a_manifest_without_a_hash_column_stops_the_run(fixture: dict[str, Path]) -> None:
    """Better to fail here than to produce a panel that cannot be frozen."""
    path = fixture["reports"] / "p1_pairs_manifest.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for row in rows:
        row["sha256"] = ""
    _write(path, list(rows[0]), rows)
    result = _run(fixture)
    assert result.returncode == 1
    assert "carry no sha256" in result.stderr + result.stdout


# ------------------------------- diagnostic-only sets and the luma companion ---
# B1: T-ctrl is measured, never trained on. It had been reaching every training list
# because `corpus` is the population for both the lists and the diagnostic panel, and
# the split label on a T-ctrl row says "train" like any other.


def _ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {row["image_id"] for row in csv.DictReader(handle)}


def _ctrl_ids(fixture: dict[str, Path]) -> set[str]:
    with (fixture["reports"] / "p1_reals_manifest_t_ctrl.csv").open(encoding="utf-8") as fh:
        reals = {row["image_id"] for row in csv.DictReader(fh)}
    with (fixture["reports"] / "p1_pairs_manifest.csv").open(encoding="utf-8") as fh:
        pairs = {row["pair_id"] for row in csv.DictReader(fh) if row["set_id"] == "t_ctrl"}
    return reals | pairs


def test_t_ctrl_never_reaches_the_s1_training_list(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    assert not (_ids(fixture["reports"] / "p2_s1_images.csv") & _ctrl_ids(fixture))


def test_t_ctrl_never_reaches_the_probe_training_subset(fixture: dict[str, Path]) -> None:
    """The panel half may hold T-ctrl rows; the subset half may not."""
    assert _run(fixture).returncode == 0
    panel = _ids(fixture["reports"] / "p2_panel_images.csv")
    with (fixture["reports"] / "p2_probe_images.csv").open(encoding="utf-8") as handle:
        subset = {r["image_id"] for r in csv.DictReader(handle) if r["image_id"] not in panel}
    assert not (subset & _ctrl_ids(fixture))


def test_t_ctrl_still_populates_the_diagnostic_panel(fixture: dict[str, Path]) -> None:
    """Withholding it from training must not empty the panel it exists for."""
    assert _run(fixture).returncode == 0
    rows = list(csv.DictReader(
        (fixture["reports"] / "p2_panel_t_ctrl_diagnostic.csv").open(encoding="utf-8")))
    assert rows
    assert _ids(fixture["reports"] / "p2_panel_t_ctrl_diagnostic.csv") <= _ctrl_ids(fixture)


def test_the_withheld_count_is_reported(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    summary = json.loads((fixture["reports"] / "p2_image_lists_summary.json").read_text())
    withheld = summary["diagnostic_only"]
    assert withheld["sets"] == ["t_ctrl"]
    assert withheld["withheld_rows"] == len(_ctrl_ids(fixture))
    assert withheld["withheld_reals"] + withheld["withheld_reconstructions"] == \
        withheld["withheld_rows"]


def _luma(fixture: dict[str, Path], ids: list[str]) -> Path:
    path = fixture["reports"] / "p2_panel_images_luma.csv"
    _write(path, ["image_id", "relative_path", "label", "view", "sha256"],
           [{"image_id": i, "relative_path": f"e7_dda_coco_luma/{i}.png",
             "label": "fake", "view": "e7_dda_coco_luma", "sha256": f"{n:064d}"}
            for n, i in enumerate(ids)])
    return path


def test_luma_rows_join_the_probe_list_under_a_distinct_id(fixture: dict[str, Path]) -> None:
    """p2_probe_features.py keys one .npz per member by image id, so a shared id
    would overwrite the standard row's reconstructions with the greyscale ones."""
    path = _luma(fixture, ["e8_0", "e8_1"])
    assert _run(fixture, luma_list=path).returncode == 0
    probe = _ids(fixture["reports"] / "p2_probe_images.csv")
    assert {"e8_0__luma", "e8_1__luma"} <= probe
    assert _ids(fixture["reports"] / "p2_luma_images.csv") == {"e8_0__luma", "e8_1__luma"}


def test_luma_rows_stay_out_of_the_frozen_panel_list(fixture: dict[str, Path]) -> None:
    """The panel digest is defined over p2_panel_images.csv; luma must not change it."""
    path = _luma(fixture, ["e8_0"])
    assert _run(fixture, luma_list=path).returncode == 0
    assert not any(i.endswith("__luma")
                   for i in _ids(fixture["reports"] / "p2_panel_images.csv"))


def test_luma_rows_are_marked_as_panel_rooted(fixture: dict[str, Path]) -> None:
    path = _luma(fixture, ["e8_0"])
    assert _run(fixture, luma_list=path).returncode == 0
    with (fixture["reports"] / "p2_luma_images.csv").open(encoding="utf-8") as handle:
        assert all(row["root"] == "panel" for row in csv.DictReader(handle))


def test_an_undeclared_luma_view_is_refused(fixture: dict[str, Path]) -> None:
    path = fixture["reports"] / "bad_luma.csv"
    _write(path, ["image_id", "relative_path", "label", "view", "sha256"],
           [{"image_id": "e8_0", "relative_path": "x/0.png", "label": "fake",
             "view": "e9_invented_luma", "sha256": "0" * 64}])
    result = _run(fixture, luma_list=path)
    assert result.returncode == 1
    assert "not a declared luma view" in result.stdout + result.stderr


def test_without_a_luma_list_no_luma_rows_appear(fixture: dict[str, Path]) -> None:
    assert _run(fixture).returncode == 0
    assert not (fixture["reports"] / "p2_luma_images.csv").is_file()
    summary = json.loads((fixture["reports"] / "p2_image_lists_summary.json").read_text())
    assert summary["luma"]["rows"] == 0
