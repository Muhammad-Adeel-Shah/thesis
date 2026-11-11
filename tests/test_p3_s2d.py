"""S2-D inputs (2026-10-21): pooled by real, verified; the held-out reals never enter training."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "p3_s2d_inputs.py"
RUNBOOK = ROOT / "env" / "kaggle_run_p3_s2d.py"
EXPECT = ROOT / "env" / "p3_s2d_expect.json"
MEMBERS = ["sd15", "sd21", "sd_ft_ema", "sdxl", "sdxl_fp16fix", "sd35", "flux", "dcae",
           "realesrgan_x4", "flux2_klein", "qwen_image", "movq_k22"]
#: git's sha256 of scripts/p2_train_s2.py as last changed in 1bc18e2 (user, 2026-10-21:
#: "the trainer must stay byte-identical"). S2-D reaches it through input files only.
TRAINER_SHA256 = "6b9c30c4992e1334c265aa62f40b91b58634daf8900a73a33cdc0952bbedb7b6"


def _module():
    spec = importlib.util.spec_from_file_location("_s2d_inputs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_s2d_inputs"] = module
    spec.loader.exec_module(module)
    return module


def _csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _png(path: Path, seed: str) -> str:
    """Write stand-in bytes (only their hash is checked) and return their sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = hashlib.sha256(seed.encode()).digest() * 4
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp: Path, chosen_p: float = 0.35) -> dict[str, Path]:
    """COCO: 2 train reals + 1 val real; Open Images: 3 train + 1 held out; 12 members each."""
    m = _module()
    corpus = tmp / "p1"
    manifests = tmp / "manifests"
    real_cols = ["image_id", "set_id", "label", "relative_path", "sha256", "excluded_reason"]
    coco = [{"image_id": f"c{i}", "set_id": "t_real", "label": "real",
             "relative_path": f"reals/t_real/c{i}.png", "sha256": "x", "excluded_reason": ""}
            for i in range(3)]
    _csv(manifests / "p1_reals_manifest_t_real.csv", real_cols, coco)
    _csv(manifests / "p1_reals_manifest_t_ctrl.csv", real_cols,
         [{"image_id": "k0", "set_id": "t_ctrl", "label": "real",
           "relative_path": "reals/t_ctrl/k0.png", "sha256": "x", "excluded_reason": ""}])
    _csv(manifests / "p1_gen_matched_manifest.csv", ["gen_id", "jpg_path"],
         [{"gen_id": "g0", "jpg_path": "gen_matched/g0.jpg"}])
    _png(corpus / "gen_matched" / "g0.jpg", "g0")
    for i in range(3):
        _png(corpus / "reals" / "t_real" / f"c{i}.png", f"c{i}")
        for mem in MEMBERS:
            _png(corpus / "pairs" / mem / f"c{i}.png", f"c{i}{mem}")
    pair_cols = ["pair_id", "image_id", "set_id", "member", "relative_path", "sha256"]
    coco_pairs = [{"pair_id": f"c{i}__{mem}", "image_id": f"c{i}", "set_id": "t_real",
                   "member": mem, "relative_path": f"pairs/{mem}/c{i}.png", "sha256": "x"}
                  for i in range(3) for mem in MEMBERS]
    _csv(tmp / "coco" / "p1_pairs_manifest.csv", pair_cols, coco_pairs)
    split_cols = ["row_id", "set_id", "kind", "content_id", "group_id", "stratum", "split"]
    side = {"c0": "train", "c1": "train", "c2": "val"}
    splits = [{"row_id": f"c{i}", "set_id": "t_real", "kind": "real", "content_id": f"c{i}",
               "group_id": f"c{i}", "stratum": "x", "split": side[f"c{i}"]} for i in range(3)]
    splits += [{"row_id": r["pair_id"], "set_id": f"pair_{r['member']}", "kind": "fake",
                "content_id": r["image_id"], "group_id": r["image_id"], "stratum": "x",
                "split": side[r["image_id"]]} for r in coco_pairs]
    splits.append({"row_id": "g0", "set_id": "gen_matched", "kind": "fake", "content_id": "g0",
                   "group_id": "g0", "stratum": "x", "split": "train"})
    _csv(tmp / "coco" / "p2_splits.csv", split_cols, splits)

    oi = {f"o{i}": ("train" if i < 3 else "held_out") for i in range(4)}
    oi_rows = [{"image_id": i, "set_id": "t_oi", "label": "real",
                "relative_path": f"reals/t_oi/{i}.png", "oi_image_id": f"src{i}",
                "sha256": _png(corpus / "oi_reals" / "t_oi" / f"{i}.png", i),
                "excluded_reason": "", "split": s} for i, s in oi.items()]
    _csv(tmp / "oi_manifest.csv", real_cols + ["oi_image_id", "split"], oi_rows)
    oi_pairs = [{"pair_id": f"{i}__{mem}", "image_id": i, "set_id": "t_oi", "member": mem,
                 "relative_path": f"oi_pairs/{mem}/{i}.png",
                 "sha256": _png(corpus / "oi_pairs" / mem / f"{i}.png", f"{i}{mem}")}
                for i in oi for mem in MEMBERS]
    _csv(tmp / "oi_pairs_manifest.csv", pair_cols, oi_pairs)
    (tmp / "digest.json").write_text(json.dumps({"manifest": {
        "rows": len(oi_pairs), "rows_per_member": {mem: len(oi) for mem in sorted(MEMBERS)},
        "content_sha256": m.content_digest(oi_pairs)}}), encoding="utf-8")
    (tmp / "choice.json").write_text(json.dumps({"chosen": "s2b_deg_p35",
                                                 "chosen_p": chosen_p}), encoding="utf-8")
    (tmp / "expect.json").write_text(json.dumps({
        "oi": {"reals_manifest_sha256": hashlib.sha256(
            (tmp / "oi_manifest.csv").read_bytes()).hexdigest(),
            "train_reals": 3, "held_out_reals": 1},
        "queue_template": ["s2d:0", "s2b_reals:0", "s2d:1"]}), encoding="utf-8")
    return {"tmp": tmp, "corpus": corpus, "manifests": manifests, "merged": tmp / "merged",
            "local": tmp / "local"}


def _run(paths: dict[str, Path], mode: str, monkeypatch: pytest.MonkeyPatch,
         *extra: str) -> int:
    tmp = paths["tmp"]
    argv = [str(SCRIPT), mode, "--expect", str(tmp / "expect.json"),
            "--oi-manifest", str(tmp / "oi_manifest.csv"), "--out-dir", str(paths["merged"]),
            "--built", str(tmp / "built.json"),
            "--out", str(tmp / ("built.json" if mode == "build" else "check.json"))]
    if mode == "build":
        argv += ["--oi-pairs-manifest", str(tmp / "oi_pairs_manifest.csv"),
                 "--oi-pairs-digest", str(tmp / "digest.json"),
                 "--local-root", str(paths["local"]),
                 "--pairs-manifest", str(tmp / "coco" / "p1_pairs_manifest.csv"),
                 "--splits", str(tmp / "coco" / "p2_splits.csv"),
                 "--manifests", str(paths["manifests"]), "--choice", str(tmp / "choice.json"),
                 "--root", str(paths["corpus"]), "--queue-out", str(tmp / "queue.json"),
                 "--workers", "4"]
    else:
        argv += ["--local-root", str(paths["local"])]
    monkeypatch.setattr(sys, "argv", argv + list(extra))
    return _module().main()


def test_build_pools_the_reals_and_resolves_the_queue(tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    assert _run(paths, "build", monkeypatch) == 0
    record = json.loads((tmp_path / "built.json").read_text("utf-8"))
    pairs = _read(paths["merged"] / "p1_pairs_manifest.csv")
    coco = _read(tmp_path / "coco" / "p1_pairs_manifest.csv")
    assert pairs[: len(coco)] == coco, "COCO rows must be unchanged and first"
    assert {r["image_id"] for r in pairs[len(coco):]} == {"o0", "o1", "o2"}
    assert len(pairs) == len(coco) + 3 * len(MEMBERS)
    reals = _read(paths["merged"] / "manifests" / "p1_reals_manifest_t_real.csv")
    assert [r["relative_path"] for r in reals[3:]] == [f"oi_reals/t_oi/o{i}.png" for i in range(3)]
    # The queue comes from the tuning result (p = 0.35), never typed.
    assert json.loads((tmp_path / "queue.json").read_text("utf-8"))["queue"] == [
        "s2d_p35:0", "s2b_reals:0", "s2d_p35:1"]
    # The pool, as the trainer's own TrainingData builds it: 2 COCO + 3 Open Images train reals.
    for run in ("s2d_p35", "s2b_reals"):
        pool = record["pools"][run]
        assert (pool["coco_reals"], pool["oi_reals"]) == (2, 3)
        assert pool["oi_share_of_reals"] == 0.6
    assert record["files_verified"]["files"] == 3 + 3 * len(MEMBERS)
    # Staging: every file the runs read -- COCO train+val pairs and reals, gen_matched, the
    # Open Images train files -- copied once to local disk, byte-identical.
    staged = record["staged"]
    coco = 3 * len(MEMBERS) + 3
    assert staged["files"] == coco + 1 + 3 + 3 * len(MEMBERS)
    assert staged["by_source"]["open_images"]["files"] == 3 + 3 * len(MEMBERS)
    assert staged["by_source"]["coco"]["files"] == coco
    assert staged["by_source"]["gen_matched"]["files"] == 1
    for rel in ("pairs/flux/c2.png", "oi_pairs/sd15/o1.png", "oi_reals/t_oi/o0.png",
                "gen_matched/g0.jpg"):
        assert (paths["local"] / rel).read_bytes() == (paths["corpus"] / rel).read_bytes()
    assert staged["seconds"] is not None and staged["bytes"] > 0
    assert record["merged"]["held_out_excluded"] == {"oi_reals": 1, "oi_pair_rows": len(MEMBERS)}
    assert _run(paths, "check", monkeypatch) == 0


def test_the_held_out_reals_are_never_read_or_merged(tmp_path: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    (paths["corpus"] / "oi_reals" / "t_oi" / "o3.png").unlink()
    for mem in MEMBERS:
        (paths["corpus"] / "oi_pairs" / mem / "o3.png").unlink()
    assert _run(paths, "build", monkeypatch) == 0, "a held-out file was read"
    merged = paths["merged"]
    for name in ("p1_pairs_manifest.csv", "p2_splits.csv",
                 "manifests/p1_reals_manifest_t_real.csv"):
        assert "o3" not in (merged / name).read_text("utf-8"), name
    assert not list(paths["local"].rglob("o3.png"))
    assert _run(paths, "check", monkeypatch) == 0
    check = json.loads((tmp_path / "check.json").read_text("utf-8"))
    assert check["held_out_rows_found"]["local copy"] == 0


def test_the_preflight_refuses_a_held_out_row(tmp_path: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    assert _run(paths, "build", monkeypatch) == 0
    target = paths["merged"] / "p1_pairs_manifest.csv"
    with target.open("a", encoding="utf-8", newline="") as handle:
        handle.write("o3__sd15,o3,t_oi,sd15,oi_pairs/sd15/o3.png,x\n")
    # Even with the build record made to agree, the held-out id itself is refused.
    built = json.loads((tmp_path / "built.json").read_text("utf-8"))
    built["merged"]["sha256"]["p1_pairs_manifest.csv"] = hashlib.sha256(
        target.read_bytes()).hexdigest()
    (tmp_path / "built.json").write_text(json.dumps(built), encoding="utf-8")
    assert _run(paths, "check", monkeypatch) == 1
    check = json.loads((tmp_path / "check.json").read_text("utf-8"))
    assert check["held_out_rows_found"]["p1_pairs_manifest.csv"] == 1


def test_a_corrupted_train_file_is_refused_before_anything_is_written(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    (paths["corpus"] / "oi_pairs" / "flux" / "o1.png").write_bytes(b"not the file")
    assert _run(paths, "build", monkeypatch) == 1
    record = json.loads((tmp_path / "built.json").read_text("utf-8"))
    assert record["files_verified"]["n_sha256_mismatch"] == 1
    assert not (paths["merged"] / "p1_pairs_manifest.csv").exists()
    assert not (tmp_path / "queue.json").exists()


def test_a_pairs_manifest_off_the_committed_digest_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    rows = _read(tmp_path / "oi_pairs_manifest.csv")
    rows[0]["sha256"] = "0" * 64
    _csv(tmp_path / "oi_pairs_manifest.csv", list(rows[0]), rows)
    assert _run(paths, "build", monkeypatch) == 1
    record = json.loads((tmp_path / "built.json").read_text("utf-8"))
    assert any("content digest" in p for p in record["problems"])


def test_a_tuning_result_without_a_config_entry_is_refused(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path, chosen_p=0.5)
    assert _run(paths, "build", monkeypatch) == 1
    record = json.loads((tmp_path / "built.json").read_text("utf-8"))
    assert any("no run 's2d_p50'" in p for p in record["problems"])
    assert not (tmp_path / "queue.json").exists()


def test_the_trainer_is_byte_identical() -> None:
    """User, 2026-10-21: S2-D changes the inputs, never the trainer."""
    data = (ROOT / "scripts" / "p2_train_s2.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(data).hexdigest() == TRAINER_SHA256


def test_the_runbook_checks_the_held_out_exclusion_before_training() -> None:
    code = RUNBOOK.read_text(encoding="utf-8")
    build = code.index('py("p3_s2d_inputs.py", "build"')
    check = code.index('py("p3_s2d_inputs.py", "check"')
    smoke = code.index('"--smoke", EXPECT["smoke_steps"]')
    train = code.index('code = py("p2_run_queue.py"')
    assert code.index('py("p2_preflight.py"') < build < check < smoke < train
    # The trainer reads the merged files, and nothing else adds reals.
    assert 'MERGED / "p1_pairs_manifest.csv"' in code and 'MERGED / "p2_splits.csv"' in code
    assert "--manifests={MERGED / 'manifests'}" in code
    # p is read from the committed tuning result, never typed into the runbook.
    assert "s2d_p70" not in code and "s2d_p35" not in code
    expect = json.loads(EXPECT.read_text(encoding="utf-8"))
    assert expect["choice"] == "reports/p3_degradation_choice.json"
    assert expect["queue_template"] == ["s2d:0", "s2b_reals:0", "s2d:1"]
    assert expect["oi"]["train_reals"] == 3600 and expect["oi"]["held_out_reals"] == 400


def test_the_runbook_logs_the_hardware_and_the_throughput() -> None:
    """User, 2026-10-21: the slowdown is diagnosed from the host, in the cells only."""
    for name in ("kaggle_run_p3_s2d.py", "kaggle_run_p3_tune.py", "kaggle_run_p3_train.py"):
        code = (ROOT / "env" / name).read_text(encoding="utf-8")
        assert code.count("hardware(DBP / \"reports\"") == 1, name
        assert "power.draw" in code and "temperature.gpu" in code and "clocks.sm" in code
        assert 'print("THROUGHPUT |"' in code, name


def test_a_copy_that_does_not_fit_is_refused_before_anything_is_written(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fixture(tmp_path)
    assert _run(paths, "build", monkeypatch, "--margin-gb", "1e9") == 1
    record = json.loads((tmp_path / "built.json").read_text("utf-8"))
    assert "does not fit" in record["staged"]["reason"]
    assert record["staged"]["bytes"] > 0 and "free_gb_before" in record["staged"]
    assert not (paths["merged"] / "p1_pairs_manifest.csv").exists()
    assert not (tmp_path / "queue.json").exists()
    assert not list(paths["local"].rglob("*.png"))


def test_the_trainer_reads_the_local_copy() -> None:
    code = RUNBOOK.read_text(encoding="utf-8")
    assert 'QUEUE = ["--devices", EXPECT["devices"], "--root", LOCAL,' in code
    assert code.count('"--local-root", LOCAL') == 2  # build and the pre-flight check
    assert "staging {staged.get('files')} files" in code


def test_absolute_gen_matched_paths_are_counted_not_copied(tmp_path: Path) -> None:
    """The matcher writes gen_matched to local disk with absolute paths in its manifest."""
    m = _module()
    root, local = tmp_path / "root", tmp_path / "local"
    gen = tmp_path / "work" / "gen_matched" / "g0.jpg"
    _png(gen, "g0")
    rel = "pairs/sd15/c0.png"
    _png(root / rel, "c0")

    class Log:
        def info(self, *args: object) -> None:
            return None

    staged = m.stage([rel, str(gen)], {}, root, local, 2, 0.0, Log())
    assert staged["ok"] and staged["already_local"] == 1
    assert staged["by_source"]["gen_matched"]["files"] == 1
    assert [p.relative_to(local).as_posix() for p in local.rglob("*") if p.is_file()] == [rel]
