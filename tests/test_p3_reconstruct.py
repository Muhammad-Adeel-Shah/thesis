"""P3 step 4a tooling: the generic shard worker, the output-size gate, the final check."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _load(name: str):
    alias = f"_p3_recon_{name}"
    spec = importlib.util.spec_from_file_location(alias, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


SHARDS = _load("p2_run_shards")
CHECK = _load("p3_oi_pairs_check")


def test_generic_worker_gets_members_device_and_only_the_given_arguments(tmp_path) -> None:
    job = SHARDS.parse_job("1:sdxl,dcae", 0, tmp_path, "pairs")
    args = argparse.Namespace(worker=Path("scripts/p1_build_pairs.py"),
                              worker_arg=["--name", "p3_oi_pairs", "--limit", "32"],
                              images=None, root=None, panel_root=None, cache_dir=None)
    command, env = SHARDS.command_for(job, args)
    assert command[2:] == [str(Path("scripts/p1_build_pairs.py")), "--members", "sdxl,dcae",
                           "--device", "cuda", "--name", "p3_oi_pairs", "--limit", "32"]
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert job.log_path.name == "pairs_job0_gpu1.log"


def test_probe_worker_command_is_unchanged(tmp_path) -> None:
    job = SHARDS.parse_job("0:movq_k22", 0, tmp_path)
    args = argparse.Namespace(worker=Path("w.py"), worker_arg=[], images=Path("i.csv"),
                              root=Path("r"), panel_root=None, cache_dir=None)
    command, _ = SHARDS.command_for(job, args)
    assert command[2:] == ["w.py", "--images", "i.csv", "--root", "r", "--members",
                           "movq_k22", "--device", "cuda", "--tag", "job0_gpu0"]
    assert job.log_path.name == "probe_job0_gpu0.log"


def _summary(path: Path, members: dict[str, tuple[int, int, str]]) -> Path:
    path.write_text(json.dumps({"members": [
        {"member": m, "written": w, "bytes_written": b, "status": s}
        for m, (w, b, s) in members.items()]}), encoding="utf-8")
    return path


def test_size_gate_projects_measured_means_and_refuses_what_does_not_fit(tmp_path) -> None:
    gb = CHECK.GB
    s0 = _summary(tmp_path / "s0.json", {"a": (32, 32 * 250_000, "OK")})
    s1 = _summary(tmp_path / "s1.json", {"b": (32, 32 * 200_000, "OK")})
    ok = CHECK.project([s0, s1], ["a", "b"], 4000, free_bytes=int(19.45 * gb), margin_gb=2.0)
    assert ok["ok"] and ok["projected_bytes"] == 4000 * 450_000
    assert ok["members"]["a"]["mean_bytes"] == 250_000
    tight = CHECK.project([s0, s1], ["a", "b"], 4000, free_bytes=int(3.5 * gb), margin_gb=2.0)
    assert not tight["ok"] and tight["projected_gb"] > tight["limit_gb"]
    missing = CHECK.project([s0], ["a", "b"], 4000, free_bytes=int(19 * gb), margin_gb=2.0)
    assert not missing["ok"] and missing["problems"] == ["b: no sample"]
    bad = _summary(tmp_path / "s2.json", {"b": (0, 0, "ERROR")})
    failed = CHECK.project([s0, bad], ["a", "b"], 4000, free_bytes=int(19 * gb), margin_gb=2)
    assert not failed["ok"] and failed["problems"] == ["b: sample ERROR"]


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_inputs_and_final_checks(tmp_path) -> None:
    from dbp.data.manifest import file_sha256

    reals_root = tmp_path / "reals"
    (reals_root / "t_oi").mkdir(parents=True)
    rows = []
    for image_id in ("x1", "x2"):
        path = reals_root / "t_oi" / f"{image_id}.png"
        path.write_bytes(image_id.encode() * 10)
        rows.append({"image_id": image_id, "set_id": "t_oi", "split": "train",
                     "sha256": file_sha256(path)})
    manifest = _write(tmp_path / "reals.csv", rows)
    assert CHECK.check_inputs(manifest, reals_root, 2)["ok"]
    (reals_root / "t_oi" / "x2.png").write_bytes(b"changed")
    bad = CHECK.check_inputs(manifest, reals_root, 2)
    assert not bad["ok"] and bad["sha256_mismatch"] == ["x2"]

    bench = _write(tmp_path / "bench.csv", [
        {"key": "sd15", "fp16_build_allowed": "True"},
        {"key": "sdxl", "fp16_build_allowed": "False"}])
    pairs, out = [], tmp_path / "work"
    for image_id in ("x1", "x2"):
        for member, precision in (("sd15", "fp16"), ("sdxl", "fp32")):
            path = out / "oi_pairs" / member / f"{image_id}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{image_id}{member}".encode())
            pairs.append({"image_id": image_id, "member": member, "precision": precision,
                          "crop": "384", "relative_path": f"oi_pairs/{member}/{image_id}.png",
                          "sha256": file_sha256(path)})
    merged = _write(tmp_path / "pairs.csv", pairs)
    good = CHECK.check_final(merged, out, manifest, bench, ["sd15", "sdxl"], 384)
    assert good["ok"] and good["rows"] == 4 and good["members"]["sd15"]["rows"] == 2
    wrong = _write(tmp_path / "wrong.csv", [{**pairs[0], "precision": "fp32"}, *pairs[1:3]])
    result = CHECK.check_final(wrong, out, manifest, bench, ["sd15", "sdxl"], 384)
    assert not result["ok"]
    assert any("precision differs from P1's bench for ['sd15']" in p for p in result["problems"])
    assert any("1 (real, member) pair(s) missing" in p for p in result["problems"])


def test_p1_pair_names_are_unchanged_by_default() -> None:
    text = (ROOT / "scripts" / "p1_build_pairs.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--name", default="p1_pairs"' in text
    assert 'f"{name}_manifest_{spec.key}.csv"' in text and 'f"{name}_manifest.csv"' in text
    # Regression (4a dry-run, 2026-10-17): member-list jobs all run as shard 0, and their
    # summaries overwrote each other until they were named by their members.
    tag = 'tag = "+".join(s.key for s in specs) if args.members else f"shard{args.shard}"'
    assert tag in text
    runbook = (ROOT / "env" / "kaggle_run_p3_reconstruct.py").read_text(encoding="utf-8")
    assert '_sample_summary_*.json' in runbook and "_sample_summary_shard" not in runbook


def test_the_4a_expect_file_matches_the_committed_bench_and_bank() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from dbp.decoders.registry import load_bank_config

    expect = json.loads((ROOT / "env" / "p3_reconstruct_expect.json").read_text("utf-8"))
    bank = load_bank_config(ROOT / "configs" / "decoder_bank.yaml")
    members = [m for job in expect["jobs"] for m in job.split(":", 1)[1].split(",")]
    assert sorted(members) == sorted(s.key for s in bank.distinct_members())
    assert {job.split(":")[0] for job in expect["jobs"]} == {"0", "1"}
    assert (ROOT / expect["throughput_csv"]).is_file()
    assert (ROOT / expect["reals_manifest"]).is_file()
    assert expect["n_reals"] == 4000 and expect["name"] == "p3_oi_pairs"


@pytest.mark.parametrize("name", ["kaggle_run_p3_reconstruct.py", "kaggle_run_p3_train.py"])
def test_new_runbooks_follow_the_rules(name) -> None:
    text = (ROOT / "env" / name).read_text(encoding="utf-8")
    assert "import dbp" not in text and "from dbp" not in text
    assert not any(line.lstrip().startswith(("!", "%")) for line in text.splitlines()
                   if not line.startswith("#"))
    assert "kaggle_bootstrap.sh" in text and "pip install" not in text


def test_the_4a_runbook_finds_the_reals_by_layout_not_by_name() -> None:
    # Regression (Kaggle, 2026-10-17): the Open Images output was published under another
    # name ("dbp-p3-lolo_data") and Cell 1's name-based lookup found nothing.
    text = (ROOT / "env" / "kaggle_run_p3_reconstruct.py").read_text(encoding="utf-8")
    assert 'find("datasets/*/*/oi/reals")' in text
    assert "dbp-p3-open-images/oi/reals" not in text
